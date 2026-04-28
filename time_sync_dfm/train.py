#!/usr/bin/env python3
"""Stage 13: Time-Synchronous DFM with Direction-Magnitude Decoupling.

Stage 12 revealed that the velocity net's direction predictions generalise
well (cosine ~0.75+ on validation) but the magnitude overfits badly
(train vpred_l2 stable at ~46, valid vpred_l2 swings 50-79). This caused
flow PER to oscillate between 17 and 44 depending on the epoch.

The root cause: piecewise target velocities have large, utterance-specific
magnitudes (~51 L2) because they divide by segment duration. Short phones
produce very large velocities, long phones produce small ones. The velocity
net memorises these patterns on training data but produces erratic magnitudes
on unseen utterances.

This version applies three complementary fixes:

Fix 1 - Normalised training targets (removes magnitude from the learning
problem entirely):
  Target velocities are normalised to unit length during training. The
  velocity net only needs to learn WHICH DIRECTION to push each frame's
  distribution, not how far. This makes the targets consistent across
  segments regardless of phone duration, eliminating the main source of
  overfitting.

Fix 2 - Regularised velocity net (reduces capacity to prevent memorisation):
  Back to 512 hidden (from 768), 0.1 dropout (from 0.05), but keeps the
  three conv layers and GroupNorm from Stage 12. Also adds explicit L2
  weight decay on velocity net parameters.

Fix 3 - Fixed magnitude at inference (ensures stable corrections):
  During inference, predicted velocities are normalised to unit length
  and then scaled by a fixed velocity_inference_scale parameter. This
  guarantees consistent correction magnitudes across all utterances.

The result: the velocity net focuses purely on direction prediction
(which it already does well), and the correction magnitude is controlled
by a single tunable hyperparameter instead of learned per-utterance.
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.utils.distributed import if_main_process, run_on_main
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


class VelocityNet(nn.Module):
    """Direction-focused velocity field with regularised temporal context.

    This version is designed to predict velocity DIRECTIONS only, not
    magnitudes. The training targets are unit-normalised, so the net
    learns which direction to push each frame's distribution without
    needing to predict how far.

    Architecture choices for generalisation:
      - Three conv layers (receptive field 13 frames / ~130ms) give
        broad context for each single-update prediction
      - GroupNorm after each conv stabilises training
      - 512 hidden (down from Stage 12's 768) reduces capacity to
        prevent memorisation of training-specific patterns
      - 0.1 dropout (up from Stage 12's 0.05) provides stronger
        regularisation against overfitting

    The output is still zero-mean across the vocab dimension to preserve
    the simplex constraint during integration.
    """

    def __init__(self, vocab_size, hidden_size, velocity_hidden=512):
        super().__init__()
        input_size = vocab_size + hidden_size + 1

        # Temporal context: 3 conv layers, receptive field = 13 frames
        self.temporal = nn.Sequential(
            nn.Conv1d(input_size, velocity_hidden, kernel_size=5, padding=2),
            nn.GroupNorm(1, velocity_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(velocity_hidden, velocity_hidden, kernel_size=5, padding=2),
            nn.GroupNorm(1, velocity_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(velocity_hidden, velocity_hidden, kernel_size=5, padding=2),
            nn.GroupNorm(1, velocity_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Per-frame refinement head
        self.head = nn.Sequential(
            nn.Linear(velocity_hidden, velocity_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(velocity_hidden, vocab_size),
        )

    def forward(self, pt, t, hidden):
        B, T, _ = pt.size()

        # Handle both scalar t (inference) and per-sample t (training)
        if isinstance(t, (int, float)):
            t_tensor = torch.full(
                (B, T, 1), float(t), device=pt.device, dtype=pt.dtype
            )
        else:
            # t is [B, 1, 1] from per-sample sampling, expand to [B, T, 1]
            t_tensor = t.expand(B, T, 1)

        inp = torch.cat([pt, hidden, t_tensor], dim=-1)  # [B, T, input_size]

        # Temporal processing: Conv1d expects [B, Channels, Time]
        x = self.temporal(inp.transpose(1, 2)).transpose(1, 2)  # [B, T, velocity_hidden]

        # Per-frame velocity prediction
        v = self.head(x)
        v = v - v.mean(dim=-1, keepdim=True)  # zero-mean preserves simplex
        return v


class ASR_Brain(sb.Brain):

    # ------------------------------------------------------------------
    # Dirichlet parametrisation helpers
    # ------------------------------------------------------------------

    def logits_to_dirichlet(self, logits):
        """Algorithm 2 line 7: A = softplus(Z) + epsilon."""
        return F.softplus(logits) + self.hparams.dfm_epsilon

    def dirichlet_mean(self, alpha):
        """Algorithm 2 line 8 Option A: p0 = alpha / sum(alpha)."""
        return alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(
            self.hparams.dfm_epsilon
        )

    # ------------------------------------------------------------------
    # Bridge and velocity
    # ------------------------------------------------------------------

    def bridge_state(self, p0, p1, t):
        """Algorithm 2 line 10: linear bridge phi_t = (1-t)*p0 + t*p1."""
        return (1.0 - t) * p0 + t * p1

    def target_velocity(self, p0, p1):
        """Algorithm 2 line 11: d/dt phi_t = p1 - p0 (constant for linear bridge)."""
        return p1 - p0

    # ------------------------------------------------------------------
    # ODE integration for inference
    # ------------------------------------------------------------------

    def integrate_flow(self, p0, hidden, num_steps):
        """Local-window Euler integration with magnitude-controlled velocities.

        Same local-window sweep as Stage 11b/12 (each frame gets one update),
        but with two critical additions for stable flow PER:

        1. Predicted velocities are normalised to unit length per frame
        2. Then scaled by a fixed velocity_inference_scale parameter

        This decouples direction (learned, generalises well) from magnitude
        (fixed, no overfitting). The velocity net was trained on unit-length
        targets, so its predictions are already near unit length, but
        normalising at inference eliminates any residual magnitude variance.

        The velocity_inference_scale parameter controls how much each
        frame's distribution changes in a single update. Too small and
        the flow barely modifies p0. Too large and it overshoots past
        the correct phoneme. The optimal value can be found by sweeping
        at test time without retraining.
        """
        dt = 1.0 / num_steps
        p = p0.clone()
        B, T_enc = hidden.size(0), hidden.size(1)
        scale = self.hparams.velocity_inference_scale

        for step in range(num_steps):
            t = step * dt

            # Frame window for this step
            frame_start = int(t * T_enc)
            frame_end = min(int((t + dt) * T_enc), T_enc)
            frame_start = min(frame_start, T_enc - 1)
            frame_end = max(frame_end, frame_start + 1)

            # H_τ: acoustic conditioning at the center of the window
            tau = min(int((frame_start + frame_end) / 2), T_enc - 1)
            h_tau = hidden[:, tau:tau+1, :].expand_as(hidden)

            # Run velocity net on ALL frames (temporal convs need context)
            v = self.modules.velocity_net(p, t, h_tau)

            # ============================================================
            # Fix 1+3: Normalise to unit length, then scale by fixed factor
            # Direction from the learned net, magnitude from the parameter.
            # This eliminates magnitude overfitting entirely.
            # ============================================================
            v_local = v[:, frame_start:frame_end, :]
            v_norm = torch.sqrt(
                (v_local ** 2).sum(dim=-1, keepdim=True)
            ).clamp_min(1e-8)
            v_local = v_local / v_norm * scale

            # Apply the scaled update to the local window
            p[:, frame_start:frame_end, :] = (
                p[:, frame_start:frame_end, :] + dt * v_local
            )

            # Project updated frames back onto simplex
            p[:, frame_start:frame_end, :] = p[:, frame_start:frame_end, :].clamp_min(
                self.hparams.dfm_epsilon
            )
            window_sum = p[:, frame_start:frame_end, :].sum(
                dim=-1, keepdim=True
            ).clamp_min(self.hparams.dfm_epsilon)
            p[:, frame_start:frame_end, :] = (
                p[:, frame_start:frame_end, :] / window_sum
            )

        return p

    # ------------------------------------------------------------------
    # Flow decoding: frame-aligned argmax + collapse (KEY CHANGE)
    # ------------------------------------------------------------------

    def decode_flow_output(self, p_flow, pout_lens):
        """Decode flow output using frame-aligned argmax + collapse.

        The flow is trained to match frame-level phoneme targets from
        AlignToFrames. Its output already has EXPLICIT time alignment,
        unlike CTC which uses IMPLICIT alignment via blank tokens.

        Decoding procedure:
          1. argmax at each frame -> most likely phoneme per frame
          2. Collapse consecutive duplicate predictions
          3. Remove blank tokens from the collapsed sequence

        This is analogous to how you would read off phonemes from a
        forced-alignment output, not how you would decode CTC.
        """
        frame_preds = p_flow.argmax(dim=-1)  # [B, T]
        B, T = frame_preds.size()

        sequences = []
        for b in range(B):
            length = int(round(float(pout_lens[b]) * T))
            length = max(1, min(T, length))
            raw = frame_preds[b, :length].tolist()

            # Collapse consecutive duplicates
            # e.g. [sil, sil, aa, aa, aa, sil, b, b] -> [sil, aa, sil, b]
            collapsed = []
            prev = None
            for idx in raw:
                if idx != prev:
                    collapsed.append(idx)
                    prev = idx

            # Remove blank tokens
            collapsed = [
                x for x in collapsed if x != self.hparams.blank_index
            ]

            sequences.append(collapsed)

        return sequences

    # ------------------------------------------------------------------
    # Frame-level target construction
    # ------------------------------------------------------------------

    def build_frame_targets(self, batch, hidden, wav_lens):
        """Algorithm 2 line 9: p1 = AlignToFrames(y, H).
        Maps phoneme-level labels to frame-level targets using TIMIT boundaries."""
        device = hidden.device
        B, T_enc = hidden.size(0), hidden.size(1)

        frame_ids = torch.full(
            (B, T_enc),
            fill_value=self.hparams.blank_index,
            dtype=torch.long,
            device=device,
        )
        frame_mask = torch.zeros(
            (B, T_enc), dtype=torch.float32, device=device
        )

        max_wav_len = batch.sig[0].size(1)

        for b in range(B):
            valid_samples = int(
                round(float(wav_lens[b].detach().cpu()) * max_wav_len)
            )
            valid_samples = max(valid_samples, 1)

            phns = batch.phn_list[b]
            ends = batch.phn_end_list[b]

            prev_end = 0
            for phon, end_sample in zip(phns, ends):
                start_frame = int(round((prev_end / valid_samples) * T_enc))
                end_frame = int(round((end_sample / valid_samples) * T_enc))

                start_frame = max(0, min(T_enc - 1, start_frame))
                end_frame = max(start_frame + 1, min(T_enc, end_frame))

                phon_id = int(
                    self.label_encoder.encode_sequence_torch([phon])[0].item()
                )

                frame_ids[b, start_frame:end_frame] = phon_id
                frame_mask[b, start_frame:end_frame] = 1.0
                prev_end = end_sample

        return frame_ids, frame_mask

    def make_target_simplex(self, frame_ids):
        """Convert frame-level phoneme IDs to smoothed one-hot distributions."""
        p1 = F.one_hot(
            frame_ids, num_classes=self.hparams.output_neurons
        ).float()
        smooth = self.hparams.target_smoothing
        p1 = (1.0 - smooth) * p1 + smooth / self.hparams.output_neurons
        return p1

    # ------------------------------------------------------------------
    # Time-synchronous bridge (Algorithm 3, lines 8-16)
    # ------------------------------------------------------------------

    def _make_smoothed_onehot(self, phon_id, device):
        """Create a smoothed one-hot distribution for a single phone."""
        V = self.hparams.output_neurons
        smooth = self.hparams.target_smoothing
        w = torch.zeros(V, device=device)
        w[phon_id] = 1.0
        w = (1.0 - smooth) * w + smooth / V
        return w

    def build_time_sync_bridge(self, batch, hidden, wav_lens, p0):
        """Algorithm 3, lines 8-18: Piecewise linear bridge with time-indexed H.

        The KEY time-synchronous feature: each segment's flow time t*
        maps to a frame position τ = floor(t* × T'), and the velocity
        net is conditioned on H_τ (the hidden state at that position)
        instead of each frame's own hidden state.

        This means the velocity net learns:
          "given that I'm at flow time t* (= position τ in the utterance),
           and the acoustic features at position τ are H_τ,
           how should I update the distribution?"

        This is what makes flow time = utterance time.

        Returns:
            pt:             [B, T, V]  bridge states (per-frame)
            vt:             [B, T, V]  target velocities (per-frame)
            t_per_frame:    [B, T, 1]  flow time at each frame
            mask:           [B, T]     valid frame mask
            hidden_indexed: [B, T, d]  time-indexed hidden states (H_τ per frame)
        """
        device = hidden.device
        B, T_enc, V = p0.size()

        pt = torch.zeros_like(p0)
        vt = torch.zeros_like(p0)
        t_per_frame = torch.zeros(B, T_enc, 1, device=device)
        hidden_indexed = torch.zeros_like(hidden)
        mask = torch.zeros(B, T_enc, device=device)

        max_wav_len = batch.sig[0].size(1)

        for b in range(B):
            valid_samples = int(
                round(float(wav_lens[b].detach().cpu()) * max_wav_len)
            )
            valid_samples = max(valid_samples, 1)

            phns = batch.phn_list[b]
            ends = batch.phn_end_list[b]

            # ---- Algorithm 3, line 9: waypoint times ----
            waypoint_times = [0.0]
            for end_sample in ends:
                waypoint_times.append(
                    min(float(end_sample) / valid_samples, 1.0)
                )

            # ---- Algorithm 3, line 10: waypoint states ----
            waypoint_ids = []
            for phon in phns:
                phon_id = int(
                    self.label_encoder.encode_sequence_torch([phon])[0].item()
                )
                waypoint_ids.append(phon_id)

            # ---- Process each segment k ----
            prev_end_sample = 0
            for k in range(len(phns)):
                end_sample = ends[k]

                start_frame = int(
                    round((prev_end_sample / valid_samples) * T_enc)
                )
                end_frame = int(
                    round((end_sample / valid_samples) * T_enc)
                )
                start_frame = max(0, min(T_enc - 1, start_frame))
                end_frame = max(start_frame + 1, min(T_enc, end_frame))
                seg_len = end_frame - start_frame

                t_start = waypoint_times[k]
                t_end = waypoint_times[k + 1]
                dt_seg = max(t_end - t_start, 1e-6)

                # Algorithm 3, line 14: sample t* for this segment
                t_star = t_start + torch.rand(1, device=device).item() * dt_seg

                # ============================================================
                # Algorithm 3, line 17: τ ← floor(t* × T')
                # Map flow time to encoder frame index. This connects
                # flow time to utterance position.
                # ============================================================
                tau = min(int(t_star * T_enc), T_enc - 1)

                # ============================================================
                # Algorithm 3, line 18: v_θ(p, t*, H_τ)
                # All frames in this segment are conditioned on H_τ,
                # NOT on their own per-frame hidden state. This is the
                # key difference from Stage 8c.
                # ============================================================
                hidden_indexed[b, start_frame:end_frame] = (
                    hidden[b, tau].detach().unsqueeze(0).expand(seg_len, -1)
                )

                # Waypoint states
                w_k = self._make_smoothed_onehot(waypoint_ids[k], device)

                if k == 0:
                    w_prev = p0[b, start_frame:end_frame].detach()
                else:
                    w_prev = self._make_smoothed_onehot(
                        waypoint_ids[k - 1], device
                    ).unsqueeze(0).expand(seg_len, -1)

                w_k_expanded = w_k.unsqueeze(0).expand(seg_len, -1)

                # Algorithm 3, line 11: bridge state
                frac = (t_star - t_start) / dt_seg
                pt[b, start_frame:end_frame] = (
                    w_prev + frac * (w_k_expanded - w_prev)
                )

                # Algorithm 3, line 12: target velocity
                # Raw velocity: (w_k - w_prev) / dt_seg
                raw_vt = (w_k_expanded - w_prev) / dt_seg

                # ============================================================
                # Fix 3: Normalise target velocity to unit length.
                # The velocity net only needs to learn DIRECTION, not
                # magnitude. This eliminates the main source of overfitting:
                # short phones had huge velocities (~100+ L2) while long
                # phones had small ones (~20 L2). After normalisation,
                # all segments have L2 = 1.0 regardless of duration.
                # ============================================================
                raw_vt_norm = torch.sqrt(
                    (raw_vt ** 2).sum(dim=-1, keepdim=True)
                ).clamp_min(1e-8)
                vt[b, start_frame:end_frame] = raw_vt / raw_vt_norm

                t_per_frame[b, start_frame:end_frame, 0] = t_star
                mask[b, start_frame:end_frame] = 1.0

                prev_end_sample = end_sample

        return pt, vt, t_per_frame, mask, hidden_indexed

    # ------------------------------------------------------------------
    # Monitoring helpers
    # ------------------------------------------------------------------

    def _init_monitor_sums(self):
        self.monitor_sums = {
            "loss": 0.0,
            "ctc_loss": 0.0,
            "fm_loss": 0.0,
            "grad_norm": 0.0,
            "logits_mean": 0.0,
            "logits_std": 0.0,
            "alpha_mean": 0.0,
            "alpha_std": 0.0,
            "p0_entropy": 0.0,
            "p0_row_sum_mean": 0.0,
            "p0_min": 0.0,
            "p1_entropy": 0.0,
            "p1_row_sum_mean": 0.0,
            "p1_min": 0.0,
            "pt_entropy": 0.0,
            "pt_row_sum_mean": 0.0,
            "pt_min": 0.0,
            "vt_mean_abs": 0.0,
            "vt_l2": 0.0,
            "vt_sum_mean": 0.0,
            "vpred_mean_abs": 0.0,
            "vpred_l2": 0.0,
            "vpred_sum_mean": 0.0,
            "velocity_mse_monitor": 0.0,
            "velocity_cosine_monitor": 0.0,
            "t_mean": 0.0,
            "blank_prob_mean": 0.0,
            "max_prob_mean": 0.0,
            "flow_entropy": 0.0,
            "flow_p0_delta": 0.0,
            "num_batches": 0,
        }

    def _update_monitor_sums(
        self,
        total_loss,
        ctc_loss,
        fm_loss,
        logits,
        alpha,
        p0,
        p1,
        pt,
        vt,
        vpred,
        t_value,
        log_probs,
        grad_norm=None,
        p_flow=None,
    ):
        probs = log_probs.exp()
        p0_entropy = -(p0 * p0.clamp_min(1e-8).log()).sum(dim=-1).mean()
        p1_entropy = -(p1 * p1.clamp_min(1e-8).log()).sum(dim=-1).mean()
        pt_entropy = -(pt * pt.clamp_min(1e-8).log()).sum(dim=-1).mean()

        blank_prob_mean = probs[..., self.hparams.blank_index].mean()
        max_prob_mean = probs.max(dim=-1).values.mean()

        vt_mean_abs = vt.abs().mean()
        vt_l2 = torch.sqrt((vt ** 2).sum(dim=-1)).mean()
        vt_sum_mean = vt.sum(dim=-1).mean()

        vpred_mean_abs = vpred.abs().mean()
        vpred_l2 = torch.sqrt((vpred ** 2).sum(dim=-1)).mean()
        vpred_sum_mean = vpred.sum(dim=-1).mean()

        velocity_mse_monitor = ((vpred - vt) ** 2).mean()
        cos = F.cosine_similarity(vpred, vt, dim=-1).mean()

        self.monitor_sums["loss"] += float(total_loss.detach().cpu())
        self.monitor_sums["ctc_loss"] += float(ctc_loss.detach().cpu())
        self.monitor_sums["fm_loss"] += float(fm_loss.detach().cpu())
        self.monitor_sums["logits_mean"] += float(
            logits.mean().detach().cpu()
        )
        self.monitor_sums["logits_std"] += float(
            logits.std().detach().cpu()
        )
        self.monitor_sums["alpha_mean"] += float(
            alpha.mean().detach().cpu()
        )
        self.monitor_sums["alpha_std"] += float(alpha.std().detach().cpu())

        self.monitor_sums["p0_entropy"] += float(p0_entropy.detach().cpu())
        self.monitor_sums["p0_row_sum_mean"] += float(
            p0.sum(dim=-1).mean().detach().cpu()
        )
        self.monitor_sums["p0_min"] += float(p0.min().detach().cpu())

        self.monitor_sums["p1_entropy"] += float(p1_entropy.detach().cpu())
        self.monitor_sums["p1_row_sum_mean"] += float(
            p1.sum(dim=-1).mean().detach().cpu()
        )
        self.monitor_sums["p1_min"] += float(p1.min().detach().cpu())

        self.monitor_sums["pt_entropy"] += float(pt_entropy.detach().cpu())
        self.monitor_sums["pt_row_sum_mean"] += float(
            pt.sum(dim=-1).mean().detach().cpu()
        )
        self.monitor_sums["pt_min"] += float(pt.min().detach().cpu())

        self.monitor_sums["vt_mean_abs"] += float(
            vt_mean_abs.detach().cpu()
        )
        self.monitor_sums["vt_l2"] += float(vt_l2.detach().cpu())
        self.monitor_sums["vt_sum_mean"] += float(
            vt_sum_mean.detach().cpu()
        )

        self.monitor_sums["vpred_mean_abs"] += float(
            vpred_mean_abs.detach().cpu()
        )
        self.monitor_sums["vpred_l2"] += float(vpred_l2.detach().cpu())
        self.monitor_sums["vpred_sum_mean"] += float(
            vpred_sum_mean.detach().cpu()
        )
        self.monitor_sums["velocity_mse_monitor"] += float(
            velocity_mse_monitor.detach().cpu()
        )
        self.monitor_sums["velocity_cosine_monitor"] += float(
            cos.detach().cpu()
        )

        self.monitor_sums["t_mean"] += float(t_value)
        self.monitor_sums["blank_prob_mean"] += float(
            blank_prob_mean.detach().cpu()
        )
        self.monitor_sums["max_prob_mean"] += float(
            max_prob_mean.detach().cpu()
        )

        if p_flow is not None:
            flow_entropy = (
                -(p_flow * p_flow.clamp_min(1e-8).log()).sum(dim=-1).mean()
            )
            flow_p0_delta = (p_flow - p0).abs().mean()
            self.monitor_sums["flow_entropy"] += float(
                flow_entropy.detach().cpu()
            )
            self.monitor_sums["flow_p0_delta"] += float(
                flow_p0_delta.detach().cpu()
            )

        if grad_norm is not None:
            self.monitor_sums["grad_norm"] += float(grad_norm)

        self.monitor_sums["num_batches"] += 1

    def _get_monitor_averages(self):
        n = max(self.monitor_sums["num_batches"], 1)
        return {
            k: (v / n if k != "num_batches" else v)
            for k, v in self.monitor_sums.items()
            if k != "num_batches"
        }

    def _get_grad_norm(self):
        total = 0.0
        found = False
        for p in self.modules.parameters():
            if p.grad is not None:
                total += float(p.grad.detach().norm().cpu()) ** 2
                found = True
        if not found:
            return None
        return total ** 0.5

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig

        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            wavs, wav_lens = self.hparams.wav_augment(wavs, wav_lens)

        feats = self.hparams.compute_features(wavs)
        feats = self.modules.normalize(feats, wav_lens)

        # Encoder: Algorithm 3, lines 4-7
        hidden = self.modules.model(feats)
        logits = self.modules.output(hidden)
        alpha = self.logits_to_dirichlet(logits)
        p0 = self.dirichlet_mean(alpha)

        # Frame targets (still needed for p1 in monitoring and CTC)
        frame_ids, frame_mask = self.build_frame_targets(
            batch, hidden, wav_lens
        )
        p1 = self.make_target_simplex(frame_ids)

        # ============================================================
        # TIME-SYNC: Piecewise bridge + time-indexed hidden (Algorithm 3)
        #
        # Two key differences from Stage 8c:
        # 1. Piecewise bridge through phone boundary waypoints
        # 2. Velocity net conditioned on H_τ (time-indexed hidden)
        #    instead of per-frame hidden states
        #
        # hidden_indexed[b, f, :] = hidden[b, τ_k, :] where τ_k is the
        # encoder frame corresponding to segment k's flow time t*_k.
        # This teaches the velocity net: "at flow time t*, the relevant
        # acoustic information is at frame τ = floor(t* × T')."
        # ============================================================
        pt, vt, t_per_frame, ts_mask, hidden_indexed = (
            self.build_time_sync_bridge(batch, hidden, wav_lens, p0)
        )

        # Velocity prediction with time-indexed hidden (NOT full hidden)
        vpred = self.modules.velocity_net(pt, t_per_frame, hidden_indexed)

        # p0 log-probs for CTC loss and p0 PER metric
        log_probs = torch.log(p0.clamp_min(1e-8))

        # Flow integration only at eval time
        if stage != sb.Stage.TRAIN:
            p_flow = self.integrate_flow(
                p0, hidden, self.hparams.num_flow_steps
            )
        else:
            p_flow = None

        return {
            "hidden": hidden,
            "logits": logits,
            "alpha": alpha,
            "p0": p0,
            "p1": p1,
            "pt": pt,
            "vt": vt,
            "vpred": vpred,
            "t": t_per_frame.mean().item(),
            "frame_ids": frame_ids,
            "frame_mask": ts_mask,  # use time-sync mask
            "log_probs": log_probs,
            "wav_lens": wav_lens,
            "p_flow": p_flow,
        }

    # ------------------------------------------------------------------
    # Objectives
    # ------------------------------------------------------------------

    def compute_objectives(self, predictions, batch, stage):
        log_probs = predictions["log_probs"]
        logits = predictions["logits"]
        alpha = predictions["alpha"]
        p0 = predictions["p0"]
        p1 = predictions["p1"]
        pt = predictions["pt"]
        vt = predictions["vt"]
        vpred = predictions["vpred"]
        t = predictions["t"]
        frame_mask = predictions["frame_mask"]
        pout_lens = predictions["wav_lens"]
        p_flow = predictions["p_flow"]

        phns, phn_lens = batch.phn_encoded

        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            phns = self.hparams.wav_augment.replicate_labels(phns)
            phn_lens = self.hparams.wav_augment.replicate_labels(phn_lens)

        # CTC loss on p0 log-probs (always)
        ctc_loss = self.hparams.compute_cost(
            log_probs, phns, pout_lens, phn_lens
        )

        # FM loss on unit-normalised targets (Fix 3)
        # With normalised targets, vpred and vt both have L2 ≈ 1.0,
        # so the FM loss is on the order of 0.5-2.0 (vs ~80-100 before).
        # lambda_fm is set higher accordingly (1.0 instead of 0.02).
        fm_sq = ((vpred - vt) ** 2).mean(dim=-1)
        fm_loss = (fm_sq * frame_mask).sum() / frame_mask.sum().clamp_min(1.0)

        # Fix 2: L2 weight decay on velocity net parameters only.
        # Penalises large weights that enable memorisation of
        # training-specific magnitude patterns. Does not affect the
        # encoder because the FM loss is detached from the encoder.
        velocity_wd = self.hparams.velocity_weight_decay
        if velocity_wd > 0:
            l2_reg = sum(
                p.pow(2).sum()
                for p in self.modules.velocity_net.parameters()
            )
            fm_loss = fm_loss + velocity_wd * l2_reg

        total_loss = (
            self.hparams.lambda_ctc * ctc_loss
            + self.hparams.lambda_fm * fm_loss
        )

        self.current_batch_stats = {
            "loss": total_loss.detach(),
            "ctc_loss": ctc_loss.detach(),
            "fm_loss": fm_loss.detach(),
            "logits": logits.detach(),
            "alpha": alpha.detach(),
            "p0": p0.detach(),
            "p1": p1.detach(),
            "pt": pt.detach(),
            "vt": vt.detach(),
            "vpred": vpred.detach(),
            "t": t,
            "log_probs": log_probs.detach(),
            "p_flow": p_flow.detach() if p_flow is not None else None,
        }

        self.ctc_metrics.append(
            batch.id, log_probs, phns, pout_lens, phn_lens
        )

        if stage != sb.Stage.TRAIN:
            # ----- p0 PER: CTC greedy decode (implicit alignment) -----
            sequence_p0 = sb.decoders.ctc_greedy_decode(
                log_probs, pout_lens, blank_id=self.hparams.blank_index
            )
            self.per_metrics_p0.append(
                ids=batch.id,
                predict=sequence_p0,
                target=phns,
                target_len=phn_lens,
                ind2lab=self.label_encoder.decode_ndim,
            )

            # ----- Flow PER: frame-aligned argmax+collapse (KEY FIX) -----
            # Uses decode_flow_output instead of ctc_greedy_decode because
            # the flow produces frame-level phoneme predictions with explicit
            # alignment, not CTC-style blank-heavy implicit alignment.
            sequence_flow = self.decode_flow_output(p_flow, pout_lens)
            self.per_metrics_flow.append(
                ids=batch.id,
                predict=sequence_flow,
                target=phns,
                target_len=phn_lens,
                ind2lab=self.label_encoder.decode_ndim,
            )

            self._update_monitor_sums(
                total_loss=total_loss,
                ctc_loss=ctc_loss,
                fm_loss=fm_loss,
                logits=logits,
                alpha=alpha,
                p0=p0,
                p1=p1,
                pt=pt,
                vt=vt,
                vpred=vpred,
                t_value=t,
                log_probs=log_probs,
                grad_norm=None,
                p_flow=p_flow,
            )

        return total_loss

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def fit_batch(self, batch):
        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)

        self.optimizer.zero_grad()
        loss.backward()

        grad_norm = self._get_grad_norm()

        self._update_monitor_sums(
            total_loss=self.current_batch_stats["loss"],
            ctc_loss=self.current_batch_stats["ctc_loss"],
            fm_loss=self.current_batch_stats["fm_loss"],
            logits=self.current_batch_stats["logits"],
            alpha=self.current_batch_stats["alpha"],
            p0=self.current_batch_stats["p0"],
            p1=self.current_batch_stats["p1"],
            pt=self.current_batch_stats["pt"],
            vt=self.current_batch_stats["vt"],
            vpred=self.current_batch_stats["vpred"],
            t_value=self.current_batch_stats["t"],
            log_probs=self.current_batch_stats["log_probs"],
            grad_norm=grad_norm,
            p_flow=self.current_batch_stats["p_flow"],
        )

        self.optimizer.step()
        return loss.detach()

    def evaluate_batch(self, batch, stage):
        with torch.no_grad():
            predictions = self.compute_forward(batch, stage=stage)
            loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()

    # ------------------------------------------------------------------
    # Stage callbacks
    # ------------------------------------------------------------------

    def on_stage_start(self, stage, epoch):
        self.ctc_metrics = self.hparams.ctc_stats()
        self._init_monitor_sums()
        if stage != sb.Stage.TRAIN:
            self.per_metrics_flow = self.hparams.per_stats()
            self.per_metrics_p0 = self.hparams.per_stats()

    def on_stage_end(self, stage, stage_loss, epoch):
        monitor_stats = self._get_monitor_averages()

        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
            self.train_monitor_stats = monitor_stats
        else:
            per_flow = self.per_metrics_flow.summarize("error_rate")
            per_p0 = self.per_metrics_p0.summarize("error_rate")

        if stage == sb.Stage.VALID:
            # Use p0 PER for LR scheduling and checkpointing (stable metric)
            old_lr, new_lr = self.hparams.lr_annealing(per_p0)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_monitor_stats,
                valid_stats={
                    "PER": per_flow,       # flow PER (argmax+collapse)
                    "PER_p0": per_p0,      # p0 PER (CTC greedy)
                    **monitor_stats,
                },
            )

            self.checkpointer.save_and_keep_only(
                meta={"PER": per_p0},
                min_keys=["PER"],
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={
                    "Epoch loaded": self.hparams.epoch_counter.current
                },
                test_stats={
                    "loss": stage_loss,
                    "PER": per_flow,
                    "PER_p0": per_p0,
                },
            )
            if if_main_process():
                with open(
                    self.hparams.test_wer_file, "w", encoding="utf-8"
                ) as w:
                    w.write("CTC loss stats:\n")
                    self.ctc_metrics.write_stats(w)
                    w.write("\nFlow PER stats (argmax+collapse decode):\n")
                    self.per_metrics_flow.write_stats(w)
                    w.write("\np0 PER stats (CTC greedy decode):\n")
                    self.per_metrics_p0.write_stats(w)
                    print(
                        "CTC and PER stats written to ",
                        self.hparams.test_wer_file,
                    )


# ======================================================================
# Data pipeline
# ======================================================================


def dataio_prep(hparams):
    """Creates the datasets and their data processing pipelines."""

    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["train_annotation"],
        replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        train_data = train_data.filtered_sorted(sort_key="duration")
        hparams["train_dataloader_opts"]["shuffle"] = False
    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        hparams["train_dataloader_opts"]["shuffle"] = False
    elif hparams["sorting"] == "random":
        pass
    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["valid_annotation"],
        replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    test_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["test_annotation"],
        replacements={"data_root": data_folder},
    )
    test_data = test_data.filtered_sorted(sort_key="duration")

    datasets = [train_data, valid_data, test_data]
    label_encoder = sb.dataio.encoder.CTCTextEncoder()

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    @sb.utils.data_pipeline.takes("phn", "ground_truth_phn_ends")
    @sb.utils.data_pipeline.provides(
        "phn_list", "phn_encoded", "phn_end_list"
    )
    def text_pipeline(phn, ground_truth_phn_ends):
        phn_list = phn.strip().split()
        yield phn_list
        yield label_encoder.encode_sequence_torch(phn_list)
        yield [int(x) for x in ground_truth_phn_ends.strip().split()]

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[train_data],
        output_key="phn_list",
        special_labels={"blank_label": hparams["blank_index"]},
        sequence_input=True,
    )

    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "sig", "phn_encoded", "phn_list", "phn_end_list"],
    )

    return train_data, valid_data, test_data, label_encoder


if __name__ == "__main__":
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    from timit_prepare import prepare_timit  # noqa

    sb.utils.distributed.ddp_init_group(run_opts)

    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    run_on_main(
        prepare_timit,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_json_train": hparams["train_annotation"],
            "save_json_valid": hparams["valid_annotation"],
            "save_json_test": hparams["test_annotation"],
            "skip_prep": hparams["skip_prep"],
            "uppercase": hparams["uppercase"],
        },
    )
    run_on_main(hparams["prepare_noise_data"])

    train_data, valid_data, test_data, label_encoder = dataio_prep(hparams)

    asr_brain = ASR_Brain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    asr_brain.label_encoder = label_encoder

    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    asr_brain.evaluate(
        test_data,
        min_key="PER",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )