#!/usr/bin/env python3
"""Stage 8: CTC ASR on TIMIT with proper Dirichlet Flow Matching.

Changes from Stage 7:
---------------------
1. Added integrate_flow() - Euler ODE integration of the learned velocity field
   from t=0 to t=1. This is the missing piece that turns the velocity network
   from a side loss into an actual generative flow that produces predictions.

2. Modified compute_forward() - During validation/test, the model now integrates
   the flow and decodes from the flow-evolved distribution instead of raw p0.
   During training, CTC still uses p0 for speed (FM loss still trains velocity net).

3. Added dual PER tracking - At validation, both p0 PER and flow PER are computed
   and logged. This lets you directly compare whether the flow improves predictions
   over the raw encoder output.

4. Added flow monitoring stats - flow_entropy and flow_p0_delta track how much
   the flow changes the distribution during integration.

5. LR scheduler now uses flow PER - The learning rate and checkpoint selection
   are based on the flow-evolved PER, since that is the actual system output.

Algorithm reference (from dissertation Algorithm 2):
  Training: sample t, build bridge pt* = (1-t)p0 + tp1, compute target velocity
            vt* = p1-p0, predict v_hat = v_theta(pt*, t, H), loss = ||v_hat - vt*||^2
  Inference: start from p0, integrate dp/dt = v_theta(p, t, H) from t=0 to t=1
             using Euler steps, decode from the resulting p_final
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
    """Predicts the velocity field v_theta(p_t, t, H).

    Takes the current simplex state p_t, the scalar flow time t, and the
    encoder hidden states H, and outputs a velocity vector for each frame.
    The output is zero-mean across the vocab dimension so it preserves the
    simplex constraint (probabilities sum to 1) during integration.
    """

    def __init__(self, vocab_size, hidden_size, velocity_hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vocab_size + hidden_size + 1, velocity_hidden),
            nn.ReLU(),
            nn.Linear(velocity_hidden, velocity_hidden),
            nn.ReLU(),
            nn.Linear(velocity_hidden, vocab_size),
        )

    def forward(self, pt, t, hidden):
        t_tensor = torch.full(
            (pt.size(0), pt.size(1), 1),
            float(t),
            device=pt.device,
            dtype=pt.dtype,
        )
        inp = torch.cat([pt, hidden, t_tensor], dim=-1)
        v = self.net(inp)
        # Zero-mean constraint: velocity should not change the sum of
        # probabilities, so we subtract the mean across the vocab dimension.
        v = v - v.mean(dim=-1, keepdim=True)
        return v


class ASR_Brain(sb.Brain):

    # ------------------------------------------------------------------
    # Dirichlet parametrisation helpers (unchanged from Stage 7)
    # ------------------------------------------------------------------

    def logits_to_dirichlet(self, logits):
        """Convert raw logits to positive Dirichlet concentration parameters.
        Matches Algorithm 2 line 7: A = softplus(Z) + epsilon."""
        return F.softplus(logits) + self.hparams.dfm_epsilon

    def dirichlet_mean(self, alpha):
        """Compute the Dirichlet mean (Option A from Algorithm 2 line 8).
        p0 = alpha / sum(alpha) for each frame."""
        return alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(
            self.hparams.dfm_epsilon
        )

    # ------------------------------------------------------------------
    # Bridge and velocity (unchanged from Stage 7)
    # ------------------------------------------------------------------

    def bridge_state(self, p0, p1, t):
        """Simplex linear bridge: phi_t(p0, p1) = (1-t)*p0 + t*p1.
        Matches Algorithm 2 line 10 with the linear bridge choice."""
        return (1.0 - t) * p0 + t * p1

    def target_velocity(self, p0, p1):
        """Analytic derivative of the linear bridge: d/dt phi_t = p1 - p0.
        Matches Algorithm 2 line 11. This is constant for all t."""
        return p1 - p0

    # ------------------------------------------------------------------
    # NEW: ODE integration for inference (Change 1)
    # ------------------------------------------------------------------

    def integrate_flow(self, p0, hidden, num_steps):
        """Euler integration of the learned velocity field from t=0 to t=1.

        This is the inference procedure that makes the velocity network
        actually produce predictions. At each step:
          p(t + dt) = p(t) + dt * v_theta(p(t), t, H)
        then project back onto the simplex (clamp + renormalise).

        Args:
            p0: Initial simplex state [B, T, V] from Dirichlet mean
            hidden: Encoder hidden states [B, T, D] for conditioning
            num_steps: Number of Euler steps (more = more accurate but slower)

        Returns:
            p_final: Flow-evolved distribution [B, T, V] on the simplex
        """
        dt = 1.0 / num_steps
        p = p0.clone()
        t = 0.0
        for step in range(num_steps):
            v = self.modules.velocity_net(p, t, hidden)
            p = p + dt * v
            # Project back onto the simplex after each step.
            # Without this, numerical drift causes values to go negative
            # or not sum to 1, which breaks the probability interpretation.
            p = p.clamp_min(self.hparams.dfm_epsilon)
            p = p / p.sum(dim=-1, keepdim=True).clamp_min(
                self.hparams.dfm_epsilon
            )
            t += dt
        return p

    # ------------------------------------------------------------------
    # Frame-level target construction (unchanged from Stage 7)
    # ------------------------------------------------------------------

    def build_frame_targets(self, batch, hidden, wav_lens):
        """Map phoneme-level labels to frame-level targets using TIMIT boundaries.
        Matches Algorithm 2 line 9: p1 = AlignToFrames(y, H)."""
        device = hidden.device
        B, T_enc = hidden.size(0), hidden.size(1)

        frame_ids = torch.full(
            (B, T_enc),
            fill_value=self.hparams.blank_index,
            dtype=torch.long,
            device=device,
        )
        frame_mask = torch.zeros((B, T_enc), dtype=torch.float32, device=device)

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
        """Convert frame-level phoneme IDs to smoothed one-hot target distributions."""
        p1 = F.one_hot(
            frame_ids, num_classes=self.hparams.output_neurons
        ).float()
        smooth = self.hparams.target_smoothing
        p1 = (1.0 - smooth) * p1 + smooth / self.hparams.output_neurons
        return p1

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
            # NEW (Change 4): flow-specific monitoring
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
        # NEW (Change 4): optional flow stats
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
        self.monitor_sums["logits_mean"] += float(logits.mean().detach().cpu())
        self.monitor_sums["logits_std"] += float(logits.std().detach().cpu())
        self.monitor_sums["alpha_mean"] += float(alpha.mean().detach().cpu())
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

        self.monitor_sums["vt_mean_abs"] += float(vt_mean_abs.detach().cpu())
        self.monitor_sums["vt_l2"] += float(vt_l2.detach().cpu())
        self.monitor_sums["vt_sum_mean"] += float(vt_sum_mean.detach().cpu())

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

        # NEW (Change 4): flow-specific stats
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
    # Forward pass (Change 2: flow integration at eval time)
    # ------------------------------------------------------------------

    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig

        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            wavs, wav_lens = self.hparams.wav_augment(wavs, wav_lens)

        feats = self.hparams.compute_features(wavs)
        feats = self.modules.normalize(feats, wav_lens)

        # Encoder forward pass: lines 5-8 of Algorithm 2
        hidden = self.modules.model(feats)          # H = f_enc(X)
        logits = self.modules.output(hidden)         # Z = f_proj(H)
        alpha = self.logits_to_dirichlet(logits)     # A = softplus(Z) + eps
        p0 = self.dirichlet_mean(alpha)              # p0 = A / sum(A)

        # Build frame-level targets: line 9 of Algorithm 2
        frame_ids, frame_mask = self.build_frame_targets(
            batch, hidden, wav_lens
        )
        p1 = self.make_target_simplex(frame_ids)

        # Sample flow time and compute bridge/velocity: lines 10-13
        t = torch.rand(1, device=hidden.device).item()
        pt = self.bridge_state(p0, p1, t)            # reference state at t
        vt = self.target_velocity(p0, p1)            # reference velocity
        vpred = self.modules.velocity_net(pt, t, hidden)  # predicted velocity

        # --- THIS IS THE KEY CHANGE (Change 2) ---
        # During training: CTC uses p0 (fast, no integration needed)
        # During validation/test: CTC uses the flow-evolved distribution
        if stage == sb.Stage.TRAIN:
            log_probs = torch.log(p0.clamp_min(1e-8))
            p_flow = None
        else:
            # Integrate the velocity field from t=0 to t=1
            p_flow = self.integrate_flow(
                p0, hidden, self.hparams.num_flow_steps
            )
            log_probs = torch.log(p_flow.clamp_min(1e-8))

        return {
            "hidden": hidden,
            "logits": logits,
            "alpha": alpha,
            "p0": p0,
            "p1": p1,
            "pt": pt,
            "vt": vt,
            "vpred": vpred,
            "t": t,
            "frame_ids": frame_ids,
            "frame_mask": frame_mask,
            "log_probs": log_probs,   # from p0 during train, from flow at eval
            "wav_lens": wav_lens,
            "p_flow": p_flow,         # NEW: None during train, tensor at eval
        }

    # ------------------------------------------------------------------
    # Objectives (Change 3: dual PER tracking)
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

        # CTC loss: during training uses p0, during eval uses flow output
        ctc_loss = self.hparams.compute_cost(
            log_probs, phns, pout_lens, phn_lens
        )

        # FM loss: always computed on the velocity prediction vs target
        # This is Algorithm 2 line 13: L += ||v_hat - vt*||^2
        fm_sq = ((vpred - vt) ** 2).mean(dim=-1)
        fm_loss = (fm_sq * frame_mask).sum() / frame_mask.sum().clamp_min(1.0)

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
            # --- CHANGE 3: Decode from FLOW output (primary metric) ---
            sequence_flow = sb.decoders.ctc_greedy_decode(
                log_probs, pout_lens, blank_id=self.hparams.blank_index
            )
            self.per_metrics_flow.append(
                ids=batch.id,
                predict=sequence_flow,
                target=phns,
                target_len=phn_lens,
                ind2lab=self.label_encoder.decode_ndim,
            )

            # --- CHANGE 3: Also decode from p0 (comparison metric) ---
            log_probs_p0 = torch.log(p0.clamp_min(1e-8))
            sequence_p0 = sb.decoders.ctc_greedy_decode(
                log_probs_p0, pout_lens, blank_id=self.hparams.blank_index
            )
            self.per_metrics_p0.append(
                ids=batch.id,
                predict=sequence_p0,
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
    # Training loop (unchanged from Stage 7)
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
    # Stage callbacks (Change 3: dual PER metrics)
    # ------------------------------------------------------------------

    def on_stage_start(self, stage, epoch):
        self.ctc_metrics = self.hparams.ctc_stats()
        self._init_monitor_sums()
        if stage != sb.Stage.TRAIN:
            # NEW: two separate PER trackers
            self.per_metrics_flow = self.hparams.per_stats()
            self.per_metrics_p0 = self.hparams.per_stats()

    def on_stage_end(self, stage, stage_loss, epoch):
        monitor_stats = self._get_monitor_averages()

        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
            self.train_monitor_stats = monitor_stats
        else:
            # NEW: compute both PERs
            per_flow = self.per_metrics_flow.summarize("error_rate")
            per_p0 = self.per_metrics_p0.summarize("error_rate")

        if stage == sb.Stage.VALID:
            # NEW (Change 5): LR scheduler uses flow PER since that is
            # the actual system output we care about
            old_lr, new_lr = self.hparams.lr_annealing(per_flow)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_monitor_stats,
                valid_stats={
                    "PER": per_flow,       # flow-evolved PER (primary)
                    "PER_p0": per_p0,      # raw encoder PER (comparison)
                    **monitor_stats,
                },
            )

            # NEW (Change 5): checkpoint based on flow PER
            self.checkpointer.save_and_keep_only(
                meta={"PER": per_flow},
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
                    w.write("\nFlow PER stats:\n")
                    self.per_metrics_flow.write_stats(w)
                    w.write("\np0 PER stats (no flow):\n")
                    self.per_metrics_p0.write_stats(w)
                    print(
                        "CTC and PER stats written to ",
                        self.hparams.test_wer_file,
                    )


# ======================================================================
# Data pipeline (unchanged from Stage 7)
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