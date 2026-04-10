#!/usr/bin/env python3
"""Recipe for training a phoneme recognizer on TIMIT.

DFM v3:
- encoder -> logits -> DFM -> log-probs -> CTC
- explicit flow-matching loss on Dirichlet bridge states
- monitoring for loss components, simplex sanity, entropy, and DFM activity
"""

import math
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


class DFMModule(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_size,
        dfm_hidden=256,
        num_flow_steps=4,
        epsilon=1e-6,
        target_scale=20.0,
        bridge_temperature=1.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.dfm_hidden = dfm_hidden
        self.num_flow_steps = num_flow_steps
        self.epsilon = epsilon
        self.target_scale = target_scale
        self.bridge_temperature = bridge_temperature

        self.velocity_net = nn.Sequential(
            nn.Linear(vocab_size + hidden_size + 1, dfm_hidden),
            nn.ReLU(),
            nn.Linear(dfm_hidden, dfm_hidden),
            nn.ReLU(),
            nn.Linear(dfm_hidden, vocab_size),
        )

    def logits_to_dirichlet(self, logits):
        return F.softplus(logits) + self.epsilon

    def dirichlet_mean(self, alpha):
        return alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(self.epsilon)

    def project_to_simplex(self, x):
        x = x.clamp_min(self.epsilon)
        return x / x.sum(dim=-1, keepdim=True).clamp_min(self.epsilon)

    def simplex_to_dirichlet(self, p, scale=None):
        if scale is None:
            scale = self.target_scale
        return scale * p.clamp_min(self.epsilon) + self.epsilon

    def velocity(self, p, t, h):
        t_tensor = torch.full(
            (p.size(0), p.size(1), 1),
            float(t),
            device=p.device,
            dtype=p.dtype,
        )
        inp = torch.cat([p, h, t_tensor], dim=-1)
        v = self.velocity_net(inp)
        v = v - v.mean(dim=-1, keepdim=True)
        return v

    def sample_dirichlet_bridge(self, alpha0, p1, t):
        alpha1 = self.simplex_to_dirichlet(p1, self.target_scale)
        alpha_t = (1.0 - t) * alpha0 + t * alpha1

        if self.bridge_temperature != 1.0:
            alpha_t = alpha_t.pow(self.bridge_temperature)

        x_t = self.dirichlet_mean(alpha_t)
        return self.project_to_simplex(x_t), alpha1, alpha_t

    def integrate(self, p_init, hidden_states):
        p = p_init
        dt = 1.0 / self.num_flow_steps
        for step in range(self.num_flow_steps):
            t = step / self.num_flow_steps
            v = self.velocity(p, t, hidden_states)
            p = p + dt * v
            p = self.project_to_simplex(p)
        return p

    def forward(self, logits, hidden_states):
        alpha0 = self.logits_to_dirichlet(logits)
        p0 = self.dirichlet_mean(alpha0)
        final_probs = self.integrate(p0, hidden_states)
        return final_probs, alpha0, p0


class ASR_Brain(sb.Brain):
    def build_frame_targets(self, batch, hidden, wav_lens):
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
            valid_samples = int(round(float(wav_lens[b].detach().cpu()) * max_wav_len))
            valid_samples = max(valid_samples, 1)

            phns = batch.phn_list[b]
            ends = batch.phn_end_list[b]

            prev_end = 0
            for phon, end_sample in zip(phns, ends):
                start_frame = int(round((prev_end / valid_samples) * T_enc))
                end_frame = int(round((end_sample / valid_samples) * T_enc))

                start_frame = max(0, min(T_enc - 1, start_frame))
                end_frame = max(start_frame + 1, min(T_enc, end_frame))

                phon_id = int(self.label_encoder.encode_sequence_torch([phon])[0].item())

                frame_ids[b, start_frame:end_frame] = phon_id
                frame_mask[b, start_frame:end_frame] = 1.0
                prev_end = end_sample

        return frame_ids, frame_mask

    def make_target_simplex(self, frame_ids):
        y = F.one_hot(
            frame_ids, num_classes=self.hparams.output_neurons
        ).float()
        smooth = self.hparams.target_smoothing
        y = (1.0 - smooth) * y + smooth / self.hparams.output_neurons
        return y

    def compute_entropy(self, probs):
        probs = probs.clamp_min(1e-8)
        ent = -(probs * probs.log()).sum(dim=-1)
        return ent.mean()

    def _init_monitor_sums(self):
        self.monitor_sums = {
            "ctc_loss": 0.0,
            "fm_loss": 0.0,
            "reg_loss": 0.0,
            "flow_delta": 0.0,
            "p0_row_sum_mean": 0.0,
            "xt_row_sum_mean": 0.0,
            "final_row_sum_mean": 0.0,
            "final_min": 0.0,
            "t_value": 0.0,
            "dfm_grad_norm": 0.0,
            "p0_entropy": 0.0,
            "xt_entropy": 0.0,
            "final_entropy": 0.0,
            "num_batches": 0,
        }

    def _update_monitor_sums(
        self,
        ctc_loss,
        fm_loss,
        reg_loss,
        p0,
        xt,
        final_probs,
        t_value,
        dfm_grad_norm=None,
    ):
        self.monitor_sums["ctc_loss"] += float(ctc_loss.detach().cpu())
        self.monitor_sums["fm_loss"] += float(fm_loss.detach().cpu())
        self.monitor_sums["reg_loss"] += float(reg_loss.detach().cpu())
        self.monitor_sums["flow_delta"] += float(
            (final_probs - p0).abs().mean().detach().cpu()
        )
        self.monitor_sums["p0_row_sum_mean"] += float(
            p0.sum(dim=-1).mean().detach().cpu()
        )
        self.monitor_sums["xt_row_sum_mean"] += float(
            xt.sum(dim=-1).mean().detach().cpu()
        )
        self.monitor_sums["final_row_sum_mean"] += float(
            final_probs.sum(dim=-1).mean().detach().cpu()
        )
        self.monitor_sums["final_min"] += float(final_probs.min().detach().cpu())
        self.monitor_sums["t_value"] += float(t_value)
        self.monitor_sums["p0_entropy"] += float(self.compute_entropy(p0).detach().cpu())
        self.monitor_sums["xt_entropy"] += float(self.compute_entropy(xt).detach().cpu())
        self.monitor_sums["final_entropy"] += float(
            self.compute_entropy(final_probs).detach().cpu()
        )
        if dfm_grad_norm is not None:
            self.monitor_sums["dfm_grad_norm"] += float(dfm_grad_norm)
        self.monitor_sums["num_batches"] += 1

    def _get_monitor_averages(self):
        n = max(self.monitor_sums["num_batches"], 1)
        return {
            "ctc_loss": self.monitor_sums["ctc_loss"] / n,
            "fm_loss": self.monitor_sums["fm_loss"] / n,
            "reg_loss": self.monitor_sums["reg_loss"] / n,
            "flow_delta": self.monitor_sums["flow_delta"] / n,
            "p0_row_sum_mean": self.monitor_sums["p0_row_sum_mean"] / n,
            "xt_row_sum_mean": self.monitor_sums["xt_row_sum_mean"] / n,
            "final_row_sum_mean": self.monitor_sums["final_row_sum_mean"] / n,
            "final_min": self.monitor_sums["final_min"] / n,
            "t_value": self.monitor_sums["t_value"] / n,
            "dfm_grad_norm": self.monitor_sums["dfm_grad_norm"] / n,
            "p0_entropy": self.monitor_sums["p0_entropy"] / n,
            "xt_entropy": self.monitor_sums["xt_entropy"] / n,
            "final_entropy": self.monitor_sums["final_entropy"] / n,
        }

    def _get_dfm_grad_norm(self):
        total = 0.0
        found = False
        for p in self.modules.dfm.parameters():
            if p.grad is not None:
                total += float(p.grad.detach().norm().cpu()) ** 2
                found = True
        if not found:
            return None
        return math.sqrt(total)

    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig

        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            wavs, wav_lens = self.hparams.wav_augment(wavs, wav_lens)

        feats = self.hparams.compute_features(wavs)
        feats = self.modules.normalize(feats, wav_lens)

        hidden = self.modules.model(feats)
        logits = self.modules.output(hidden)

        final_probs, alpha0, p0 = self.modules.dfm(logits, hidden)
        log_probs = torch.log(final_probs.clamp_min(1e-8))

        return {
            "log_probs": log_probs,
            "wav_lens": wav_lens,
            "hidden": hidden,
            "logits": logits,
            "alpha0": alpha0,
            "p0": p0,
            "final_probs": final_probs,
        }

    def compute_objectives(self, predictions, batch, stage):
        log_probs = predictions["log_probs"]
        pout_lens = predictions["wav_lens"]
        hidden = predictions["hidden"]
        alpha0 = predictions["alpha0"]
        p0 = predictions["p0"]
        final_probs = predictions["final_probs"]

        phns, phn_lens = batch.phn_encoded

        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            phns = self.hparams.wav_augment.replicate_labels(phns)
            phn_lens = self.hparams.wav_augment.replicate_labels(phn_lens)

        ctc_loss = self.hparams.compute_cost(log_probs, phns, pout_lens, phn_lens)

        frame_ids, frame_mask = self.build_frame_targets(batch, hidden, pout_lens)
        p1 = self.make_target_simplex(frame_ids)

        t = torch.rand(1, device=hidden.device).item()
        xt, alpha1, alpha_t = self.modules.dfm.sample_dirichlet_bridge(alpha0, p1, t)

        v_pred = self.modules.dfm.velocity(xt, t, hidden)

        # v3 still uses a practical target velocity for stability
        v_target = p1 - p0

        fm_sq = (v_pred - v_target) ** 2
        fm_sq = fm_sq.mean(dim=-1)
        fm_loss = (fm_sq * frame_mask).sum() / frame_mask.sum().clamp_min(1.0)

        reg_loss = F.mse_loss(final_probs, p0)

        loss = (
            self.hparams.lambda_ctc * ctc_loss
            + self.hparams.lambda_fm * fm_loss
            + self.hparams.lambda_reg * reg_loss
        )

        self.current_batch_stats = {
            "ctc_loss": ctc_loss.detach(),
            "fm_loss": fm_loss.detach(),
            "reg_loss": reg_loss.detach(),
            "p0": p0.detach(),
            "xt": xt.detach(),
            "final_probs": final_probs.detach(),
            "t": t,
        }

        self.ctc_metrics.append(batch.id, log_probs, phns, pout_lens, phn_lens)

        if stage != sb.Stage.TRAIN:
            sequence = sb.decoders.ctc_greedy_decode(
                log_probs, pout_lens, blank_id=self.hparams.blank_index
            )
            self.per_metrics.append(
                ids=batch.id,
                predict=sequence,
                target=phns,
                target_len=phn_lens,
                ind2lab=self.label_encoder.decode_ndim,
            )

            self._update_monitor_sums(
                ctc_loss=ctc_loss,
                fm_loss=fm_loss,
                reg_loss=reg_loss,
                p0=p0,
                xt=xt,
                final_probs=final_probs,
                t_value=t,
                dfm_grad_norm=None,
            )

        return loss

    def fit_batch(self, batch):
        outputs = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)

        self.optimizer.zero_grad()
        loss.backward()

        dfm_grad_norm = self._get_dfm_grad_norm()
        self._update_monitor_sums(
            ctc_loss=self.current_batch_stats["ctc_loss"],
            fm_loss=self.current_batch_stats["fm_loss"],
            reg_loss=self.current_batch_stats["reg_loss"],
            p0=self.current_batch_stats["p0"],
            xt=self.current_batch_stats["xt"],
            final_probs=self.current_batch_stats["final_probs"],
            t_value=self.current_batch_stats["t"],
            dfm_grad_norm=dfm_grad_norm,
        )

        self.optimizer.step()
        return loss.detach()

    def evaluate_batch(self, batch, stage):
        with torch.no_grad():
            outputs = self.compute_forward(batch, stage=stage)
            loss = self.compute_objectives(outputs, batch, stage=stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch):
        self.ctc_metrics = self.hparams.ctc_stats()
        self._init_monitor_sums()

        if stage != sb.Stage.TRAIN:
            self.per_metrics = self.hparams.per_stats()

    def on_stage_end(self, stage, stage_loss, epoch):
        monitor_stats = self._get_monitor_averages()

        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
            self.train_monitor_stats = monitor_stats
        else:
            per = self.per_metrics.summarize("error_rate")

        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(per)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats={
                    "loss": self.train_loss,
                    "ctc_loss": self.train_monitor_stats["ctc_loss"],
                    "fm_loss": self.train_monitor_stats["fm_loss"],
                    "reg_loss": self.train_monitor_stats["reg_loss"],
                    "flow_delta": self.train_monitor_stats["flow_delta"],
                    "p0_row_sum_mean": self.train_monitor_stats["p0_row_sum_mean"],
                    "xt_row_sum_mean": self.train_monitor_stats["xt_row_sum_mean"],
                    "final_row_sum_mean": self.train_monitor_stats["final_row_sum_mean"],
                    "final_min": self.train_monitor_stats["final_min"],
                    "t_mean": self.train_monitor_stats["t_value"],
                    "dfm_grad_norm": self.train_monitor_stats["dfm_grad_norm"],
                    "p0_entropy": self.train_monitor_stats["p0_entropy"],
                    "xt_entropy": self.train_monitor_stats["xt_entropy"],
                    "final_entropy": self.train_monitor_stats["final_entropy"],
                },
                valid_stats={
                    "loss": stage_loss,
                    "PER": per,
                    "ctc_loss": monitor_stats["ctc_loss"],
                    "fm_loss": monitor_stats["fm_loss"],
                    "reg_loss": monitor_stats["reg_loss"],
                    "flow_delta": monitor_stats["flow_delta"],
                    "p0_row_sum_mean": monitor_stats["p0_row_sum_mean"],
                    "xt_row_sum_mean": monitor_stats["xt_row_sum_mean"],
                    "final_row_sum_mean": monitor_stats["final_row_sum_mean"],
                    "final_min": monitor_stats["final_min"],
                    "t_mean": monitor_stats["t_value"],
                    "p0_entropy": monitor_stats["p0_entropy"],
                    "xt_entropy": monitor_stats["xt_entropy"],
                    "final_entropy": monitor_stats["final_entropy"],
                },
            )

            self.checkpointer.save_and_keep_only(
                meta={"PER": per},
                min_keys=["PER"],
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats={
                    "loss": stage_loss,
                    "PER": per,
                    "ctc_loss": monitor_stats["ctc_loss"],
                    "fm_loss": monitor_stats["fm_loss"],
                    "reg_loss": monitor_stats["reg_loss"],
                    "flow_delta": monitor_stats["flow_delta"],
                    "p0_row_sum_mean": monitor_stats["p0_row_sum_mean"],
                    "xt_row_sum_mean": monitor_stats["xt_row_sum_mean"],
                    "final_row_sum_mean": monitor_stats["final_row_sum_mean"],
                    "final_min": monitor_stats["final_min"],
                    "t_mean": monitor_stats["t_value"],
                    "p0_entropy": monitor_stats["p0_entropy"],
                    "xt_entropy": monitor_stats["xt_entropy"],
                    "final_entropy": monitor_stats["final_entropy"],
                },
            )
            if if_main_process():
                with open(self.hparams.test_wer_file, "w", encoding="utf-8") as w:
                    w.write("CTC loss stats:\n")
                    self.ctc_metrics.write_stats(w)
                    w.write("\nPER stats:\n")
                    self.per_metrics.write_stats(w)
                    print("CTC and PER stats written to ", self.hparams.test_wer_file)


def dataio_prep(hparams):
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["train_annotation"],
        replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        train_data = train_data.filtered_sorted(sort_key="duration")
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(sort_key="duration", reverse=True)
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
    @sb.utils.data_pipeline.provides("phn_list", "phn_encoded", "phn_end_list")
    def text_pipeline(phn, ground_truth_phn_ends):
        phn_list = phn.strip().split()
        yield phn_list

        phn_encoded = label_encoder.encode_sequence_torch(phn_list)
        yield phn_encoded

        phn_end_list = [int(x) for x in ground_truth_phn_ends.strip().split()]
        yield phn_end_list

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