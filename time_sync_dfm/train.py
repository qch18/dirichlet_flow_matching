#!/usr/bin/env python3
"""Stage 7: CTC ASR on TIMIT with trainable flow-matching loss.

Adds:
- learned velocity network v_theta(pt, t, H)
- target velocity vt* = p1 - p0
- FM loss = ||vpred - vt||^2
- total loss = lambda_ctc * ctc_loss + lambda_fm * fm_loss
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
        v = v - v.mean(dim=-1, keepdim=True)
        return v


class ASR_Brain(sb.Brain):
    def logits_to_dirichlet(self, logits):
        return F.softplus(logits) + self.hparams.dfm_epsilon

    def dirichlet_mean(self, alpha):
        return alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(
            self.hparams.dfm_epsilon
        )

    def bridge_state(self, p0, p1, t):
        return (1.0 - t) * p0 + t * p1

    def target_velocity(self, p0, p1):
        return p1 - p0

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

                phon_id = int(
                    self.label_encoder.encode_sequence_torch([phon])[0].item()
                )

                frame_ids[b, start_frame:end_frame] = phon_id
                frame_mask[b, start_frame:end_frame] = 1.0
                prev_end = end_sample

        return frame_ids, frame_mask

    def make_target_simplex(self, frame_ids):
        p1 = F.one_hot(
            frame_ids, num_classes=self.hparams.output_neurons
        ).float()
        smooth = self.hparams.target_smoothing
        p1 = (1.0 - smooth) * p1 + smooth / self.hparams.output_neurons
        return p1

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

        self.monitor_sums["vpred_mean_abs"] += float(vpred_mean_abs.detach().cpu())
        self.monitor_sums["vpred_l2"] += float(vpred_l2.detach().cpu())
        self.monitor_sums["vpred_sum_mean"] += float(vpred_sum_mean.detach().cpu())
        self.monitor_sums["velocity_mse_monitor"] += float(
            velocity_mse_monitor.detach().cpu()
        )
        self.monitor_sums["velocity_cosine_monitor"] += float(cos.detach().cpu())

        self.monitor_sums["t_mean"] += float(t_value)
        self.monitor_sums["blank_prob_mean"] += float(blank_prob_mean.detach().cpu())
        self.monitor_sums["max_prob_mean"] += float(max_prob_mean.detach().cpu())

        if grad_norm is not None:
            self.monitor_sums["grad_norm"] += float(grad_norm)

        self.monitor_sums["num_batches"] += 1

    def _get_monitor_averages(self):
        n = max(self.monitor_sums["num_batches"], 1)
        return {k: (v / n if k != "num_batches" else v) for k, v in self.monitor_sums.items() if k != "num_batches"}

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

    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig

        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            wavs, wav_lens = self.hparams.wav_augment(wavs, wav_lens)

        feats = self.hparams.compute_features(wavs)
        feats = self.modules.normalize(feats, wav_lens)

        hidden = self.modules.model(feats)
        logits = self.modules.output(hidden)
        alpha = self.logits_to_dirichlet(logits)
        p0 = self.dirichlet_mean(alpha)

        frame_ids, frame_mask = self.build_frame_targets(batch, hidden, wav_lens)
        p1 = self.make_target_simplex(frame_ids)

        t = torch.rand(1, device=hidden.device).item()
        pt = self.bridge_state(p0, p1, t)
        vt = self.target_velocity(p0, p1)
        vpred = self.modules.velocity_net(pt, t, hidden)

        log_probs = torch.log(p0.clamp_min(1e-8))

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
            "log_probs": log_probs,
            "wav_lens": wav_lens,
        }

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

        phns, phn_lens = batch.phn_encoded

        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            phns = self.hparams.wav_augment.replicate_labels(phns)
            phn_lens = self.hparams.wav_augment.replicate_labels(phn_lens)

        ctc_loss = self.hparams.compute_cost(log_probs, phns, pout_lens, phn_lens)

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
            )

        return total_loss

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
        )

        self.optimizer.step()
        return loss.detach()

    def evaluate_batch(self, batch, stage):
        with torch.no_grad():
            predictions = self.compute_forward(batch, stage=stage)
            loss = self.compute_objectives(predictions, batch, stage=stage)
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
                train_stats=self.train_monitor_stats,
                valid_stats={"PER": per, **monitor_stats},
            )

            self.checkpointer.save_and_keep_only(
                meta={"PER": per},
                min_keys=["PER"],
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats={"PER": per, **monitor_stats},
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
        raise NotImplementedError("sorting must be random, ascending or descending")

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["valid_annotation"],
        replacements={"data_root": data_folder},
    ).filtered_sorted(sort_key="duration")

    test_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["test_annotation"],
        replacements={"data_root": data_folder},
    ).filtered_sorted(sort_key="duration")

    datasets = [train_data, valid_data, test_data]
    label_encoder = sb.dataio.encoder.CTCTextEncoder()

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        return sb.dataio.dataio.read_audio(wav)

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    @sb.utils.data_pipeline.takes("phn", "ground_truth_phn_ends")
    @sb.utils.data_pipeline.provides("phn_list", "phn_encoded", "phn_end_list")
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