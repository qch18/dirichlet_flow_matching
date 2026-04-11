#!/usr/bin/env python3
"""Stage 2: baseline CTC ASR on TIMIT with Dirichlet parameterisation.

This stage adds:
- Dirichlet parameters A = softplus(logits) + eps
- initial simplex state p0 = A / sum(A)

Still NO bridge, NO velocity net, NO FM loss yet.
Training is still plain CTC, but now through p0.
"""

import os
import sys

import torch
import torch.nn.functional as F
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.utils.distributed import if_main_process, run_on_main
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


class ASR_Brain(sb.Brain):
    def logits_to_dirichlet(self, logits):
        return F.softplus(logits) + self.hparams.dfm_epsilon

    def dirichlet_mean(self, alpha):
        return alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(
            self.hparams.dfm_epsilon
        )

    def _init_monitor_sums(self):
        self.monitor_sums = {
            "ctc_loss": 0.0,
            "grad_norm": 0.0,
            "logits_mean": 0.0,
            "logits_std": 0.0,
            "alpha_mean": 0.0,
            "alpha_std": 0.0,
            "p0_entropy": 0.0,
            "p0_row_sum_mean": 0.0,
            "p0_min": 0.0,
            "blank_prob_mean": 0.0,
            "max_prob_mean": 0.0,
            "num_batches": 0,
        }

    def _update_monitor_sums(
        self,
        ctc_loss,
        logits,
        alpha,
        p0,
        log_probs,
        grad_norm=None,
    ):
        probs = log_probs.exp()
        p0_entropy = -(p0 * p0.clamp_min(1e-8).log()).sum(dim=-1).mean()
        blank_prob_mean = probs[..., self.hparams.blank_index].mean()
        max_prob_mean = probs.max(dim=-1).values.mean()

        self.monitor_sums["ctc_loss"] += float(ctc_loss.detach().cpu())
        self.monitor_sums["logits_mean"] += float(logits.mean().detach().cpu())
        self.monitor_sums["logits_std"] += float(logits.std().detach().cpu())
        self.monitor_sums["alpha_mean"] += float(alpha.mean().detach().cpu())
        self.monitor_sums["alpha_std"] += float(alpha.std().detach().cpu())
        self.monitor_sums["p0_entropy"] += float(p0_entropy.detach().cpu())
        self.monitor_sums["p0_row_sum_mean"] += float(
            p0.sum(dim=-1).mean().detach().cpu()
        )
        self.monitor_sums["p0_min"] += float(p0.min().detach().cpu())
        self.monitor_sums["blank_prob_mean"] += float(blank_prob_mean.detach().cpu())
        self.monitor_sums["max_prob_mean"] += float(max_prob_mean.detach().cpu())

        if grad_norm is not None:
            self.monitor_sums["grad_norm"] += float(grad_norm)

        self.monitor_sums["num_batches"] += 1

    def _get_monitor_averages(self):
        n = max(self.monitor_sums["num_batches"], 1)
        return {
            "ctc_loss": self.monitor_sums["ctc_loss"] / n,
            "grad_norm": self.monitor_sums["grad_norm"] / n,
            "logits_mean": self.monitor_sums["logits_mean"] / n,
            "logits_std": self.monitor_sums["logits_std"] / n,
            "alpha_mean": self.monitor_sums["alpha_mean"] / n,
            "alpha_std": self.monitor_sums["alpha_std"] / n,
            "p0_entropy": self.monitor_sums["p0_entropy"] / n,
            "p0_row_sum_mean": self.monitor_sums["p0_row_sum_mean"] / n,
            "p0_min": self.monitor_sums["p0_min"] / n,
            "blank_prob_mean": self.monitor_sums["blank_prob_mean"] / n,
            "max_prob_mean": self.monitor_sums["max_prob_mean"] / n,
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

    def compute_forward(self, batch, stage):
        "Computes hidden states, logits, Dirichlet params, and simplex state p0."
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig

        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            wavs, wav_lens = self.hparams.wav_augment(wavs, wav_lens)

        feats = self.hparams.compute_features(wavs)
        feats = self.modules.normalize(feats, wav_lens)

        hidden = self.modules.model(feats)           # H
        logits = self.modules.output(hidden)         # Z
        alpha = self.logits_to_dirichlet(logits)     # A
        p0 = self.dirichlet_mean(alpha)              # initial simplex state
        log_probs = torch.log(p0.clamp_min(1e-8))    # CTC now uses p0

        return {
            "hidden": hidden,
            "logits": logits,
            "alpha": alpha,
            "p0": p0,
            "log_probs": log_probs,
            "wav_lens": wav_lens,
        }

    def compute_objectives(self, predictions, batch, stage):
        "Computes plain CTC loss on p0."
        log_probs = predictions["log_probs"]
        logits = predictions["logits"]
        alpha = predictions["alpha"]
        p0 = predictions["p0"]
        pout_lens = predictions["wav_lens"]

        phns, phn_lens = batch.phn_encoded

        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            phns = self.hparams.wav_augment.replicate_labels(phns)
            phn_lens = self.hparams.wav_augment.replicate_labels(phn_lens)

        ctc_loss = self.hparams.compute_cost(log_probs, phns, pout_lens, phn_lens)

        self.current_batch_stats = {
            "ctc_loss": ctc_loss.detach(),
            "logits": logits.detach(),
            "alpha": alpha.detach(),
            "p0": p0.detach(),
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
                ctc_loss=ctc_loss,
                logits=logits,
                alpha=alpha,
                p0=p0,
                log_probs=log_probs,
                grad_norm=None,
            )

        return ctc_loss

    def fit_batch(self, batch):
        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)

        self.optimizer.zero_grad()
        loss.backward()

        grad_norm = self._get_grad_norm()

        self._update_monitor_sums(
            ctc_loss=self.current_batch_stats["ctc_loss"],
            logits=self.current_batch_stats["logits"],
            alpha=self.current_batch_stats["alpha"],
            p0=self.current_batch_stats["p0"],
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
                train_stats={
                    "loss": self.train_loss,
                    "ctc_loss": self.train_monitor_stats["ctc_loss"],
                    "grad_norm": self.train_monitor_stats["grad_norm"],
                    "logits_mean": self.train_monitor_stats["logits_mean"],
                    "logits_std": self.train_monitor_stats["logits_std"],
                    "alpha_mean": self.train_monitor_stats["alpha_mean"],
                    "alpha_std": self.train_monitor_stats["alpha_std"],
                    "p0_entropy": self.train_monitor_stats["p0_entropy"],
                    "p0_row_sum_mean": self.train_monitor_stats["p0_row_sum_mean"],
                    "p0_min": self.train_monitor_stats["p0_min"],
                    "blank_prob_mean": self.train_monitor_stats["blank_prob_mean"],
                    "max_prob_mean": self.train_monitor_stats["max_prob_mean"],
                },
                valid_stats={
                    "loss": stage_loss,
                    "PER": per,
                    "ctc_loss": monitor_stats["ctc_loss"],
                    "logits_mean": monitor_stats["logits_mean"],
                    "logits_std": monitor_stats["logits_std"],
                    "alpha_mean": monitor_stats["alpha_mean"],
                    "alpha_std": monitor_stats["alpha_std"],
                    "p0_entropy": monitor_stats["p0_entropy"],
                    "p0_row_sum_mean": monitor_stats["p0_row_sum_mean"],
                    "p0_min": monitor_stats["p0_min"],
                    "blank_prob_mean": monitor_stats["blank_prob_mean"],
                    "max_prob_mean": monitor_stats["max_prob_mean"],
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
                    "logits_mean": monitor_stats["logits_mean"],
                    "logits_std": monitor_stats["logits_std"],
                    "alpha_mean": monitor_stats["alpha_mean"],
                    "alpha_std": monitor_stats["alpha_std"],
                    "p0_entropy": monitor_stats["p0_entropy"],
                    "p0_row_sum_mean": monitor_stats["p0_row_sum_mean"],
                    "p0_min": monitor_stats["p0_min"],
                    "blank_prob_mean": monitor_stats["blank_prob_mean"],
                    "max_prob_mean": monitor_stats["max_prob_mean"],
                },
            )

            if if_main_process():
                with open(
                    self.hparams.test_wer_file, "w", encoding="utf-8"
                ) as w:
                    w.write("CTC loss stats:\n")
                    self.ctc_metrics.write_stats(w)
                    w.write("\nPER stats:\n")
                    self.per_metrics.write_stats(w)
                    print(
                        "CTC and PER stats written to ",
                        self.hparams.test_wer_file,
                    )


def dataio_prep(hparams):
    "Creates the datasets and their data processing pipelines."

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

    @sb.utils.data_pipeline.takes("phn")
    @sb.utils.data_pipeline.provides("phn_list", "phn_encoded")
    def text_pipeline(phn):
        phn_list = phn.strip().split()
        yield phn_list
        phn_encoded = label_encoder.encode_sequence_torch(phn_list)
        yield phn_encoded

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[train_data],
        output_key="phn_list",
        special_labels={"blank_label": hparams["blank_index"]},
        sequence_input=True,
    )

    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig", "phn_encoded"])

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