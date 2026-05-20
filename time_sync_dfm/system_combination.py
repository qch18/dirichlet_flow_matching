#!/usr/bin/env python3
"""System Combination Experiments for Time-Synchronous DFM.

Combines CTC (p0) and flow (p_flow) probability distributions before
decoding. This is standard practice in ASR (used to combine CTC +
attention models, multiple CTC models, etc).

Three combination strategies:
  1. Linear interpolation: p_combined = α * p0 + (1-α) * p_flow
  2. Log-linear interpolation: log p = α * log p0 + (1-α) * log p_flow
  3. Max combination: p_combined = max(p0, p_flow) per frame per class

Each is swept across α values from 0.0 to 1.0.
α=1.0 = pure CTC (p0), α=0.0 = pure flow (p_flow).

The combined distribution is decoded using BOTH:
  - CTC greedy decode (for the CTC-style implicit alignment)
  - Argmax + collapse (for the flow-style explicit alignment)

Usage:
    python system_combination.py hparams/train.yaml

Works on existing Stage 13 checkpoint with no retraining.
"""

import os
import sys
import csv
import json

import torch
import torch.nn.functional as F
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils.logger import get_logger

from train import VelocityNet, ASR_Brain, dataio_prep

logger = get_logger(__name__)


class CombinationASR(ASR_Brain):
    """ASR_Brain with system combination at decode time."""

    def compute_objectives_combined(self, predictions, batch, stage,
                                     alpha, combination_method,
                                     base_scale):
        """Compute PER using combined p0 + p_flow distributions."""
        p0 = predictions["p0"]
        pout_lens = predictions["wav_lens"]
        hidden = predictions["hidden"]

        phns, phn_lens = batch.phn_encoded

        # Run flat-scale flow integration (no confidence weighting)
        p_flow = self.integrate_flow_flat(
            p0, hidden, self.hparams.num_flow_steps, base_scale
        )

        # ============================================================
        # System Combination: blend p0 and p_flow
        # ============================================================
        if combination_method == "linear":
            # Linear interpolation on probability simplex
            p_combined = alpha * p0 + (1.0 - alpha) * p_flow

        elif combination_method == "log_linear":
            # Log-linear interpolation (geometric mean, renormalized)
            log_p0 = torch.log(p0.clamp_min(1e-8))
            log_pf = torch.log(p_flow.clamp_min(1e-8))
            log_combined = alpha * log_p0 + (1.0 - alpha) * log_pf
            p_combined = torch.softmax(log_combined, dim=-1)

        elif combination_method == "max":
            # Per-class max (union of confident predictions)
            p_combined = torch.max(p0, p_flow)
            # Renormalize
            p_combined = p_combined / p_combined.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-8)

        # ============================================================
        # Decode combined distribution with CTC greedy
        # ============================================================
        log_probs_combined = torch.log(p_combined.clamp_min(1e-8))
        sequence_ctc = sb.decoders.ctc_greedy_decode(
            log_probs_combined, pout_lens,
            blank_id=self.hparams.blank_index
        )
        self.per_metrics_ctc_combined.append(
            ids=batch.id,
            predict=sequence_ctc,
            target=phns,
            target_len=phn_lens,
            ind2lab=self.label_encoder.decode_ndim,
        )

        # ============================================================
        # Decode combined distribution with argmax + collapse
        # ============================================================
        sequence_flow = self.decode_flow_output(p_combined, pout_lens)
        self.per_metrics_flow_combined.append(
            ids=batch.id,
            predict=sequence_flow,
            target=phns,
            target_len=phn_lens,
            ind2lab=self.label_encoder.decode_ndim,
        )

        return 0.0  # loss not needed for eval

    def integrate_flow_flat(self, p0, hidden, num_steps, base_scale):
        """Flat-scale integration for system combination.

        Unlike confidence-weighted integration, this applies the SAME
        scale to every frame. This produces a p_flow that's genuinely
        different from p0, which is what system combination needs.

        When using the flow output ALONE, confidence weighting is better
        (protects confident frames). But when COMBINING with CTC, the
        α weighting in the combination handles that protection instead.
        """
        dt = 1.0 / num_steps
        p = p0.clone()
        B, T_enc = hidden.size(0), hidden.size(1)

        for step in range(num_steps):
            t = step * dt
            frame_start = int(t * T_enc)
            frame_end = min(int((t + dt) * T_enc), T_enc)
            frame_start = min(frame_start, T_enc - 1)
            frame_end = max(frame_end, frame_start + 1)

            tau = min(int((frame_start + frame_end) / 2), T_enc - 1)
            h_tau = hidden[:, tau:tau+1, :].expand_as(hidden)

            v = self.modules.velocity_net(p, t, h_tau)

            v_local = v[:, frame_start:frame_end, :]
            v_norm = torch.sqrt(
                (v_local ** 2).sum(dim=-1, keepdim=True)
            ).clamp_min(1e-8)
            v_unit = v_local / v_norm

            # Flat scale: same correction magnitude for every frame
            v_scaled = v_unit * base_scale

            p[:, frame_start:frame_end, :] = (
                p[:, frame_start:frame_end, :] + dt * v_scaled
            )
            p[:, frame_start:frame_end, :] = (
                p[:, frame_start:frame_end, :].clamp_min(
                    self.hparams.dfm_epsilon
                )
            )
            ws = p[:, frame_start:frame_end, :].sum(
                dim=-1, keepdim=True
            ).clamp_min(self.hparams.dfm_epsilon)
            p[:, frame_start:frame_end, :] = (
                p[:, frame_start:frame_end, :] / ws
            )

        return p


def run_combination_eval(asr_brain, test_data, hparams,
                          alpha, combination_method, base_scale):
    """Run one combination evaluation."""

    # Init metrics
    asr_brain.per_metrics_ctc_combined = hparams["per_stats"]()
    asr_brain.per_metrics_flow_combined = hparams["per_stats"]()

    # Critical: set modules to eval mode so normalizer uses frozen
    # training statistics instead of updating with test data
    asr_brain.modules.eval()

    with torch.no_grad():
        for batch in sb.dataio.dataloader.make_dataloader(
            test_data, **hparams["test_dataloader_opts"]
        ):
            batch = batch.to(asr_brain.device)
            wavs, wav_lens = batch.sig

            feats = hparams["compute_features"](wavs)
            feats = asr_brain.modules.normalize(feats, wav_lens)

            hidden = asr_brain.modules.model(feats)
            logits = asr_brain.modules.output(hidden)
            alpha_dir = F.softplus(logits) + hparams["dfm_epsilon"]
            p0 = alpha_dir / alpha_dir.sum(dim=-1, keepdim=True).clamp_min(
                hparams["dfm_epsilon"]
            )

            predictions = {
                "p0": p0,
                "hidden": hidden,
                "wav_lens": wav_lens,
            }

            asr_brain.compute_objectives_combined(
                predictions, batch, sb.Stage.TEST,
                alpha, combination_method, base_scale
            )

    ctc_per = asr_brain.per_metrics_ctc_combined.summarize("error_rate")
    flow_per = asr_brain.per_metrics_flow_combined.summarize("error_rate")

    return {"ctc_combined_per": ctc_per, "flow_combined_per": flow_per}


def main():
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    from timit_prepare import prepare_timit

    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    run_on_main(prepare_timit, kwargs={
        "data_folder": hparams["data_folder"],
        "save_json_train": hparams["train_annotation"],
        "save_json_valid": hparams["valid_annotation"],
        "save_json_test": hparams["test_annotation"],
        "skip_prep": hparams["skip_prep"],
        "uppercase": hparams["uppercase"],
    })
    run_on_main(hparams["prepare_noise_data"])

    train_data, valid_data, test_data, label_encoder = dataio_prep(hparams)

    asr_brain = CombinationASR(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    asr_brain.label_encoder = label_encoder
    asr_brain.checkpointer.recover_if_possible(min_key="PER")

    output_dir = hparams["output_folder"]

    # ==================================================================
    # 2D Grid Search: sweep both base_scale and α
    # For each scale, the flow produces a different distribution.
    # Higher scale = more different from p0 = needs higher α (more CTC).
    # ==================================================================
    print("\n" + "=" * 70)
    print("2D GRID SEARCH: LINEAR INTERPOLATION")
    print("  Sweeping base_scale × α to find optimal combination")
    print("  α=1.0 = pure CTC,  α=0.0 = pure flow")
    print("=" * 70)

    scale_values = [3, 5, 7, 10, 15, 20, 30, 50]
    alpha_values = [1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5]
    grid_results = []

    for s in scale_values:
        print(f"\n  --- Scale={s} ---")
        for a in alpha_values:
            r = run_combination_eval(
                asr_brain, test_data, hparams,
                alpha=a, combination_method="linear",
                base_scale=s
            )
            per = r["ctc_combined_per"]
            grid_results.append({
                "experiment": "grid_search",
                "base_scale": s,
                "alpha": a,
                "combination": "linear",
                "ctc_decode_per": per,
                "flow_decode_per": r["flow_combined_per"],
            })
            marker = " <<<" if per < 15.65 else ""
            print(f"    α={a:.2f}: PER={per:.2f}{marker}")

    best = min(grid_results, key=lambda x: x["ctc_decode_per"])
    print(f"\n  >>> Best: scale={best['base_scale']}, α={best['alpha']}, "
          f"PER={best['ctc_decode_per']:.2f}")

    # ==================================================================
    # Log-linear at best scale
    # ==================================================================
    best_scale = best["base_scale"]
    print("\n" + "=" * 70)
    print(f"LOG-LINEAR INTERPOLATION (scale={best_scale})")
    print("=" * 70)

    loglin_results = []
    for a in alpha_values:
        r = run_combination_eval(
            asr_brain, test_data, hparams,
            alpha=a, combination_method="log_linear",
            base_scale=best_scale
        )
        per = r["ctc_combined_per"]
        loglin_results.append({
            "experiment": "log_linear",
            "base_scale": best_scale,
            "alpha": a,
            "combination": "log_linear",
            "ctc_decode_per": per,
            "flow_decode_per": r["flow_combined_per"],
        })
        marker = " <<<" if per < 15.65 else ""
        print(f"  α={a:.2f}: PER={per:.2f}{marker}")

    best_loglin = min(loglin_results, key=lambda x: x["ctc_decode_per"])

    # ==================================================================
    # Save results
    # ==================================================================
    all_results = grid_results + loglin_results

    json_path = os.path.join(output_dir, "combination_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "grid_search": grid_results,
            "log_linear": loglin_results,
            "best_linear": {
                "scale": best["base_scale"],
                "alpha": best["alpha"],
                "per": best["ctc_decode_per"],
            },
            "best_log_linear": {
                "scale": best_loglin["base_scale"],
                "alpha": best_loglin["alpha"],
                "per": best_loglin["ctc_decode_per"],
            },
        }, f, indent=2)
    print(f"\nResults saved to: {json_path}")

    csv_path = os.path.join(output_dir, "combination_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "experiment", "base_scale", "alpha", "combination",
            "ctc_decode_per", "flow_decode_per"
        ])
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)
    print(f"Results saved to: {csv_path}")

    # ==================================================================
    # Summary
    # ==================================================================
    print("\n" + "=" * 70)
    print("GRID SEARCH RESULTS (cells showing PER, <<< = better than p0)")
    print("=" * 70)
    print(f"{'Scale':>8} |", end="")
    for a in alpha_values:
        print(f" α={a:.2f}", end="")
    print()
    print("-" * (10 + 8 * len(alpha_values)))

    for s in scale_values:
        print(f"{s:>8} |", end="")
        for a in alpha_values:
            r = next((x for x in grid_results
                      if x["base_scale"] == s and x["alpha"] == a), None)
            if r:
                per = r["ctc_decode_per"]
                if per < 15.60:
                    print(f" {per:5.2f}*", end="")
                else:
                    print(f" {per:6.2f}", end="")
            else:
                print("    ---", end="")
        print()

    p0_per = 15.65
    print(f"\n>>> p0 baseline (no flow): {p0_per:.2f}")
    print(f">>> Best linear combination: PER={best['ctc_decode_per']:.2f} "
          f"(scale={best['base_scale']}, α={best['alpha']})")
    print(f">>> Best log-linear combination: PER={best_loglin['ctc_decode_per']:.2f} "
          f"(scale={best_loglin['base_scale']}, α={best_loglin['alpha']})")

    overall_best_per = min(best["ctc_decode_per"],
                           best_loglin["ctc_decode_per"])
    improvement = p0_per - overall_best_per
    if improvement > 0:
        print(f">>> Improvement: {improvement:.2f} PER points "
              f"({improvement/p0_per*100:.1f}% relative)")


if __name__ == "__main__":
    main()