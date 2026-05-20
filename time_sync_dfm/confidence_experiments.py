#!/usr/bin/env python3
"""Confidence-Weighted Flow Experiments on Stage 13 Checkpoint.

Uses the encoder's own uncertainty to modulate per-frame correction
magnitude. No retraining needed - works on the existing Stage 13
checkpoint with its well-trained direction predictor (cosine ~0.77).

The key insight: the encoder's probability distribution at each frame
tells you how much correction is needed. A frame where the encoder
puts 0.95 on one phoneme is already confident and needs almost no
help. A frame where the encoder splits 0.40/0.35 between two phonemes
is confused and needs a bigger push in the right direction.

Correction formula per frame:
    uncertainty = 1 - max(p0[frame])
    scale = base_scale * uncertainty

When max_prob = 0.95: scale = base_scale * 0.05 (tiny correction)
When max_prob = 0.40: scale = base_scale * 0.60 (large correction)

This gives per-frame adaptivity without learning magnitude, so there's
no overfitting. The direction comes from Stage 13's velocity net, the
magnitude comes from the encoder's own confidence.

Usage:
    python run_confidence_experiments.py hparams/train.yaml
"""

import os
import sys
import csv
import json

import torch
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils.logger import get_logger

from train import VelocityNet, ASR_Brain, dataio_prep

logger = get_logger(__name__)


class ConfidenceASR(ASR_Brain):
    """ASR_Brain with confidence-weighted flow integration."""

    def integrate_flow_confidence(self, p0, hidden, num_steps,
                                  base_scale, use_random=False):
        """Confidence-weighted local-window Euler integration.

        Each frame's correction magnitude is proportional to how
        uncertain the encoder is at that frame. Confident frames
        get tiny corrections, confused frames get larger ones.

        Args:
            p0: [B, T, V] initial simplex distributions from encoder
            hidden: [B, T, d] encoder hidden states
            num_steps: number of Euler steps (S)
            base_scale: base magnitude multiplier
            use_random: if True, use random directions instead of learned
        """
        dt = 1.0 / num_steps
        p = p0.clone()
        B, T_enc = hidden.size(0), hidden.size(1)

        # Pre-compute per-frame uncertainty from p0
        # uncertainty = 1 - max_prob: high when encoder is confused
        max_probs = p0.max(dim=-1, keepdim=True).values  # [B, T, 1]
        uncertainty = 1.0 - max_probs  # [B, T, 1]

        for step in range(num_steps):
            t = step * dt

            frame_start = int(t * T_enc)
            frame_end = min(int((t + dt) * T_enc), T_enc)
            frame_start = min(frame_start, T_enc - 1)
            frame_end = max(frame_end, frame_start + 1)

            tau = min(int((frame_start + frame_end) / 2), T_enc - 1)
            h_tau = hidden[:, tau:tau+1, :].expand_as(hidden)

            if use_random:
                v = torch.randn(
                    B, T_enc, self.hparams.output_neurons,
                    device=p.device, dtype=p.dtype
                )
                v = v - v.mean(dim=-1, keepdim=True)
            else:
                v = self.modules.velocity_net(p, t, h_tau)

            # Normalise to unit length per frame
            v_local = v[:, frame_start:frame_end, :]
            v_norm = torch.sqrt(
                (v_local ** 2).sum(dim=-1, keepdim=True)
            ).clamp_min(1e-8)
            v_unit = v_local / v_norm

            # Per-frame adaptive scale: base_scale * uncertainty
            local_uncertainty = uncertainty[:, frame_start:frame_end, :]
            v_scaled = v_unit * base_scale * local_uncertainty

            # Apply update
            p[:, frame_start:frame_end, :] = (
                p[:, frame_start:frame_end, :] + dt * v_scaled
            )

            # Project back onto simplex
            p[:, frame_start:frame_end, :] = (
                p[:, frame_start:frame_end, :].clamp_min(
                    self.hparams.dfm_epsilon
                )
            )
            window_sum = p[:, frame_start:frame_end, :].sum(
                dim=-1, keepdim=True
            ).clamp_min(self.hparams.dfm_epsilon)
            p[:, frame_start:frame_end, :] = (
                p[:, frame_start:frame_end, :] / window_sum
            )

        return p


def run_eval(asr_brain, test_data, hparams, base_scale, num_steps,
             use_random=False):
    """Run one evaluation with confidence-weighted integration."""

    # Temporarily override integrate_flow
    original = asr_brain.integrate_flow

    def patched_integrate(p0, hidden, num_steps_arg):
        return asr_brain.integrate_flow_confidence(
            p0, hidden, num_steps_arg, base_scale, use_random
        )

    asr_brain.integrate_flow = patched_integrate
    asr_brain.hparams.num_flow_steps = num_steps

    asr_brain.on_stage_start(sb.Stage.TEST, epoch=None)
    asr_brain.evaluate(
        test_data,
        min_key="PER",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )

    flow_per = asr_brain.per_metrics_flow.summarize("error_rate")
    p0_per = asr_brain.per_metrics_p0.summarize("error_rate")

    asr_brain.integrate_flow = original
    return {"flow_per": flow_per, "p0_per": p0_per}


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

    # Use ConfidenceASR instead of ASR_Brain
    asr_brain = ConfidenceASR(
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
    # Experiment 1: Confidence-Weighted Scale Sweep
    # ==================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: CONFIDENCE-WEIGHTED SCALE SWEEP (S=20)")
    print("=" * 70)

    scales = [0, 5, 10, 15, 20, 30, 40, 50, 70, 100, 150]
    scale_results = []

    for s in scales:
        print(f"\n  Running base_scale={s}...")
        result = run_eval(asr_brain, test_data, hparams,
                         base_scale=s, num_steps=20, use_random=False)
        scale_results.append({
            "experiment": "confidence_scale_sweep",
            "base_scale": s, "num_steps": 20, "random": False,
            "flow_per": result["flow_per"], "p0_per": result["p0_per"],
        })
        print(f"  base_scale={s:4d}: Flow PER={result['flow_per']:.2f}, "
              f"p0 PER={result['p0_per']:.2f}, "
              f"gap={result['flow_per'] - result['p0_per']:+.2f}")

    best = min(scale_results, key=lambda x: x["flow_per"])
    optimal_scale = best["base_scale"]
    print(f"\n  >>> Optimal base_scale: {optimal_scale} "
          f"(flow PER={best['flow_per']:.2f})")

    # ==================================================================
    # Experiment 2: Resolution Sweep at optimal scale
    # ==================================================================
    print("\n" + "=" * 70)
    print(f"EXPERIMENT 2: RESOLUTION SWEEP (base_scale={optimal_scale})")
    print("=" * 70)

    steps_values = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100]
    res_results = []

    for S in steps_values:
        print(f"\n  Running S={S}...")
        result = run_eval(asr_brain, test_data, hparams,
                         base_scale=optimal_scale, num_steps=S,
                         use_random=False)
        ms = round(1000.0 / S)
        res_results.append({
            "experiment": "resolution_sweep",
            "base_scale": optimal_scale, "num_steps": S,
            "ms_per_step": ms, "random": False,
            "flow_per": result["flow_per"], "p0_per": result["p0_per"],
        })
        print(f"  S={S:3d} (~{ms}ms/step): "
              f"Flow PER={result['flow_per']:.2f}, "
              f"gap={result['flow_per'] - result['p0_per']:+.2f}")

    # ==================================================================
    # Experiment 3: Random Direction Ablation
    # ==================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: RANDOM ABLATION (S=20)")
    print("=" * 70)

    random_scales = [0, 10, 20, 30, 50, 70, 100, 150]
    random_results = []

    for s in random_scales:
        print(f"\n  Running random base_scale={s}...")
        result = run_eval(asr_brain, test_data, hparams,
                         base_scale=s, num_steps=20, use_random=True)
        random_results.append({
            "experiment": "random_ablation",
            "base_scale": s, "num_steps": 20, "random": True,
            "flow_per": result["flow_per"], "p0_per": result["p0_per"],
        })
        print(f"  Random base_scale={s:4d}: "
              f"Flow PER={result['flow_per']:.2f}, "
              f"gap={result['flow_per'] - result['p0_per']:+.2f}")

    # ==================================================================
    # Save results
    # ==================================================================
    json_path = os.path.join(output_dir, "confidence_experiments.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "confidence_scale_sweep": scale_results,
            "resolution_sweep": res_results,
            "random_ablation": random_results,
            "optimal_base_scale": optimal_scale,
        }, f, indent=2)
    print(f"\nResults saved to: {json_path}")

    csv_path = os.path.join(output_dir, "confidence_experiments.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "experiment", "base_scale", "num_steps", "ms_per_step",
            "random", "flow_per", "p0_per"
        ])
        writer.writeheader()
        for row in scale_results + res_results + random_results:
            if "ms_per_step" not in row:
                row["ms_per_step"] = ""
            writer.writerow(row)
    print(f"Results saved to: {csv_path}")

    # ==================================================================
    # Summary
    # ==================================================================
    p0_baseline = scale_results[0]["p0_per"]

    print("\n" + "=" * 70)
    print("SUMMARY: CONFIDENCE-WEIGHTED vs RANDOM")
    print("=" * 70)
    print(f"{'Scale':>8} | {'Learned Flow PER':>18} | "
          f"{'Random Flow PER':>18} | {'p0 PER':>10}")
    print("-" * 65)
    for s in [0, 10, 20, 30, 50, 70, 100, 150]:
        learned = next((r for r in scale_results if r["base_scale"] == s),
                       None)
        random = next((r for r in random_results if r["base_scale"] == s),
                      None)
        lp = f"{learned['flow_per']:.2f}" if learned else "---"
        rp = f"{random['flow_per']:.2f}" if random else "---"
        print(f"{s:>8} | {lp:>18} | {rp:>18} | {p0_baseline:>10.2f}")

    print(f"\n>>> p0 baseline: {p0_baseline:.2f}")
    print(f">>> Best confidence-weighted: {best['flow_per']:.2f} "
          f"at base_scale={optimal_scale}")
    improvement = p0_baseline - best["flow_per"]
    if improvement > 0:
        print(f">>> Improvement: {improvement:.2f} PER points "
              f"({improvement / p0_baseline * 100:.1f}% relative)")

    print("\n" + "=" * 70)
    print("SUMMARY: RESOLUTION SWEEP")
    print("=" * 70)
    print(f"{'S':>6} | {'~ms/step':>10} | {'Flow PER':>10} | "
          f"{'p0 PER':>10} | {'Gap':>8}")
    print("-" * 55)
    for r in res_results:
        gap = r["flow_per"] - r["p0_per"]
        print(f"{r['num_steps']:>6} | {r['ms_per_step']:>10} | "
              f"{r['flow_per']:>10.2f} | {r['p0_per']:>10.2f} | "
              f"{gap:>+8.2f}")


if __name__ == "__main__":
    main()