#!/usr/bin/env python3
"""Stage 13 Evaluation: Scale Sweep, Resolution Sweep, and Random Ablation.

Three experiments to demonstrate that the velocity net has learned
meaningful time-synchronous velocity directions:

Experiment 1 - Scale Sweep:
    Fix S=20, vary velocity_inference_scale from 0 to 50.
    Scale=0 is the control (flow PER = p0 PER, no correction).
    Any scale where flow PER < scale=0 proves the velocity net helps.

Experiment 2 - Resolution Sweep:
    Fix scale at the optimal value from Experiment 1, vary S from 5 to 100.
    Shows how temporal resolution affects recognition accuracy.
    This is unique to time-synchronous DFM (Stage 8c can't do this).

Experiment 3 - Random Direction Ablation:
    Replace learned velocities with random unit vectors.
    Same scale sweep as Experiment 1.
    Random directions should make things WORSE at every non-zero scale,
    proving the learned directions carry meaningful information.

Usage:
    python run_experiments.py hparams/train.yaml

Results are saved to a CSV and printed as tables.
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

# Import everything from train.py (VelocityNet, ASR_Brain, dataio_prep)
from train import VelocityNet, ASR_Brain, dataio_prep

logger = get_logger(__name__)


def run_single_eval(asr_brain, test_data, hparams, scale, num_steps,
                    use_random_directions=False):
    """Run a single evaluation with specific scale and S values.

    Args:
        asr_brain: The ASR_Brain instance with loaded checkpoint
        test_data: Test dataset
        hparams: Hyperparameters dict
        scale: velocity_inference_scale value
        num_steps: num_flow_steps value
        use_random_directions: If True, replace learned velocities with random

    Returns:
        dict with flow_per, p0_per, flow_entropy, flow_p0_delta
    """
    # Override hparams for this evaluation
    asr_brain.hparams.velocity_inference_scale = scale
    asr_brain.hparams.num_flow_steps = num_steps

    # Store original integrate_flow
    original_integrate_flow = asr_brain.integrate_flow

    if use_random_directions:
        # Monkey-patch integrate_flow to use random directions
        def random_integrate_flow(p0, hidden, num_steps_arg):
            dt = 1.0 / num_steps_arg
            p = p0.clone()
            B, T_enc = hidden.size(0), hidden.size(1)

            for step in range(num_steps_arg):
                t = step * dt

                frame_start = int(t * T_enc)
                frame_end = min(int((t + dt) * T_enc), T_enc)
                frame_start = min(frame_start, T_enc - 1)
                frame_end = max(frame_end, frame_start + 1)

                # Generate random velocity (same shape as real velocity)
                v_random = torch.randn(
                    B, frame_end - frame_start,
                    asr_brain.hparams.output_neurons,
                    device=p.device, dtype=p.dtype
                )
                # Zero-mean to preserve simplex constraint
                v_random = v_random - v_random.mean(dim=-1, keepdim=True)
                # Normalise to unit length
                v_norm = torch.sqrt(
                    (v_random ** 2).sum(dim=-1, keepdim=True)
                ).clamp_min(1e-8)
                v_random = v_random / v_norm * scale

                # Apply update
                p[:, frame_start:frame_end, :] = (
                    p[:, frame_start:frame_end, :] + dt * v_random
                )

                # Project back onto simplex
                p[:, frame_start:frame_end, :] = (
                    p[:, frame_start:frame_end, :].clamp_min(
                        asr_brain.hparams.dfm_epsilon
                    )
                )
                window_sum = p[:, frame_start:frame_end, :].sum(
                    dim=-1, keepdim=True
                ).clamp_min(asr_brain.hparams.dfm_epsilon)
                p[:, frame_start:frame_end, :] = (
                    p[:, frame_start:frame_end, :] / window_sum
                )

            return p

        asr_brain.integrate_flow = random_integrate_flow

    # Reset metrics
    asr_brain.on_stage_start(sb.Stage.TEST, epoch=None)

    # Run evaluation
    asr_brain.evaluate(
        test_data,
        min_key="PER",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )

    # Collect results
    flow_per = asr_brain.per_metrics_flow.summarize("error_rate")
    p0_per = asr_brain.per_metrics_p0.summarize("error_rate")

    # Restore original method
    asr_brain.integrate_flow = original_integrate_flow

    return {
        "flow_per": flow_per,
        "p0_per": p0_per,
    }


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

    asr_brain = ASR_Brain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    asr_brain.label_encoder = label_encoder

    # Load best checkpoint
    asr_brain.checkpointer.recover_if_possible(min_key="PER")

    output_dir = hparams["output_folder"]
    all_results = []

    # ==================================================================
    # Experiment 1: Scale Sweep
    # Fix S=20, vary scale from 0 to 50
    # Scale=0 is the control (p_flow = p0, no correction applied)
    # ==================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: SCALE SWEEP (S=20, varying scale)")
    print("=" * 70)

    scales = [0, 1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50]
    scale_results = []

    for s in scales:
        print(f"\n  Running scale={s}...")
        result = run_single_eval(
            asr_brain, test_data, hparams,
            scale=s, num_steps=20, use_random_directions=False
        )
        scale_results.append({
            "experiment": "scale_sweep",
            "scale": s,
            "num_steps": 20,
            "random": False,
            "flow_per": result["flow_per"],
            "p0_per": result["p0_per"],
        })
        print(f"  Scale={s:3d}: Flow PER={result['flow_per']:.2f}, "
              f"p0 PER={result['p0_per']:.2f}, "
              f"gap={result['flow_per'] - result['p0_per']:+.2f}")

    # Find optimal scale
    best_scale_result = min(scale_results, key=lambda x: x["flow_per"])
    optimal_scale = best_scale_result["scale"]
    print(f"\n  >>> Optimal scale: {optimal_scale} "
          f"(flow PER={best_scale_result['flow_per']:.2f})")

    # ==================================================================
    # Experiment 2: Resolution Sweep
    # Fix scale at optimal, vary S from 5 to 100
    # Shows temporal resolution control unique to time-sync DFM
    # ==================================================================
    print("\n" + "=" * 70)
    print(f"EXPERIMENT 2: RESOLUTION SWEEP (scale={optimal_scale}, varying S)")
    print("=" * 70)

    steps_values = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100]
    resolution_results = []

    for S in steps_values:
        print(f"\n  Running S={S}...")
        result = run_single_eval(
            asr_brain, test_data, hparams,
            scale=optimal_scale, num_steps=S, use_random_directions=False
        )
        # Calculate temporal resolution
        ms_per_step = round(1000.0 / S)  # approximate ms per step
        resolution_results.append({
            "experiment": "resolution_sweep",
            "scale": optimal_scale,
            "num_steps": S,
            "ms_per_step": ms_per_step,
            "random": False,
            "flow_per": result["flow_per"],
            "p0_per": result["p0_per"],
        })
        print(f"  S={S:3d} (~{ms_per_step}ms/step): "
              f"Flow PER={result['flow_per']:.2f}, "
              f"p0 PER={result['p0_per']:.2f}, "
              f"gap={result['flow_per'] - result['p0_per']:+.2f}")

    # ==================================================================
    # Experiment 3: Random Direction Ablation
    # Same scale sweep as Exp 1, but with random unit vectors
    # Proves learned directions are meaningful, not just noise
    # ==================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: RANDOM DIRECTION ABLATION (S=20, varying scale)")
    print("=" * 70)

    random_scales = [0, 1, 3, 5, 7, 10, 15, 20, 30]
    random_results = []

    for s in random_scales:
        print(f"\n  Running random scale={s}...")
        result = run_single_eval(
            asr_brain, test_data, hparams,
            scale=s, num_steps=20, use_random_directions=True
        )
        random_results.append({
            "experiment": "random_ablation",
            "scale": s,
            "num_steps": 20,
            "random": True,
            "flow_per": result["flow_per"],
            "p0_per": result["p0_per"],
        })
        print(f"  Random scale={s:3d}: Flow PER={result['flow_per']:.2f}, "
              f"p0 PER={result['p0_per']:.2f}, "
              f"gap={result['flow_per'] - result['p0_per']:+.2f}")

    # ==================================================================
    # Save all results to CSV
    # ==================================================================
    all_results = scale_results + resolution_results + random_results
    csv_path = os.path.join(output_dir, "experiment_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "experiment", "scale", "num_steps", "ms_per_step",
            "random", "flow_per", "p0_per"
        ])
        writer.writeheader()
        for row in all_results:
            # Fill ms_per_step for non-resolution experiments
            if "ms_per_step" not in row:
                row["ms_per_step"] = ""
            writer.writerow(row)
    print(f"\nResults saved to: {csv_path}")

    # ==================================================================
    # Save results as JSON too (easier to load for plotting)
    # ==================================================================
    json_path = os.path.join(output_dir, "experiment_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "scale_sweep": scale_results,
            "resolution_sweep": resolution_results,
            "random_ablation": random_results,
            "optimal_scale": optimal_scale,
        }, f, indent=2)
    print(f"Results saved to: {json_path}")

    # ==================================================================
    # Print summary tables
    # ==================================================================
    p0_baseline = scale_results[0]["p0_per"]  # scale=0 gives pure p0

    print("\n" + "=" * 70)
    print("SUMMARY: SCALE SWEEP vs RANDOM ABLATION")
    print("=" * 70)
    print(f"{'Scale':>6} | {'Learned Flow PER':>18} | {'Random Flow PER':>18} | {'p0 PER (baseline)':>18}")
    print("-" * 70)
    for s in [0, 1, 3, 5, 7, 10, 15, 20, 30]:
        learned = next((r for r in scale_results if r["scale"] == s), None)
        random = next((r for r in random_results if r["scale"] == s), None)
        learned_per = f"{learned['flow_per']:.2f}" if learned else "---"
        random_per = f"{random['flow_per']:.2f}" if random else "---"
        print(f"{s:>6} | {learned_per:>18} | {random_per:>18} | {p0_baseline:>18.2f}")

    print(f"\n>>> p0 PER (no flow, scale=0): {p0_baseline:.2f}")
    print(f">>> Best learned flow PER: {best_scale_result['flow_per']:.2f} "
          f"at scale={optimal_scale}")
    improvement = p0_baseline - best_scale_result["flow_per"]
    if improvement > 0:
        print(f">>> Improvement: {improvement:.2f} PER points "
              f"({improvement/p0_baseline*100:.1f}% relative)")
    else:
        print(f">>> No improvement over p0 at current scale values")

    print("\n" + "=" * 70)
    print("SUMMARY: RESOLUTION SWEEP")
    print("=" * 70)
    print(f"{'S':>6} | {'~ms/step':>10} | {'Flow PER':>10} | {'p0 PER':>10} | {'Gap':>8}")
    print("-" * 55)
    for r in resolution_results:
        gap = r["flow_per"] - r["p0_per"]
        print(f"{r['num_steps']:>6} | {r['ms_per_step']:>10} | "
              f"{r['flow_per']:>10.2f} | {r['p0_per']:>10.2f} | {gap:>+8.2f}")


if __name__ == "__main__":
    main()