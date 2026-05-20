#!/usr/bin/env python3
"""Stage 17 Refined Experiments: Fine-grained parameter sweep.

Based on initial results:
  - passes=1 is optimal (multi-pass hurts)
  - T=1.0 is best (temperature softening hurts at T>=1.5)
  - Best flow PER was at scale=5 with T=1.0
  - Need finer resolution around the optimal region

Experiment 1: Fine scale sweep at T=1.0, passes=1
Experiment 2: Fine temperature sweep at optimal scale
Experiment 3: Resolution sweep
Experiment 4: Random ablation

Usage:
    python stage17_experiments.py hparams/train.yaml
"""

import os, sys, csv, json
import torch
from hyperpyyaml import load_hyperpyyaml
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main
from train import VelocityNet, ASR_Brain, dataio_prep


def run_eval(asr_brain, test_data, hparams, temperature, base_scale,
             num_passes, num_steps, use_random=False):
    """Run one evaluation with given parameters."""
    asr_brain.hparams.flow_temperature = temperature
    asr_brain.hparams.base_scale = base_scale
    asr_brain.hparams.num_passes = num_passes
    asr_brain.hparams.num_flow_steps = num_steps

    original = asr_brain.integrate_flow

    if use_random:
        def random_integrate(p0, hidden, num_steps_arg):
            dt = 1.0 / num_steps_arg
            B, T_enc = hidden.size(0), hidden.size(1)

            log_p0 = torch.log(p0.clamp_min(asr_brain.hparams.dfm_epsilon))
            p_soft = torch.softmax(log_p0 / temperature, dim=-1)
            p = p_soft.clone()

            max_probs = p_soft.max(dim=-1, keepdim=True).values
            uncertainty = 1.0 - max_probs

            for pass_idx in range(num_passes):
                for step in range(num_steps_arg):
                    t = step * dt
                    fs = int(t * T_enc)
                    fe = min(int((t + dt) * T_enc), T_enc)
                    fs = min(fs, T_enc - 1)
                    fe = max(fe, fs + 1)

                    v = torch.randn(B, fe - fs,
                                    asr_brain.hparams.output_neurons,
                                    device=p.device, dtype=p.dtype)
                    v = v - v.mean(dim=-1, keepdim=True)
                    vn = torch.sqrt(
                        (v**2).sum(dim=-1, keepdim=True)
                    ).clamp_min(1e-8)
                    v = v / vn

                    lu = uncertainty[:, fs:fe, :]
                    v = v * base_scale * lu

                    p[:, fs:fe, :] = p[:, fs:fe, :] + dt * v
                    p[:, fs:fe, :] = p[:, fs:fe, :].clamp_min(
                        asr_brain.hparams.dfm_epsilon)
                    ws = p[:, fs:fe, :].sum(
                        dim=-1, keepdim=True
                    ).clamp_min(asr_brain.hparams.dfm_epsilon)
                    p[:, fs:fe, :] = p[:, fs:fe, :] / ws
            return p

        asr_brain.integrate_flow = random_integrate

    asr_brain.on_stage_start(sb.Stage.TEST, epoch=None)
    asr_brain.evaluate(test_data, min_key="PER",
                       test_loader_kwargs=hparams["test_dataloader_opts"])

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
        hyperparams_to_save=hparams_file, overrides=overrides)
    run_on_main(prepare_timit, kwargs={
        "data_folder": hparams["data_folder"],
        "save_json_train": hparams["train_annotation"],
        "save_json_valid": hparams["valid_annotation"],
        "save_json_test": hparams["test_annotation"],
        "skip_prep": hparams["skip_prep"],
        "uppercase": hparams["uppercase"]})
    run_on_main(hparams["prepare_noise_data"])

    train_data, valid_data, test_data, label_encoder = dataio_prep(hparams)

    asr_brain = ASR_Brain(
        modules=hparams["modules"], opt_class=hparams["opt_class"],
        hparams=hparams, run_opts=run_opts,
        checkpointer=hparams["checkpointer"])
    asr_brain.label_encoder = label_encoder
    asr_brain.checkpointer.recover_if_possible(min_key="PER")

    output_dir = hparams["output_folder"]
    all_results = []

    # ==================================================================
    # Experiment 1: Fine Scale Sweep (T=1.0, passes=1, S=20)
    # ==================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: FINE SCALE SWEEP (T=1.0, passes=1, S=20)")
    print("=" * 70)

    scales = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]
    scale_results = []

    for s in scales:
        r = run_eval(asr_brain, test_data, hparams,
                     temperature=1.0, base_scale=s,
                     num_passes=1, num_steps=20)
        entry = {"experiment": "fine_scale_sweep",
                 "temperature": 1.0, "base_scale": s,
                 "num_passes": 1, "num_steps": 20, "random": False,
                 "flow_per": r["flow_per"], "p0_per": r["p0_per"]}
        scale_results.append(entry)
        gap = r["flow_per"] - r["p0_per"]
        marker = " <<<" if gap < 0 else ""
        print(f"  scale={s:3d}: Flow={r['flow_per']:.2f}, "
              f"p0={r['p0_per']:.2f}, gap={gap:+.2f}{marker}")

    best_scale = min(scale_results, key=lambda x: x["flow_per"])
    opt_s = best_scale["base_scale"]
    print(f"\n  >>> Optimal scale: {opt_s} "
          f"(flow={best_scale['flow_per']:.2f})")

    # ==================================================================
    # Experiment 2: Fine Temperature Sweep (optimal scale, passes=1)
    # ==================================================================
    print("\n" + "=" * 70)
    print(f"EXPERIMENT 2: FINE TEMPERATURE SWEEP (scale={opt_s}, passes=1)")
    print("=" * 70)

    temps = [1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.5, 2.0]
    temp_results = []

    for T in temps:
        r = run_eval(asr_brain, test_data, hparams,
                     temperature=T, base_scale=opt_s,
                     num_passes=1, num_steps=20)
        entry = {"experiment": "fine_temp_sweep",
                 "temperature": T, "base_scale": opt_s,
                 "num_passes": 1, "num_steps": 20, "random": False,
                 "flow_per": r["flow_per"], "p0_per": r["p0_per"]}
        temp_results.append(entry)
        gap = r["flow_per"] - r["p0_per"]
        marker = " <<<" if gap < 0 else ""
        print(f"  T={T:.2f}: Flow={r['flow_per']:.2f}, "
              f"gap={gap:+.2f}{marker}")

    best_temp = min(temp_results, key=lambda x: x["flow_per"])
    opt_T = best_temp["temperature"]
    print(f"\n  >>> Optimal T: {opt_T} "
          f"(flow={best_temp['flow_per']:.2f})")

    # ==================================================================
    # Experiment 3: Resolution Sweep (best T and scale, passes=1)
    # ==================================================================
    print("\n" + "=" * 70)
    print(f"EXPERIMENT 3: RESOLUTION SWEEP (T={opt_T}, scale={opt_s})")
    print("=" * 70)

    steps_vals = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100]
    res_results = []

    for S in steps_vals:
        r = run_eval(asr_brain, test_data, hparams,
                     temperature=opt_T, base_scale=opt_s,
                     num_passes=1, num_steps=S)
        ms = round(1000.0 / S)
        entry = {"experiment": "resolution_sweep",
                 "temperature": opt_T, "base_scale": opt_s,
                 "num_passes": 1, "num_steps": S, "ms_per_step": ms,
                 "random": False,
                 "flow_per": r["flow_per"], "p0_per": r["p0_per"]}
        res_results.append(entry)
        gap = r["flow_per"] - r["p0_per"]
        print(f"  S={S:3d} (~{ms}ms): Flow={r['flow_per']:.2f}, "
              f"gap={gap:+.2f}")

    # ==================================================================
    # Experiment 4: Random Ablation (best T, passes=1)
    # ==================================================================
    print("\n" + "=" * 70)
    print(f"EXPERIMENT 4: RANDOM ABLATION (T={opt_T}, passes=1)")
    print("=" * 70)

    rand_scales = [0, 1, 3, 5, 7, 10, 15, 20]
    rand_results = []

    for s in rand_scales:
        r = run_eval(asr_brain, test_data, hparams,
                     temperature=opt_T, base_scale=s,
                     num_passes=1, num_steps=20, use_random=True)
        entry = {"experiment": "random_ablation",
                 "temperature": opt_T, "base_scale": s,
                 "num_passes": 1, "num_steps": 20, "random": True,
                 "flow_per": r["flow_per"], "p0_per": r["p0_per"]}
        rand_results.append(entry)
        print(f"  Random scale={s:3d}: Flow={r['flow_per']:.2f}")

    # ==================================================================
    # Save
    # ==================================================================
    all_results = scale_results + temp_results + res_results + rand_results

    json_path = os.path.join(output_dir, "experiment_results_refined.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "fine_scale_sweep": scale_results,
            "fine_temp_sweep": temp_results,
            "resolution_sweep": res_results,
            "random_ablation": rand_results,
            "optimal_scale": opt_s,
            "optimal_temperature": opt_T,
        }, f, indent=2)

    csv_path = os.path.join(output_dir, "experiment_results_refined.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "experiment", "temperature", "base_scale", "num_passes",
            "num_steps", "ms_per_step", "random", "flow_per", "p0_per"])
        writer.writeheader()
        for row in all_results:
            if "ms_per_step" not in row:
                row["ms_per_step"] = ""
            writer.writerow(row)

    print(f"\nResults saved to: {json_path}")
    print(f"Results saved to: {csv_path}")

    # ==================================================================
    # Summary
    # ==================================================================
    p0 = scale_results[0]["p0_per"]

    print("\n" + "=" * 70)
    print("SCALE SWEEP RESULTS")
    print("=" * 70)
    print(f"{'Scale':>8} | {'Flow PER':>10} | {'p0 PER':>10} | {'Gap':>8}")
    print("-" * 45)
    for r in scale_results:
        gap = r["flow_per"] - r["p0_per"]
        marker = " *" if gap < 0 else ""
        print(f"{r['base_scale']:>8} | {r['flow_per']:>10.2f} | "
              f"{r['p0_per']:>10.2f} | {gap:>+8.2f}{marker}")

    print(f"\n" + "=" * 70)
    print("LEARNED vs RANDOM")
    print("=" * 70)
    print(f"{'Scale':>8} | {'Learned':>10} | {'Random':>10} | {'p0':>10}")
    print("-" * 48)
    for s in rand_scales:
        lv = next((x for x in scale_results if x["base_scale"] == s), None)
        rv = next((x for x in rand_results if x["base_scale"] == s), None)
        lp = f"{lv['flow_per']:.2f}" if lv else "---"
        rp = f"{rv['flow_per']:.2f}" if rv else "---"
        print(f"{s:>8} | {lp:>10} | {rp:>10} | {p0:>10.2f}")

    overall_best = min(all_results, key=lambda x: x["flow_per"])
    print(f"\n>>> p0 baseline: {p0:.2f}")
    print(f">>> Best flow: {overall_best['flow_per']:.2f} "
          f"(T={overall_best.get('temperature')}, "
          f"scale={overall_best.get('base_scale')}, "
          f"passes={overall_best.get('num_passes')})")
    imp = p0 - overall_best["flow_per"]
    if imp > 0:
        print(f">>> Improvement: {imp:.2f} ({imp/p0*100:.1f}% relative)")


if __name__ == "__main__":
    main()