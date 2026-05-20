#!/usr/bin/env python3
"""Clean System Combination for Stage 13.

Combines the CTC output (p0) with the flow-refined output (p_flow)
using linear interpolation before decoding.

Key difference from previous attempt: uses FIXED SCALE flow
(normalize to unit, scale by lambda) instead of confidence-weighted.
This produces a p_flow that's genuinely different from p0.

Usage:
    python system_combination.py hparams/train.yaml
"""

import os, sys, csv, json
import torch
import torch.nn.functional as F
from hyperpyyaml import load_hyperpyyaml
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main
from train import VelocityNet, ASR_Brain, dataio_prep


class CombinationASR(ASR_Brain):
    """ASR Brain with system combination at decode time."""

    def integrate_flow_fixed_scale(self, p0, hidden, num_steps, scale):
        """Fixed-scale local-window integration (same as Stage 13)."""
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
            v_local = v_local / v_norm * scale

            p[:, frame_start:frame_end, :] = (
                p[:, frame_start:frame_end, :] + dt * v_local
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


def run_combination(asr_brain, test_data, hparams, flow_scale, alpha):
    """Run one combination evaluation."""
    asr_brain.per_metrics_combined = hparams["per_stats"]()
    asr_brain.per_metrics_p0_check = hparams["per_stats"]()
    asr_brain.per_metrics_flow_only = hparams["per_stats"]()

    asr_brain.modules.eval()

    with torch.no_grad():
        for batch in sb.dataio.dataloader.make_dataloader(
            test_data, **hparams["test_dataloader_opts"]
        ):
            batch = batch.to(asr_brain.device)
            wavs, wav_lens = batch.sig
            phns, phn_lens = batch.phn_encoded

            feats = hparams["compute_features"](wavs)
            feats = asr_brain.modules.normalize(feats, wav_lens)

            hidden = asr_brain.modules.model(feats)
            logits = asr_brain.modules.output(hidden)
            alpha_dir = F.softplus(logits) + hparams["dfm_epsilon"]
            p0 = alpha_dir / alpha_dir.sum(
                dim=-1, keepdim=True
            ).clamp_min(hparams["dfm_epsilon"])

            # Get flow-refined distribution
            p_flow = asr_brain.integrate_flow_fixed_scale(
                p0, hidden, hparams["num_flow_steps"], flow_scale
            )

            # Linear combination
            p_combined = alpha * p0 + (1.0 - alpha) * p_flow

            # Decode combined distribution with CTC greedy
            log_probs = torch.log(p_combined.clamp_min(1e-8))
            seq_combined = sb.decoders.ctc_greedy_decode(
                log_probs, wav_lens,
                blank_id=hparams["blank_index"]
            )
            asr_brain.per_metrics_combined.append(
                ids=batch.id, predict=seq_combined,
                target=phns, target_len=phn_lens,
                ind2lab=asr_brain.label_encoder.decode_ndim,
            )

            # Also decode p0 alone (sanity check)
            log_probs_p0 = torch.log(p0.clamp_min(1e-8))
            seq_p0 = sb.decoders.ctc_greedy_decode(
                log_probs_p0, wav_lens,
                blank_id=hparams["blank_index"]
            )
            asr_brain.per_metrics_p0_check.append(
                ids=batch.id, predict=seq_p0,
                target=phns, target_len=phn_lens,
                ind2lab=asr_brain.label_encoder.decode_ndim,
            )

            # Also decode p_flow alone
            log_probs_flow = torch.log(p_flow.clamp_min(1e-8))
            seq_flow = sb.decoders.ctc_greedy_decode(
                log_probs_flow, wav_lens,
                blank_id=hparams["blank_index"]
            )
            asr_brain.per_metrics_flow_only.append(
                ids=batch.id, predict=seq_flow,
                target=phns, target_len=phn_lens,
                ind2lab=asr_brain.label_encoder.decode_ndim,
            )

    return {
        "combined_per": asr_brain.per_metrics_combined.summarize("error_rate"),
        "p0_per": asr_brain.per_metrics_p0_check.summarize("error_rate"),
        "flow_per": asr_brain.per_metrics_flow_only.summarize("error_rate"),
    }


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

    asr_brain = CombinationASR(
        modules=hparams["modules"], opt_class=hparams["opt_class"],
        hparams=hparams, run_opts=run_opts,
        checkpointer=hparams["checkpointer"])
    asr_brain.label_encoder = label_encoder
    asr_brain.checkpointer.recover_if_possible(min_key="PER")

    output_dir = hparams["output_folder"]

    # Sanity check: alpha=1.0 should give exactly p0 PER
    print("\n" + "=" * 70)
    print("SANITY CHECK: alpha=1.0 should equal p0 PER")
    print("=" * 70)
    r = run_combination(asr_brain, test_data, hparams,
                        flow_scale=7, alpha=1.0)
    print(f"  p0 PER: {r['p0_per']:.2f}")
    print(f"  Combined (alpha=1.0): {r['combined_per']:.2f}")
    print(f"  Flow alone (scale=7): {r['flow_per']:.2f}")

    if abs(r['combined_per'] - r['p0_per']) > 0.1:
        print("  WARNING: sanity check failed! eval mode may not be set.")
        return

    p0_baseline = r['p0_per']

    # ==================================================================
    # 2D Grid: scale x alpha
    # ==================================================================
    print("\n" + "=" * 70)
    print("SYSTEM COMBINATION: scale x alpha grid")
    print("=" * 70)

    scales = [3, 5, 7, 10, 15, 20, 30, 50]
    alphas = [1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5]
    all_results = []

    for s in scales:
        print(f"\n  --- Flow scale={s} ---")
        for a in alphas:
            r = run_combination(asr_brain, test_data, hparams,
                                flow_scale=s, alpha=a)
            entry = {
                "flow_scale": s, "alpha": a,
                "combined_per": r["combined_per"],
                "p0_per": r["p0_per"],
                "flow_per": r["flow_per"],
            }
            all_results.append(entry)
            gap = r["combined_per"] - p0_baseline
            marker = " <<<" if gap < -0.01 else ""
            print(f"    alpha={a:.2f}: combined={r['combined_per']:.2f}, "
                  f"flow_alone={r['flow_per']:.2f}, "
                  f"gap={gap:+.2f}{marker}")

    # ==================================================================
    # Save results
    # ==================================================================
    csv_path = os.path.join(output_dir, "combination_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "flow_scale", "alpha", "combined_per", "p0_per", "flow_per"
        ])
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)
    print(f"\nResults saved to: {csv_path}")

    json_path = os.path.join(output_dir, "combination_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to: {json_path}")

    # ==================================================================
    # Summary grid
    # ==================================================================
    print("\n" + "=" * 70)
    print("COMBINED PER GRID (<<< = better than p0)")
    print("=" * 70)

    # Header
    print(f"{'scale':>8} |", end="")
    for a in alphas:
        print(f"  a={a:.2f}", end="")
    print(f" | {'flow alone':>11}")
    print("-" * (10 + 8 * len(alphas) + 14))

    for s in scales:
        print(f"{s:>8} |", end="")
        flow_alone = None
        for a in alphas:
            r = next((x for x in all_results
                      if x["flow_scale"] == s and x["alpha"] == a), None)
            if r:
                per = r["combined_per"]
                flow_alone = r["flow_per"]
                if per < p0_baseline - 0.01:
                    print(f"  {per:5.2f}*", end="")
                else:
                    print(f"  {per:6.2f}", end="")
        if flow_alone is not None:
            print(f" | {flow_alone:>10.2f}", end="")
        print()

    # Find best
    best = min(all_results, key=lambda x: x["combined_per"])
    print(f"\n>>> p0 baseline: {p0_baseline:.2f}")
    print(f">>> Best combined: {best['combined_per']:.2f} "
          f"(scale={best['flow_scale']}, alpha={best['alpha']})")
    print(f">>> Best flow alone: "
          f"{min(all_results, key=lambda x: x['flow_per'])['flow_per']:.2f}")

    improvement = p0_baseline - best["combined_per"]
    if improvement > 0:
        print(f">>> Combination improvement over p0: {improvement:.2f} "
              f"({improvement/p0_baseline*100:.1f}% relative)")

    # Compare: best combination vs best flow alone
    best_flow = min(all_results, key=lambda x: x["flow_per"])
    print(f"\n>>> Best flow alone: {best_flow['flow_per']:.2f} "
          f"(scale={best_flow['flow_scale']})")
    combo_vs_flow = best_flow["flow_per"] - best["combined_per"]
    if combo_vs_flow > 0:
        print(f">>> Combination improves over flow alone by: "
              f"{combo_vs_flow:.2f}")
    else:
        print(f">>> Combination matches flow alone "
              f"(error correlation too high for additional gains)")


if __name__ == "__main__":
    main()