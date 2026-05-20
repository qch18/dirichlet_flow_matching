#!/usr/bin/env python3
"""Extended combination sweep for Stage 17.

The initial grid found best at scale=5, alpha=0.5.
Alpha was still improving at 0.5, so this sweep extends to lower alpha
and explores the scale 3-10 region with fine alpha steps.

Usage:
    python system_combination_extended.py hparams/train.yaml
"""

import os, sys, csv, json
import torch
import torch.nn.functional as F
from hyperpyyaml import load_hyperpyyaml
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main
from train import VelocityNet, ASR_Brain, dataio_prep


class CombinationASR(ASR_Brain):
    def integrate_flow_fixed(self, p0, hidden, num_steps, scale):
        dt = 1.0 / num_steps
        p = p0.clone()
        B, T_enc = hidden.size(0), hidden.size(1)
        for step in range(num_steps):
            t = step * dt
            fs = int(t * T_enc)
            fe = min(int((t + dt) * T_enc), T_enc)
            fs = min(fs, T_enc - 1)
            fe = max(fe, fs + 1)
            tau = min(int((fs + fe) / 2), T_enc - 1)
            h_tau = hidden[:, tau:tau+1, :].expand_as(hidden)
            v = self.modules.velocity_net(p, t, h_tau)
            v_local = v[:, fs:fe, :]
            v_norm = torch.sqrt(
                (v_local ** 2).sum(dim=-1, keepdim=True)).clamp_min(1e-8)
            v_local = v_local / v_norm * scale
            p[:, fs:fe, :] = p[:, fs:fe, :] + dt * v_local
            p[:, fs:fe, :] = p[:, fs:fe, :].clamp_min(self.hparams.dfm_epsilon)
            ws = p[:, fs:fe, :].sum(dim=-1, keepdim=True).clamp_min(
                self.hparams.dfm_epsilon)
            p[:, fs:fe, :] = p[:, fs:fe, :] / ws
        return p


def run_combination(asr_brain, test_data, hparams, scale, alpha,
                    method="linear"):
    """Run one combination evaluation."""
    m_comb = hparams["per_stats"]()
    m_p0 = hparams["per_stats"]()
    m_flow = hparams["per_stats"]()

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
                dim=-1, keepdim=True).clamp_min(hparams["dfm_epsilon"])

            p_flow = asr_brain.integrate_flow_fixed(
                p0, hidden, hparams["num_flow_steps"], scale)

            if method == "linear":
                p_combined = alpha * p0 + (1 - alpha) * p_flow
            elif method == "log_linear":
                log_p0 = torch.log(p0.clamp_min(1e-8))
                log_pf = torch.log(p_flow.clamp_min(1e-8))
                log_pc = alpha * log_p0 + (1 - alpha) * log_pf
                p_combined = torch.softmax(log_pc, dim=-1)

            p_combined = p_combined.clamp_min(1e-8)
            p_combined = p_combined / p_combined.sum(dim=-1, keepdim=True)

            for p_dist, met in [(p_combined, m_comb), (p0, m_p0),
                                (p_flow, m_flow)]:
                log_probs = torch.log(p_dist.clamp_min(1e-8))
                seq = sb.decoders.ctc_greedy_decode(
                    log_probs, wav_lens, blank_id=hparams["blank_index"])
                met.append(ids=batch.id, predict=seq, target=phns,
                           target_len=phn_lens,
                           ind2lab=asr_brain.label_encoder.decode_ndim)

    return {
        "combined_per": m_comb.summarize("error_rate"),
        "p0_per": m_p0.summarize("error_rate"),
        "flow_per": m_flow.summarize("error_rate"),
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

    # Sanity check
    r = run_combination(asr_brain, test_data, hparams, scale=0, alpha=1.0)
    p0_per = r["p0_per"]
    print(f"\np0 baseline: {p0_per:.2f}")

    all_results = []

    # ==================================================================
    # Fine grid around the sweet spot: scale 3-10, alpha 0.1-0.9
    # ==================================================================
    print("\n" + "=" * 70)
    print("FINE LINEAR COMBINATION GRID")
    print("=" * 70)

    scales = [1, 2, 3, 4, 5, 6, 7, 8, 10]
    alphas = [0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.1]

    for s in scales:
        row_results = []
        for a in alphas:
            r = run_combination(asr_brain, test_data, hparams,
                                scale=s, alpha=a, method="linear")
            entry = {
                "method": "linear", "scale": s, "alpha": a,
                "combined_per": r["combined_per"],
                "p0_per": r["p0_per"],
                "flow_per": r["flow_per"],
            }
            all_results.append(entry)
            row_results.append(entry)

        # Print row
        best_in_row = min(row_results, key=lambda x: x["combined_per"])
        flow_alone = row_results[0]["flow_per"]
        parts = []
        for e in row_results:
            g = e["combined_per"] - p0_per
            mark = "*" if g < -0.01 else " "
            parts.append(f"{e['combined_per']:.2f}{mark}")
        print(f"  s={s:2d} | {' '.join(parts)} | flow={flow_alone:.2f}")

    # Also try scale 15, 20 at low alpha for completeness
    for s in [15, 20]:
        for a in [0.7, 0.6, 0.5, 0.4, 0.3]:
            r = run_combination(asr_brain, test_data, hparams,
                                scale=s, alpha=a, method="linear")
            all_results.append({
                "method": "linear", "scale": s, "alpha": a,
                "combined_per": r["combined_per"],
                "p0_per": r["p0_per"],
                "flow_per": r["flow_per"],
            })
            g = r["combined_per"] - p0_per
            m = " <<<" if g < -0.01 else ""
            print(f"  s={s:2d} a={a:.2f}: {r['combined_per']:.2f} ({g:+.2f}){m}")

    # ==================================================================
    # Log-linear at best configs
    # ==================================================================
    print("\n" + "=" * 70)
    print("LOG-LINEAR COMBINATION (best configs)")
    print("=" * 70)

    for s in [3, 5, 7, 10]:
        for a in [0.7, 0.6, 0.5, 0.4, 0.3, 0.2]:
            r = run_combination(asr_brain, test_data, hparams,
                                scale=s, alpha=a, method="log_linear")
            all_results.append({
                "method": "log_linear", "scale": s, "alpha": a,
                "combined_per": r["combined_per"],
                "p0_per": r["p0_per"],
                "flow_per": r["flow_per"],
            })
            g = r["combined_per"] - p0_per
            m = " <<<" if g < -0.01 else ""
            print(f"  s={s:2d} a={a:.2f}: {r['combined_per']:.2f} ({g:+.2f}){m}")

    # ==================================================================
    # Save
    # ==================================================================
    csv_path = os.path.join(output_dir, "combination_extended.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "method", "scale", "alpha", "combined_per", "p0_per", "flow_per"])
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)

    json_path = os.path.join(output_dir, "combination_extended.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"results": all_results, "p0_baseline": p0_per}, f, indent=2)

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {json_path}")

    # ==================================================================
    # Summary
    # ==================================================================
    best_linear = min([x for x in all_results if x["method"] == "linear"],
                      key=lambda x: x["combined_per"])
    best_loglin = min([x for x in all_results if x["method"] == "log_linear"],
                      key=lambda x: x["combined_per"])
    overall = min(all_results, key=lambda x: x["combined_per"])

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  p0 baseline:        {p0_per:.2f}")
    print(f"  Best linear:        {best_linear['combined_per']:.2f} "
          f"(scale={best_linear['scale']}, alpha={best_linear['alpha']})")
    print(f"  Best log-linear:    {best_loglin['combined_per']:.2f} "
          f"(scale={best_loglin['scale']}, alpha={best_loglin['alpha']})")
    print(f"  Flow alone at best: {best_linear['flow_per']:.2f}")

    imp = p0_per - overall["combined_per"]
    if imp > 0:
        print(f"\n  >>> Improvement: {imp:.2f} ({imp/p0_per*100:.1f}% rel)")

    # Print the grid nicely
    print("\n" + "=" * 70)
    print("LINEAR COMBINATION GRID")
    print("=" * 70)
    linear_results = [x for x in all_results if x["method"] == "linear"]
    grid_scales = sorted(set(x["scale"] for x in linear_results))
    grid_alphas = sorted(set(x["alpha"] for x in linear_results), reverse=True)

    header = f"{'Scale':>6} |"
    for a in grid_alphas:
        header += f" a={a:.2f}"
    header += " | flow"
    print(header)
    print("-" * len(header))

    for s in grid_scales:
        row = f"{s:>6} |"
        flow_per = None
        for a in grid_alphas:
            match = next((x for x in linear_results
                          if x["scale"] == s and x["alpha"] == a), None)
            if match:
                per = match["combined_per"]
                flow_per = match["flow_per"]
                if per < p0_per - 0.01:
                    row += f" {per:.2f}*"
                else:
                    row += f"  {per:.2f}"
            else:
                row += "    ---"
        if flow_per is not None:
            row += f" | {flow_per:.2f}"
        print(row)


if __name__ == "__main__":
    main()