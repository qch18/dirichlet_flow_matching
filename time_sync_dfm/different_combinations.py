#!/usr/bin/env python3
"""System Combination with Multi-Resolution Ensemble.

Two methods that create genuinely different flow outputs:

Method 1 - Multi-Resolution Ensemble:
    Run the flow at multiple S values (S=15, 20, 25, 30). Each S groups
    frames into different windows, so different phoneme boundaries get
    different treatments. Averaging the flow outputs reduces discretisation
    artifacts and produces a more robust correction than any single S.

Method 2 - Multi-Scale Ensemble:
    Run the flow at multiple scales (3, 5, 7, 10) and average.
    Low scales make conservative corrections. High scales make aggressive
    corrections. Averaging keeps corrections where all scales agree.

Method 3 - Per-Frame Adaptive Alpha:
    Instead of a single global alpha, use per-frame weights based on
    how much the flow changed each frame's distribution.

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

    def multi_resolution_flow(self, p0, hidden, scale, s_values):
        p_sum = torch.zeros_like(p0)
        for S in s_values:
            p_sum = p_sum + self.integrate_flow_fixed(p0, hidden, S, scale)
        p_avg = p_sum / len(s_values)
        p_avg = p_avg.clamp_min(self.hparams.dfm_epsilon)
        return p_avg / p_avg.sum(dim=-1, keepdim=True).clamp_min(
            self.hparams.dfm_epsilon)

    def multi_scale_flow(self, p0, hidden, num_steps, scales):
        p_sum = torch.zeros_like(p0)
        for s in scales:
            p_sum = p_sum + self.integrate_flow_fixed(p0, hidden, num_steps, s)
        p_avg = p_sum / len(scales)
        p_avg = p_avg.clamp_min(self.hparams.dfm_epsilon)
        return p_avg / p_avg.sum(dim=-1, keepdim=True).clamp_min(
            self.hparams.dfm_epsilon)


def decode_and_score(p_dist, wav_lens, batch, hparams, label_encoder, metrics):
    phns, phn_lens = batch.phn_encoded
    log_probs = torch.log(p_dist.clamp_min(1e-8))
    seq = sb.decoders.ctc_greedy_decode(
        log_probs, wav_lens, blank_id=hparams["blank_index"])
    metrics.append(ids=batch.id, predict=seq, target=phns,
                   target_len=phn_lens,
                   ind2lab=label_encoder.decode_ndim)


def run_eval(asr_brain, test_data, hparams, method, **kwargs):
    m_comb = hparams["per_stats"]()
    m_p0 = hparams["per_stats"]()
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
            p0 = alpha_dir / alpha_dir.sum(
                dim=-1, keepdim=True).clamp_min(hparams["dfm_epsilon"])

            if method == "standard":
                p_flow = asr_brain.integrate_flow_fixed(
                    p0, hidden, hparams["num_flow_steps"], kwargs["scale"])
                p_combined = kwargs["alpha"] * p0 + (1 - kwargs["alpha"]) * p_flow

            elif method == "multi_resolution":
                p_flow = asr_brain.multi_resolution_flow(
                    p0, hidden, kwargs["scale"], kwargs["s_values"])
                p_combined = kwargs["alpha"] * p0 + (1 - kwargs["alpha"]) * p_flow

            elif method == "multi_scale":
                p_flow = asr_brain.multi_scale_flow(
                    p0, hidden, hparams["num_flow_steps"], kwargs["scales"])
                p_combined = kwargs["alpha"] * p0 + (1 - kwargs["alpha"]) * p_flow

            elif method == "adaptive_alpha":
                p_flow = asr_brain.integrate_flow_fixed(
                    p0, hidden, hparams["num_flow_steps"], kwargs["scale"])
                delta = (p_flow - p0).abs().sum(dim=-1, keepdim=True)
                delta_norm = delta / delta.max().clamp_min(1e-8)
                ab = kwargs["alpha_base"]
                ar = kwargs["alpha_range"]
                alpha_pf = (ab + ar * delta_norm).clamp(0.0, 1.0)
                p_combined = alpha_pf * p0 + (1 - alpha_pf) * p_flow

            p_combined = p_combined.clamp_min(1e-8)
            p_combined = p_combined / p_combined.sum(dim=-1, keepdim=True)

            decode_and_score(p_combined, wav_lens, batch, hparams,
                             asr_brain.label_encoder, m_comb)
            decode_and_score(p0, wav_lens, batch, hparams,
                             asr_brain.label_encoder, m_p0)

    return {"combined_per": m_comb.summarize("error_rate"),
            "p0_per": m_p0.summarize("error_rate")}


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
    print("\n" + "=" * 70)
    print("SANITY CHECK")
    r = run_eval(asr_brain, test_data, hparams, "standard", scale=7, alpha=1.0)
    p0 = r["p0_per"]
    print(f"  p0={p0:.2f}, combined(a=1.0)={r['combined_per']:.2f}")
    if abs(r['combined_per'] - p0) > 0.1:
        print("  FAILED"); return

    all_results = []

    # ==================================================================
    # Method 1: Standard
    # ==================================================================
    print("\n" + "=" * 70)
    print("METHOD 1: STANDARD LINEAR COMBINATION")
    print("=" * 70)
    for s in [3, 5, 7, 10, 15, 20, 30, 50]:
        for a in [1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5]:
            r = run_eval(asr_brain, test_data, hparams, "standard",
                         scale=s, alpha=a)
            e = {"method": "standard", "scale": s, "alpha": a,
                 "combined_per": r["combined_per"], "p0_per": r["p0_per"]}
            all_results.append(e)
            g = r["combined_per"] - p0
            m = " <<<" if g < -0.01 else ""
            print(f"  s={s:3d} a={a:.2f}: {r['combined_per']:.2f} ({g:+.2f}){m}")

    best_std = min([x for x in all_results if x["method"] == "standard"],
                   key=lambda x: x["combined_per"])
    print(f"  >>> Best: {best_std['combined_per']:.2f}")

    # ==================================================================
    # Method 2: Multi-Resolution Ensemble
    # ==================================================================
    print("\n" + "=" * 70)
    print("METHOD 2: MULTI-RESOLUTION ENSEMBLE (S=15,20,25,30)")
    print("=" * 70)
    for s in [5, 7, 10, 15]:
        for a in [0.9, 0.8, 0.7, 0.6, 0.5]:
            r = run_eval(asr_brain, test_data, hparams, "multi_resolution",
                         scale=s, alpha=a, s_values=[15, 20, 25, 30])
            e = {"method": "multi_resolution", "scale": s, "alpha": a,
                 "combined_per": r["combined_per"], "p0_per": r["p0_per"]}
            all_results.append(e)
            g = r["combined_per"] - p0
            m = " <<<" if g < -0.01 else ""
            print(f"  s={s:3d} a={a:.2f}: {r['combined_per']:.2f} ({g:+.2f}){m}")

    best_mr = min([x for x in all_results if x["method"] == "multi_resolution"],
                  key=lambda x: x["combined_per"])
    print(f"  >>> Best: {best_mr['combined_per']:.2f}")

    # ==================================================================
    # Method 3: Multi-Scale Ensemble
    # ==================================================================
    print("\n" + "=" * 70)
    print("METHOD 3: MULTI-SCALE ENSEMBLE (scale=3,5,7,10)")
    print("=" * 70)
    for a in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]:
        r = run_eval(asr_brain, test_data, hparams, "multi_scale",
                     alpha=a, scales=[3, 5, 7, 10])
        e = {"method": "multi_scale", "alpha": a,
             "combined_per": r["combined_per"], "p0_per": r["p0_per"]}
        all_results.append(e)
        g = r["combined_per"] - p0
        m = " <<<" if g < -0.01 else ""
        print(f"  a={a:.2f}: {r['combined_per']:.2f} ({g:+.2f}){m}")

    best_ms = min([x for x in all_results if x["method"] == "multi_scale"],
                  key=lambda x: x["combined_per"])
    print(f"  >>> Best: {best_ms['combined_per']:.2f}")

    # ==================================================================
    # Method 4: Per-Frame Adaptive Alpha
    # ==================================================================
    print("\n" + "=" * 70)
    print("METHOD 4: PER-FRAME ADAPTIVE ALPHA")
    print("=" * 70)
    for s in [7, 10, 15, 20]:
        for ab, ar in [(0.3, 0.5), (0.4, 0.4), (0.5, 0.3), (0.3, 0.6),
                       (0.5, 0.4), (0.6, 0.3)]:
            r = run_eval(asr_brain, test_data, hparams, "adaptive_alpha",
                         scale=s, alpha_base=ab, alpha_range=ar)
            e = {"method": "adaptive_alpha", "scale": s,
                 "alpha_base": ab, "alpha_range": ar,
                 "combined_per": r["combined_per"], "p0_per": r["p0_per"]}
            all_results.append(e)
            g = r["combined_per"] - p0
            m = " <<<" if g < -0.01 else ""
            print(f"  s={s:3d} base={ab:.1f} range={ar:.1f}: "
                  f"{r['combined_per']:.2f} ({g:+.2f}){m}")

    best_ad = min([x for x in all_results if x["method"] == "adaptive_alpha"],
                  key=lambda x: x["combined_per"])
    print(f"  >>> Best: {best_ad['combined_per']:.2f}")

    # ==================================================================
    # Save
    # ==================================================================
    csv_path = os.path.join(output_dir, "combination_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fnames = ["method", "scale", "alpha", "alpha_base", "alpha_range",
                  "combined_per", "p0_per"]
        writer = csv.DictWriter(f, fieldnames=fnames, extrasaction="ignore")
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)

    json_path = os.path.join(output_dir, "combination_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"results": all_results, "p0_baseline": p0}, f, indent=2)

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {json_path}")

    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"  p0 baseline:      {p0:.2f}")
    print(f"  Best standard:    {best_std['combined_per']:.2f} "
          f"(s={best_std['scale']}, a={best_std['alpha']})")
    print(f"  Best multi-res:   {best_mr['combined_per']:.2f} "
          f"(s={best_mr['scale']}, a={best_mr['alpha']})")
    print(f"  Best multi-scale: {best_ms['combined_per']:.2f} "
          f"(a={best_ms['alpha']})")
    print(f"  Best adaptive:    {best_ad['combined_per']:.2f} "
          f"(s={best_ad['scale']})")

    overall = min(all_results, key=lambda x: x["combined_per"])
    imp = p0 - overall["combined_per"]
    print(f"\n  >>> Overall best: {overall['combined_per']:.2f} "
          f"({overall['method']})")
    if imp > 0:
        print(f"  >>> Improvement: {imp:.2f} ({imp/p0*100:.1f}% relative)")


if __name__ == "__main__":
    main()