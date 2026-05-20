#!/usr/bin/env python3
"""Dissertation Plots and Tables Generator.

Reads experiment CSVs and produces:
  1. Publication-quality matplotlib figures (saved as PDF)
  2. LaTeX table code (saved as .tex files)

Covers all experiments: scale sweep, resolution sweep, random ablation,
combination grid, and training curves.

Usage:
    python plot_results.py --results_dir <path_to_results>
    
    If --results_dir not given, looks for CSVs in current directory.
    Outputs saved to ./dissertation_figures/

This script does NOT require GPU or SpeechBrain.
"""

import os
import csv
import argparse
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Publication style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.5,
    "lines.markersize": 6,
})


def read_csv(path):
    """Read a CSV file into a list of dicts."""
    if not os.path.exists(path):
        print(f"  Warning: {path} not found, skipping")
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def safe_float(val, default=None):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def plot_scale_sweep(data, p0_per, out_dir):
    """Plot scale sweep: Flow PER vs scale."""
    rows = [r for r in data if r.get("experiment") == "fine_scale_sweep"
            or (r.get("experiment", "") == "" and r.get("random") == "False")]
    if not rows:
        # Try alternate format
        rows = [r for r in data if safe_float(r.get("temperature")) == 1.0
                and r.get("random") == "False"
                and safe_float(r.get("num_passes")) == 1]
    if not rows:
        print("  No scale sweep data found")
        return

    scales = [safe_float(r["base_scale"]) for r in rows]
    flow_pers = [safe_float(r["flow_per"]) for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(scales, flow_pers, "b-o", label="Flow PER", zorder=3)
    ax.axhline(y=p0_per, color="gray", linestyle="--", linewidth=1,
               label=f"p0 baseline ({p0_per:.2f}%)")

    # Find and annotate optimum
    best_idx = np.argmin(flow_pers)
    ax.annotate(f"Optimal: scale={scales[best_idx]:.0f}\n"
                f"PER={flow_pers[best_idx]:.2f}%",
                xy=(scales[best_idx], flow_pers[best_idx]),
                xytext=(scales[best_idx] + 2, flow_pers[best_idx] - 0.5),
                arrowprops=dict(arrowstyle="->", color="blue"),
                fontsize=10, color="blue")

    ax.set_xlabel("Velocity Inference Scale ($\\lambda_{\\mathrm{scale}}$)")
    ax.set_ylabel("Test PER (%)")
    ax.set_title("Scale Sweep")
    ax.legend()

    path = os.path.join(out_dir, "scale_sweep.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_scale_sweep_wide(data, p0_per, out_dir):
    """Plot scale sweep with full range including catastrophic region."""
    rows = [r for r in data if r.get("random") == "False"
            and safe_float(r.get("num_passes", 1)) == 1
            and safe_float(r.get("temperature", 1.0)) == 1.0]
    if not rows:
        print("  No wide scale data found")
        return

    scales = sorted(set(safe_float(r["base_scale"]) for r in rows))
    flow_map = {}
    for r in rows:
        s = safe_float(r["base_scale"])
        flow_map[s] = safe_float(r["flow_per"])

    scales_plot = [s for s in scales if s in flow_map]
    flows_plot = [flow_map[s] for s in scales_plot]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(scales_plot, flows_plot, "b-o", label="Flow PER", zorder=3)
    ax.axhline(y=p0_per, color="gray", linestyle="--",
               label=f"p0 baseline ({p0_per:.2f}%)")

    # Shade regions
    best_idx = np.argmin(flows_plot)
    best_scale = scales_plot[best_idx]
    ax.axvspan(0, best_scale, alpha=0.05, color="green",
               label="Improvement region")
    if best_scale < max(scales_plot):
        ax.axvspan(best_scale, max(scales_plot), alpha=0.05, color="red",
                   label="Over-correction region")

    ax.set_xlabel("Velocity Inference Scale ($\\lambda_{\\mathrm{scale}}$)")
    ax.set_ylabel("Test PER (%)")
    ax.set_title("Scale Sweep — Full Range")
    ax.legend(loc="upper left")

    path = os.path.join(out_dir, "scale_sweep_wide.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_learned_vs_random(data, p0_per, out_dir):
    """Plot learned vs random direction ablation."""
    learned = [r for r in data if r.get("random") == "False"
               and safe_float(r.get("num_passes", 1)) == 1
               and safe_float(r.get("temperature", 1.0)) == 1.0]
    random = [r for r in data if r.get("random") == "True"]

    if not learned or not random:
        print("  No random ablation data found")
        return

    l_scales = sorted(safe_float(r["base_scale"]) for r in learned)
    r_scales = sorted(safe_float(r["base_scale"]) for r in random)

    l_map = {safe_float(r["base_scale"]): safe_float(r["flow_per"])
             for r in learned}
    r_map = {safe_float(r["base_scale"]): safe_float(r["flow_per"])
             for r in random}

    common = sorted(set(l_map.keys()) & set(r_map.keys()))
    if not common:
        print("  No overlapping scales for comparison")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(common, [l_map[s] for s in common], "b-o",
            label="Learned directions")
    ax.plot(common, [r_map[s] for s in common], "r--s",
            label="Random directions")
    ax.axhline(y=p0_per, color="gray", linestyle=":", linewidth=1,
               label=f"p0 baseline ({p0_per:.2f}%)")

    ax.set_xlabel("Velocity Inference Scale ($\\lambda_{\\mathrm{scale}}$)")
    ax.set_ylabel("Test PER (%)")
    ax.set_title("Learned vs Random Direction Ablation")
    ax.legend()

    path = os.path.join(out_dir, "learned_vs_random.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_resolution_sweep(data, p0_per, out_dir):
    """Plot resolution sweep: PER vs number of Euler steps."""
    rows = [r for r in data if r.get("experiment", "").startswith("resolution")]
    if not rows:
        print("  No resolution sweep data found")
        return

    steps = [safe_float(r["num_steps"]) for r in rows]
    flows = [safe_float(r["flow_per"]) for r in rows]

    sorted_pairs = sorted(zip(steps, flows))
    steps, flows = zip(*sorted_pairs)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, flows, "b-o", label="Flow PER", zorder=3)
    ax.axhline(y=p0_per, color="gray", linestyle="--",
               label=f"p0 baseline ({p0_per:.2f}%)")

    # Annotate key points
    best_idx = np.argmin(flows)
    ax.annotate(f"S={steps[best_idx]:.0f}\n({flows[best_idx]:.2f}%)",
                xy=(steps[best_idx], flows[best_idx]),
                xytext=(steps[best_idx] + 8, flows[best_idx] - 0.8),
                arrowprops=dict(arrowstyle="->", color="blue"),
                fontsize=10, color="blue")

    # Add ms per step on top axis
    ax2 = ax.twiny()
    ms_ticks = [200, 100, 50, 33, 20, 10]
    s_positions = [1000 / ms for ms in ms_ticks]
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(s_positions)
    ax2.set_xticklabels([f"{ms}" for ms in ms_ticks])
    ax2.set_xlabel("Milliseconds per step")

    ax.set_xlabel("Number of Euler Steps ($S$)")
    ax.set_ylabel("Test PER (%)")
    ax.set_title("Temporal Resolution Sweep")
    ax.legend(loc="upper right")

    path = os.path.join(out_dir, "resolution_sweep.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_combination_heatmap(data, p0_per, out_dir):
    """Plot combination grid as a heatmap."""
    rows = [r for r in data if r.get("method", "linear") == "linear"]
    if not rows:
        rows = data  # Try all rows
    if not rows:
        print("  No combination data found")
        return

    scales = sorted(set(safe_float(r.get("scale", r.get("flow_scale", 0)))
                        for r in rows))
    alphas = sorted(set(safe_float(r.get("alpha", 1.0)) for r in rows))

    if len(scales) < 2 or len(alphas) < 2:
        print("  Not enough grid points for heatmap")
        return

    grid = np.full((len(scales), len(alphas)), np.nan)
    for r in rows:
        s = safe_float(r.get("scale", r.get("flow_scale")))
        a = safe_float(r.get("alpha"))
        per = safe_float(r.get("combined_per"))
        if s in scales and a in alphas and per is not None:
            si = scales.index(s)
            ai = alphas.index(a)
            grid[si, ai] = per

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(grid, aspect="auto", cmap="RdYlGn_r",
                   vmin=max(p0_per - 0.8, np.nanmin(grid)),
                   vmax=min(p0_per + 1.0, np.nanmax(grid)))

    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([f"{a:.2f}" for a in alphas], rotation=45)
    ax.set_yticks(range(len(scales)))
    ax.set_yticklabels([f"{s:.0f}" for s in scales])

    ax.set_xlabel("Interpolation Weight ($\\alpha$)")
    ax.set_ylabel("Flow Scale ($\\lambda_{\\mathrm{scale}}$)")
    ax.set_title("System Combination PER (%)")

    # Annotate cells
    for i in range(len(scales)):
        for j in range(len(alphas)):
            if not np.isnan(grid[i, j]):
                color = "white" if grid[i, j] > p0_per else "black"
                weight = "bold" if grid[i, j] == np.nanmin(grid) else "normal"
                ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                        fontsize=8, color=color, fontweight=weight)

    plt.colorbar(im, ax=ax, label="Combined PER (%)")

    path = os.path.join(out_dir, "combination_heatmap.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_combination_alpha_curves(data, p0_per, out_dir):
    """Plot combination PER vs alpha for best scales."""
    rows = [r for r in data if r.get("method", "linear") == "linear"]
    if not rows:
        rows = data
    if not rows:
        print("  No combination data for alpha curves")
        return

    # Group by scale
    scale_groups = {}
    for r in rows:
        s = safe_float(r.get("scale", r.get("flow_scale")))
        a = safe_float(r.get("alpha"))
        per = safe_float(r.get("combined_per"))
        if s is not None and a is not None and per is not None:
            scale_groups.setdefault(s, []).append((a, per))

    # Pick top 4 most interesting scales
    best_per_scale = {s: min(v, key=lambda x: x[1])[1]
                      for s, v in scale_groups.items()}
    top_scales = sorted(best_per_scale.keys(),
                        key=lambda s: best_per_scale[s])[:4]

    colors = ["blue", "red", "green", "orange"]
    markers = ["o", "s", "^", "D"]

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, s in enumerate(top_scales):
        points = sorted(scale_groups[s])
        as_vals, pers = zip(*points)
        ax.plot(as_vals, pers, f"-{markers[i]}", color=colors[i],
                label=f"scale={s:.0f}")

    ax.axhline(y=p0_per, color="gray", linestyle="--",
               label=f"p0 baseline ({p0_per:.2f}%)")

    ax.set_xlabel("Interpolation Weight ($\\alpha$)")
    ax.set_ylabel("Combined PER (%)")
    ax.set_title("System Combination: PER vs $\\alpha$")
    ax.legend()
    ax.invert_xaxis()  # alpha=1 (pure p0) on left

    path = os.path.join(out_dir, "combination_alpha_curves.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def generate_latex_table(data, p0_per, experiment_name, out_dir):
    """Generate LaTeX table code for an experiment."""
    if not data:
        return

    tex_path = os.path.join(out_dir, f"table_{experiment_name}.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        if experiment_name == "scale_sweep":
            f.write("\\begin{table}[htbp]\n\\centering\n")
            f.write("\\caption{Scale sweep results.}\n")
            f.write("\\label{tab:scale_sweep}\n")
            f.write("\\begin{tabular}{rccl}\n\\toprule\n")
            f.write("$\\lambda_{\\text{scale}}$ & "
                    "\\textbf{Flow PER (\\%)} & "
                    "\\textbf{p0 PER (\\%)} & "
                    "\\textbf{Gap} \\\\\n\\midrule\n")
            rows = sorted(data, key=lambda x: safe_float(x["base_scale"]))
            best_per = min(safe_float(r["flow_per"]) for r in rows)
            for r in rows:
                s = safe_float(r["base_scale"])
                fp = safe_float(r["flow_per"])
                pp = safe_float(r["p0_per"])
                gap = fp - pp
                bold = fp == best_per
                if bold:
                    f.write(f"\\textbf{{{s:.0f}}} & "
                            f"\\textbf{{{fp:.2f}}} & "
                            f"\\textbf{{{pp:.2f}}} & "
                            f"$\\mathbf{{{gap:+.2f}}}$ \\\\\n")
                else:
                    f.write(f"{s:.0f} & {fp:.2f} & {pp:.2f} & "
                            f"${gap:+.2f}$ \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

        elif experiment_name == "combination":
            rows = [r for r in data
                    if r.get("method", "linear") == "linear"]
            if not rows:
                rows = data
            scales = sorted(set(safe_float(
                r.get("scale", r.get("flow_scale", 0))) for r in rows))
            alphas = sorted(set(safe_float(
                r.get("alpha", 1.0)) for r in rows), reverse=True)

            # Build lookup
            lookup = {}
            for r in rows:
                s = safe_float(r.get("scale", r.get("flow_scale")))
                a = safe_float(r.get("alpha"))
                lookup[(s, a)] = safe_float(r.get("combined_per"))

            best_per = min(v for v in lookup.values() if v is not None)

            n_alpha = len(alphas)
            f.write("\\begin{table}[htbp]\n\\centering\n")
            f.write("\\caption{System combination grid (combined PER \\%).}\n")
            f.write("\\label{tab:combination}\n")
            f.write(f"\\small\n\\begin{{tabular}}{{r{'c' * n_alpha}}}\n")
            f.write("\\toprule\n")
            f.write(" & " + " & ".join(
                f"$\\alpha={a:.2f}$" for a in alphas) + " \\\\\n")
            f.write("\\midrule\n")
            for s in scales:
                row_parts = [f"{s:.0f}"]
                for a in alphas:
                    per = lookup.get((s, a))
                    if per is not None:
                        if per == best_per:
                            row_parts.append(f"\\textbf{{{per:.2f}}}")
                        elif per < p0_per - 0.01:
                            row_parts.append(f"{per:.2f}*")
                        else:
                            row_parts.append(f"{per:.2f}")
                    else:
                        row_parts.append("---")
                f.write(" & ".join(row_parts) + " \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n")
            f.write(f"\\begin{{flushleft}}\n\\small p0 PER = {p0_per:.2f}\\%. "
                    f"* indicates improvement over p0. "
                    f"Bold = best.\n\\end{{flushleft}}\n")
            f.write("\\end{table}\n")

    print(f"  Saved: {tex_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate dissertation plots and tables")
    parser.add_argument("--results_dir", type=str, default=".",
                        help="Directory containing experiment CSVs")
    parser.add_argument("--output_dir", type=str,
                        default="dissertation_figures",
                        help="Output directory for figures and tables")
    parser.add_argument("--p0_per", type=float, default=None,
                        help="p0 baseline PER (auto-detected if not given)")
    args = parser.parse_args()

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # Try to find CSVs
    rd = args.results_dir
    experiment_csv = None
    combination_csv = None

    for name in ["experiment_results_refined.csv",
                 "experiment_results.csv"]:
        path = os.path.join(rd, name)
        if os.path.exists(path):
            experiment_csv = path
            break

    for name in ["combination_extended.csv",
                 "combination_results.csv"]:
        path = os.path.join(rd, name)
        if os.path.exists(path):
            combination_csv = path
            break

    # Read data
    exp_data = read_csv(experiment_csv) if experiment_csv else []
    comb_data = read_csv(combination_csv) if combination_csv else []

    print(f"Loaded {len(exp_data)} experiment rows, "
          f"{len(comb_data)} combination rows")

    # Detect p0 baseline
    p0_per = args.p0_per
    if p0_per is None:
        for r in exp_data:
            v = safe_float(r.get("p0_per"))
            if v is not None:
                p0_per = v
                break
    if p0_per is None:
        for r in comb_data:
            v = safe_float(r.get("p0_per"))
            if v is not None:
                p0_per = v
                break
    if p0_per is None:
        print("ERROR: Could not detect p0 PER. Use --p0_per flag.")
        return

    print(f"p0 baseline: {p0_per:.2f}%")
    print(f"Output directory: {out_dir}")

    # Generate plots
    print("\nGenerating plots...")

    if exp_data:
        # Scale sweep
        scale_data = [r for r in exp_data
                      if r.get("random") == "False"
                      and safe_float(r.get("num_passes", 1)) == 1]
        if scale_data:
            plot_scale_sweep(exp_data, p0_per, out_dir)
            plot_scale_sweep_wide(exp_data, p0_per, out_dir)

        # Random ablation
        random_data = [r for r in exp_data if r.get("random") == "True"]
        if random_data:
            plot_learned_vs_random(exp_data, p0_per, out_dir)

        # Resolution sweep
        res_data = [r for r in exp_data
                    if "resolution" in r.get("experiment", "")]
        if res_data:
            plot_resolution_sweep(exp_data, p0_per, out_dir)

        # LaTeX table for scale sweep
        scale_rows = [r for r in exp_data
                      if r.get("experiment", "") in
                      ("fine_scale_sweep", "scale_sweep")
                      and r.get("random") == "False"]
        if scale_rows:
            generate_latex_table(scale_rows, p0_per, "scale_sweep", out_dir)

    if comb_data:
        plot_combination_heatmap(comb_data, p0_per, out_dir)
        plot_combination_alpha_curves(comb_data, p0_per, out_dir)
        generate_latex_table(comb_data, p0_per, "combination", out_dir)

    # Generate summary table of ALL results
    print("\nGenerating summary table...")
    summary_path = os.path.join(out_dir, "table_summary.tex")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n")
        f.write("\\caption{Summary of all experimental results.}\n")
        f.write("\\label{tab:all_results}\n")
        f.write("\\begin{tabular}{llccc}\n\\toprule\n")
        f.write("\\textbf{Experiment} & \\textbf{Configuration} & "
                "\\textbf{Best PER (\\%)} & \\textbf{p0 PER (\\%)} & "
                "\\textbf{Improvement} \\\\\n\\midrule\n")

        entries = []

        # Best from scale sweep
        if exp_data:
            scale_rows = [r for r in exp_data
                          if r.get("random") == "False"
                          and safe_float(r.get("num_passes", 1)) == 1]
            if scale_rows:
                best = min(scale_rows, key=lambda x: safe_float(x["flow_per"]))
                entries.append((
                    "Scale sweep",
                    f"$\\lambda_{{\\text{{scale}}}}="
                    f"{safe_float(best['base_scale']):.0f}$",
                    safe_float(best["flow_per"]),
                    p0_per
                ))

            # Best from random ablation
            rand_rows = [r for r in exp_data if r.get("random") == "True"]
            if rand_rows:
                best = min(rand_rows,
                           key=lambda x: safe_float(x["flow_per"]))
                entries.append((
                    "Random ablation",
                    f"scale={safe_float(best['base_scale']):.0f}",
                    safe_float(best["flow_per"]),
                    p0_per
                ))

            # Best from resolution sweep
            res_rows = [r for r in exp_data
                        if "resolution" in r.get("experiment", "")]
            if res_rows:
                best = min(res_rows,
                           key=lambda x: safe_float(x["flow_per"]))
                entries.append((
                    "Resolution sweep",
                    f"$S={safe_float(best['num_steps']):.0f}$",
                    safe_float(best["flow_per"]),
                    p0_per
                ))

        # Best from combination
        if comb_data:
            linear = [r for r in comb_data
                      if r.get("method", "linear") == "linear"]
            if linear:
                best = min(linear,
                           key=lambda x: safe_float(x["combined_per"]))
                s = safe_float(best.get("scale", best.get("flow_scale")))
                a = safe_float(best.get("alpha"))
                entries.append((
                    "System combination",
                    f"scale={s:.0f}, $\\alpha={a:.2f}$",
                    safe_float(best["combined_per"]),
                    p0_per
                ))

            loglin = [r for r in comb_data
                      if r.get("method") == "log_linear"]
            if loglin:
                best = min(loglin,
                           key=lambda x: safe_float(x["combined_per"]))
                s = safe_float(best.get("scale", best.get("flow_scale")))
                a = safe_float(best.get("alpha"))
                entries.append((
                    "Log-linear combination",
                    f"scale={s:.0f}, $\\alpha={a:.2f}$",
                    safe_float(best["combined_per"]),
                    p0_per
                ))

        for name, config, best_per, baseline in entries:
            imp = baseline - best_per
            imp_str = f"$-{imp:.2f}$" if imp > 0 else f"$+{abs(imp):.2f}$"
            f.write(f"{name} & {config} & {best_per:.2f} & "
                    f"{baseline:.2f} & {imp_str} \\\\\n")

        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"  Saved: {summary_path}")

    print(f"\nDone! All outputs in: {out_dir}/")
    print("Include PDFs in LaTeX with: "
          "\\includegraphics[width=\\textwidth]{figures/scale_sweep.pdf}")


if __name__ == "__main__":
    main()
