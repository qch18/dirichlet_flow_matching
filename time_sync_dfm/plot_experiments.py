#!/usr/bin/env python3
"""Plot experiment results from Stage 13 evaluation.

Generates three publication-ready figures:
  1. Scale sweep: learned vs random directions
  2. Resolution sweep: PER vs temporal resolution
  3. Combined summary figure

Usage:
    python plot_experiments.py results/updated_ctc/dfm_stage13_direction_decoupled/experiment_results.json

Outputs PNG files in the same directory as the JSON.
"""

import sys
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def load_results(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def plot_scale_sweep(results, output_dir):
    """Figure 1: Scale sweep with learned vs random directions."""
    scale_data = results["scale_sweep"]
    random_data = results["random_ablation"]

    scales_learned = [r["scale"] for r in scale_data]
    per_learned = [r["flow_per"] for r in scale_data]
    p0_per = scale_data[0]["p0_per"]  # scale=0 = p0 PER

    scales_random = [r["scale"] for r in random_data]
    per_random = [r["flow_per"] for r in random_data]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(scales_learned, per_learned, "b-o", linewidth=2, markersize=6,
            label="Learned directions", zorder=3)
    ax.plot(scales_random, per_random, "r--s", linewidth=2, markersize=6,
            label="Random directions", zorder=3)
    ax.axhline(y=p0_per, color="gray", linestyle=":", linewidth=1.5,
               label=f"p0 baseline (no flow) = {p0_per:.2f}%", zorder=2)

    # Mark optimal scale
    optimal_scale = results["optimal_scale"]
    best_per = min(per_learned)
    ax.annotate(
        f"Optimal: scale={optimal_scale}\nPER={best_per:.2f}%",
        xy=(optimal_scale, best_per),
        xytext=(optimal_scale + 5, best_per + 2),
        arrowprops=dict(arrowstyle="->", color="blue"),
        fontsize=10, color="blue",
    )

    ax.set_xlabel("Velocity Inference Scale", fontsize=12)
    ax.set_ylabel("Test PER (%)", fontsize=12)
    ax.set_title("Effect of Velocity Scale on Flow PER\n"
                 "(Learned vs Random Directions, S=20)", fontsize=13)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1, max(scales_learned) + 2)

    plt.tight_layout()
    path = os.path.join(output_dir, "fig_scale_sweep.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_resolution_sweep(results, output_dir):
    """Figure 2: Resolution sweep showing temporal control."""
    res_data = results["resolution_sweep"]
    optimal_scale = results["optimal_scale"]

    steps = [r["num_steps"] for r in res_data]
    per_flow = [r["flow_per"] for r in res_data]
    per_p0 = [r["p0_per"] for r in res_data]
    ms_per_step = [r.get("ms_per_step", round(1000 / r["num_steps"]))
                   for r in res_data]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(steps, per_flow, "b-o", linewidth=2, markersize=6,
             label="Flow PER", zorder=3)
    ax1.plot(steps, per_p0, "gray", linestyle=":", linewidth=1.5,
             label=f"p0 PER (no flow)", zorder=2)

    ax1.set_xlabel("Number of Euler Steps (S)", fontsize=12)
    ax1.set_ylabel("Test PER (%)", fontsize=12)
    ax1.set_title(f"Temporal Resolution Control\n"
                  f"(scale={optimal_scale}, varying number of Euler steps)",
                  fontsize=13)

    # Add secondary x-axis for ms per step
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    tick_positions = steps
    tick_labels = [f"~{ms}ms" for ms in ms_per_step]
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels, fontsize=8, rotation=45)
    ax2.set_xlabel("Approximate temporal resolution per step", fontsize=10)

    ax1.legend(fontsize=10, loc="upper right")
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "fig_resolution_sweep.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_combined_summary(results, output_dir):
    """Figure 3: Combined 2-panel figure for dissertation."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Scale sweep (learned vs random)
    scale_data = results["scale_sweep"]
    random_data = results["random_ablation"]
    optimal_scale = results["optimal_scale"]

    scales_learned = [r["scale"] for r in scale_data]
    per_learned = [r["flow_per"] for r in scale_data]
    p0_per = scale_data[0]["p0_per"]

    scales_random = [r["scale"] for r in random_data]
    per_random = [r["flow_per"] for r in random_data]

    ax1.plot(scales_learned, per_learned, "b-o", linewidth=2, markersize=5,
             label="Learned directions")
    ax1.plot(scales_random, per_random, "r--s", linewidth=2, markersize=5,
             label="Random directions")
    ax1.axhline(y=p0_per, color="gray", linestyle=":", linewidth=1.5,
                label=f"p0 baseline = {p0_per:.2f}%")
    ax1.set_xlabel("Velocity Inference Scale", fontsize=11)
    ax1.set_ylabel("Test PER (%)", fontsize=11)
    ax1.set_title("(a) Scale Sweep: Learned vs Random", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel B: Resolution sweep
    res_data = results["resolution_sweep"]
    steps = [r["num_steps"] for r in res_data]
    per_flow = [r["flow_per"] for r in res_data]
    per_p0_res = [r["p0_per"] for r in res_data]

    ax2.plot(steps, per_flow, "b-o", linewidth=2, markersize=5,
             label="Flow PER")
    ax2.plot(steps, per_p0_res, "gray", linestyle=":", linewidth=1.5,
             label="p0 PER (no flow)")
    ax2.set_xlabel("Number of Euler Steps (S)", fontsize=11)
    ax2.set_ylabel("Test PER (%)", fontsize=11)
    ax2.set_title(f"(b) Resolution Sweep (scale={optimal_scale})", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "fig_combined_summary.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_experiments.py <path_to_experiment_results.json>")
        sys.exit(1)

    json_path = sys.argv[1]
    output_dir = os.path.dirname(json_path)

    results = load_results(json_path)
    plot_scale_sweep(results, output_dir)
    plot_resolution_sweep(results, output_dir)
    plot_combined_summary(results, output_dir)

    print(f"\nAll figures saved to: {output_dir}")


if __name__ == "__main__":
    main()