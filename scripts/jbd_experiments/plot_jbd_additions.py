from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PALETTE = {
    "rabvs": "#C53A32",
    "greedy_partition": "#1F4E79",
    "boba_ucb": "#2C7A5B",
    "uncertainty_partition": "#8A8F98",
    "random_partition": "#D9DCE1",
    "ink": "#26282B",
    "grid": "#D9DCE1",
    "white": "#FFFFFF",
}

LABELS = {
    "rabvs": "RABVS",
    "greedy_partition": "Mean-score greedy",
    "boba_ucb": "BOBa-UCB",
    "uncertainty_partition": "Uncertainty",
    "random_partition": "Random",
}

mpl.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.size": 10.0,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 8.0,
        "axes.linewidth": 0.55,
        "lines.linewidth": 0.9,
        "lines.markersize": 4.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def style_axis(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(PALETTE["ink"])
        spine.set_linewidth(0.55)
    ax.tick_params(length=3.0, width=0.55, color=PALETTE["ink"], pad=3)
    ax.grid(axis="both", color=PALETTE["grid"], linewidth=0.35, alpha=0.75)
    ax.set_axisbelow(True)


def panel_label(ax: plt.Axes, label: str, x: float = 0.015, y: float = 0.985) -> None:
    ax.text(
        x,
        y,
        f"({label})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
        color=PALETTE["ink"],
        clip_on=False,
    )


def budget_figure(data: pd.DataFrame, out_dir: Path) -> None:
    summary = (
        data.groupby(["method", "nominal_budget_fraction"], as_index=False)
        .agg(
            realized_budget_fraction=("realized_budget_fraction", "mean"),
            hit_retention=("Hit@1000_retention", "mean"),
            bedroc=("BEDROC_alpha20", "mean"),
            yield_per_1000=("hits_per_1000_scores", "mean"),
        )
        .sort_values(["method", "nominal_budget_fraction"])
    )

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.0), constrained_layout=True)
    metrics = [
        ("hit_retention", "Hit@1000 retention"),
        ("bedroc", "BEDROC (alpha = 20)"),
        ("yield_per_1000", "Hits per 1,000 scores"),
    ]
    method_order = [
        "rabvs",
        "greedy_partition",
        "boba_ucb",
        "uncertainty_partition",
        "random_partition",
    ]
    for panel, (ax, (column, ylabel)) in enumerate(zip(axes, metrics)):
        for method in method_order:
            block = summary[summary.method == method]
            ax.plot(
                100 * block.nominal_budget_fraction,
                block[column],
                marker="o",
                markerfacecolor=PALETTE[method],
                markeredgecolor=PALETTE["ink"],
                markeredgewidth=0.35,
                color=PALETTE[method],
                label=LABELS[method],
                zorder=5 if method == "rabvs" else 3,
            )
        ax.set_xscale("log")
        ax.set_xticks([1, 2.5, 5, 10, 20, 50], ["1", "2.5", "5", "10", "20", "50"])
        ax.set_xlabel("Nominal scoring budget (%)")
        ax.set_ylabel(ylabel)
        panel_label(ax, chr(ord("a") + panel))
        style_axis(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=5,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.1,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ["pdf", "png"]:
        fig.savefig(
            out_dir / f"Fig7_budget_quality.{suffix}",
            dpi=650,
            bbox_inches="tight",
            pad_inches=5 / 72,
            facecolor="white",
        )
    plt.close(fig)


def sensitivity_figure(data: pd.DataFrame, out_dir: Path) -> None:
    summary = (
        data.groupby(["clean_weight", "novelty_weight"], as_index=False)
        .agg(
            hit=("Hit@1000", "mean"),
            bedroc=("BEDROC_alpha20", "mean"),
            clean=("alert_free_ratio_at1000", "mean"),
            scaffold=("unique_scaffolds_at1000", "mean"),
        )
    )
    metrics = [
        ("hit", "Mean Hit@1000", ".1f"),
        ("bedroc", "Mean BEDROC", ".3f"),
        ("clean", "Alert-free ratio", ".3f"),
        ("scaffold", "Unique scaffolds", ".0f"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.0), constrained_layout=True)
    for panel, (ax, (column, colorbar_label, fmt)) in enumerate(zip(axes.ravel(), metrics)):
        pivot = summary.pivot(index="clean_weight", columns="novelty_weight", values=column)
        image = ax.imshow(pivot.values, cmap="RdBu_r", aspect="auto")
        for row in range(pivot.shape[0]):
            for col in range(pivot.shape[1]):
                value = pivot.values[row, col]
                ax.text(col, row, format(value, fmt), ha="center", va="center", fontsize=8)
        ax.set_xticks(np.arange(pivot.shape[1]), [f"{x:.2f}" for x in pivot.columns])
        ax.set_yticks(np.arange(pivot.shape[0]), [f"{x:.2f}" for x in pivot.index])
        ax.set_xlabel("Novelty weight")
        ax.set_ylabel("Cleanliness weight")
        panel_label(ax, chr(ord("a") + panel), -0.075, 1.055)
        style_axis(ax)
        colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.025)
        colorbar.set_label(colorbar_label, fontsize=9)
        colorbar.ax.tick_params(labelsize=8, width=0.55)
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ["pdf", "png"]:
        fig.savefig(
            out_dir / f"FigS5_rrf_sensitivity.{suffix}",
            dpi=650,
            bbox_inches="tight",
            pad_inches=5 / 72,
            facecolor="white",
        )
    plt.close(fig)


def scale_figure(data: pd.DataFrame, amortization: pd.DataFrame, out_dir: Path) -> None:
    summary = (
        data.groupby("library_size", as_index=False)
        .agg(
            scores_avoided=("scores_avoided", "mean"),
            overlap=("topK_overlap", "mean"),
            full_seconds=("full_score_seconds", "mean"),
            online_seconds=("online_seconds", "mean"),
            speedup=("online_speedup", "mean"),
            rss=("peak_rss_gib", "max"),
            index_seconds=("index_seconds", "mean"),
        )
        .sort_values("library_size")
    )
    size_m = summary.library_size / 1_000_000
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.9), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(size_m, summary.scores_avoided, marker="o", color=PALETTE["rabvs"], label="Unique scores avoided")
    ax.plot(size_m, summary.overlap, marker="o", color=PALETTE["greedy_partition"], label="Top-1000 overlap")
    ax.set_ylabel("Fraction")
    ax.set_ylim(0.4, 1.0)
    ax.legend(loc="lower right", frameon=False)

    ax = axes[0, 1]
    ax.plot(size_m, summary.full_seconds, marker="o", color=PALETTE["greedy_partition"], label="Exhaustive scoring")
    ax.plot(size_m, summary.online_seconds, marker="o", color=PALETTE["rabvs"], label="RABVS online")
    ax.set_ylabel("Target-specific time (s)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.12, 1.0), frameon=False)

    ax = axes[1, 0]
    ax.plot(size_m, summary.rss, marker="o", color=PALETTE["boba_ucb"])
    ax.set_ylabel("Peak RSS (GiB)")
    ax.set_xlabel("Library size (millions)")

    ax = axes[1, 1]
    ax.plot(
        amortization.target_count,
        amortization.amortized_speedup,
        marker="o",
        color=PALETTE["rabvs"],
    )
    ax.axhline(1.0, color=PALETTE["ink"], linewidth=0.55, linestyle="--")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100], ["1", "2", "5", "10", "20", "50", "100"])
    ax.set_ylabel("End-to-end speedup")
    ax.set_xlabel("Targets sharing one index")

    for panel, ax in enumerate(axes.ravel()):
        if panel < 2:
            ax.set_xlabel("Library size (millions)")
        panel_label(ax, chr(ord("a") + panel))
        style_axis(ax)

    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ["pdf", "png"]:
        fig.savefig(
            out_dir / f"Fig8_chembl36_scaling.{suffix}",
            dpi=650,
            bbox_inches="tight",
            pad_inches=5 / 72,
            facecolor="white",
        )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", required=True)
    parser.add_argument("--sensitivity", required=True)
    parser.add_argument("--scale")
    parser.add_argument("--amortization")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    budget_figure(pd.read_csv(args.budget), out_dir)
    sensitivity_figure(pd.read_csv(args.sensitivity), out_dir)
    if args.scale and args.amortization:
        scale_figure(pd.read_csv(args.scale), pd.read_csv(args.amortization), out_dir)


if __name__ == "__main__":
    main()
