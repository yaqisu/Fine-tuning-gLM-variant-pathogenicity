#!/usr/bin/env python3
"""
Step 6 — Compare evaluation results across multiple filter/stratify strategies.

Reads all_metrics.tsv files from a set of eval output directories and produces
a unified comparison: bar charts, heatmaps, and a long-format TSV.

Usage:
    # Compare a set of named eval directories
    python evaluation/06_compare_strategies.py \
        --dirs \
            "all=results/predictions/ClinVar.260309only.seq12k.pb/eval_all" \
            "rare_1e-3=results/predictions/ClinVar.260309only.seq12k.pb/eval_rare" \
            "conserved=results/predictions/ClinVar.260309only.seq12k.pb/eval_conserved" \
        --outdir results/predictions/ClinVar.260309only.seq12k.pb/comparison

    # Automatically collect strata subdirs produced by --mode stratify
    python evaluation/06_compare_strategies.py \
        --strat_dir results/predictions/ClinVar.260309only.seq12k.pb/strat_af \
        --outdir    results/predictions/ClinVar.260309only.seq12k.pb/strat_af_comparison

    # Focus on specific methods only
    python evaluation/06_compare_strategies.py \
        --dirs "all=..." "rare=..." \
        --methods "ours,REVEL_score,AlphaMissense_score,BayesDel_addAF_score" \
        --outdir results/.../comparison
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from core import COLORS, col_color, plot_comparison_across_strata


def parse_args():
    p = argparse.ArgumentParser(description="Compare evaluation results across strategies.")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--dirs",      nargs="+",
                       help="Named eval directories in 'label=path' format. "
                            "Each directory must contain all_metrics.tsv.")
    group.add_argument("--strat_dir", type=Path,
                       help="Root directory of a --mode stratify run. "
                            "Subdirectories are collected automatically.")

    p.add_argument("--outdir",   required=True, type=Path,
                   help="Output directory for comparison plots and tables")
    p.add_argument("--metric_file", default="all_metrics.tsv",
                   help="Filename to look for inside each eval directory "
                        "(default: all_metrics.tsv)")
    p.add_argument("--methods",  default=None,
                   help="Comma-separated list of method columns to focus on. "
                        "Default: ours + top 15 by mean AUROC across strata.")
    p.add_argument("--top_n",    default=15, type=int,
                   help="Number of top methods to include in plots (default: 15)")
    p.add_argument("--metrics",  default="auroc,prauc,pauroc_fpr10",
                   help="Comma-separated metrics to plot (default: auroc,prauc,pauroc_fpr10)")
    return p.parse_args()


def collect_from_dirs(dir_specs: list[str], metric_file: str) -> pd.DataFrame:
    """Parse 'label=path' pairs and load metric files."""
    frames = []
    for spec in dir_specs:
        if "=" not in spec:
            raise ValueError(f"Expected 'label=path' format, got: '{spec}'")
        label, path_str = spec.split("=", 1)
        mpath = Path(path_str.strip()) / metric_file
        if not mpath.exists():
            print(f"  WARNING: {mpath} not found — skipping '{label}'")
            continue
        df = pd.read_csv(mpath, sep="\t")
        df["stratum"] = label.strip()
        frames.append(df)
        print(f"  Loaded '{label}': {len(df)} methods from {mpath}")
    if not frames:
        raise FileNotFoundError("No metric files found. Check --dirs paths.")
    return pd.concat(frames, ignore_index=True)


def collect_from_strat_dir(strat_dir: Path, metric_file: str) -> pd.DataFrame:
    """Collect metric files from all subdirectories of a stratify run."""
    frames = []
    for subdir in sorted(strat_dir.iterdir()):
        if not subdir.is_dir():
            continue
        mpath = subdir / metric_file
        if not mpath.exists():
            continue
        df = pd.read_csv(mpath, sep="\t")
        df["stratum"] = subdir.name
        frames.append(df)
        print(f"  Loaded stratum '{subdir.name}': {len(df)} methods")
    if not frames:
        raise FileNotFoundError(f"No {metric_file} found under {strat_dir}")
    return pd.concat(frames, ignore_index=True)


def select_methods(comparison: pd.DataFrame, methods_arg: str | None,
                   top_n: int) -> list[str]:
    if methods_arg:
        return [m.strip() for m in methods_arg.split(",")]
    top = (comparison[comparison["column"] != "ours"]
           .dropna(subset=["auroc"])
           .groupby("column")["auroc"].mean()
           .nlargest(top_n).index.tolist())
    return ["ours"] + top


def plot_grouped_bar(comparison: pd.DataFrame, methods: list, metric: str,
                     out_path: Path) -> None:
    """
    Grouped bar chart: x = methods, groups = strata, y = metric value.
    One bar cluster per method.
    """
    strata  = comparison["stratum"].unique().tolist()
    focus   = comparison[comparison["column"].isin(methods)]

    x      = np.arange(len(methods))
    width  = min(0.8 / len(strata), 0.25)

    cmap   = plt.cm.get_cmap("tab10", len(strata))
    fig, ax = plt.subplots(figsize=(max(12, len(methods) * 0.8), 6))

    for i, stratum in enumerate(strata):
        sub  = focus[focus["stratum"] == stratum].set_index("column")
        vals = [sub.loc[m, metric] if m in sub.index else np.nan for m in methods]
        color = cmap(i)
        ax.bar(x + i * width - width * (len(strata) - 1) / 2, vals, width,
               label=stratum, color=color, alpha=0.85, edgecolor="white", lw=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(metric.upper(), fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="gray", ls=":", lw=0.8)
    ax.set_title(f"{metric.upper()} by Method and Stratum", fontsize=12)
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_method_heatmap(comparison: pd.DataFrame, methods: list,
                        metric: str, out_path: Path) -> None:
    """Heatmap: rows = methods, columns = strata, values = metric."""
    strata  = comparison["stratum"].unique().tolist()
    pivot   = (comparison[comparison["column"].isin(methods)]
               .pivot_table(index="column", columns="stratum", values=metric, aggfunc="first")
               .reindex(index=methods, columns=strata))

    data = pivot.values.astype(float)
    fig, ax = plt.subplots(figsize=(max(6, len(strata) * 1.5),
                                    max(4, len(methods) * 0.5)))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0.4, vmax=1.0)
    ax.set_xticks(range(len(strata)))
    ax.set_xticklabels(strata, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=8)
    if methods[0] == "ours":
        ax.get_yticklabels()[0].set_color("red")
    for i in range(len(methods)):
        for j in range(len(strata)):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=7, color="black")
    plt.colorbar(im, ax=ax, shrink=0.6, label=metric.upper())
    ax.set_title(f"{metric.upper()} Heatmap — Methods × Strata", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_rank_chart(comparison: pd.DataFrame, methods: list,
                    metric: str, out_path: Path) -> None:
    """
    Line chart showing rank of each method across strata.
    Lower rank = better.
    """
    strata = comparison["stratum"].unique().tolist()
    focus  = comparison[comparison["column"].isin(methods)]

    ranks_dict = {}
    for stratum in strata:
        sub = focus[focus["stratum"] == stratum][["column", metric]].dropna(subset=[metric])
        sub = sub.sort_values(metric, ascending=False).reset_index(drop=True)
        sub["rank"] = sub.index + 1
        for _, row in sub.iterrows():
            ranks_dict.setdefault(row["column"], {})[stratum] = row["rank"]

    fig, ax = plt.subplots(figsize=(max(8, len(strata) * 1.5), 6))
    x = np.arange(len(strata))
    for method in methods:
        ranks = [ranks_dict.get(method, {}).get(s, np.nan) for s in strata]
        lw    = 2.5 if method == "ours" else 1.0
        color = col_color(method)
        ax.plot(x, ranks, marker="o", lw=lw, color=color, label=method)

    ax.set_xticks(x)
    ax.set_xticklabels(strata, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(f"Rank by {metric.upper()} (1 = best)", fontsize=11)
    ax.set_title(f"Method Rank Across Strata ({metric.upper()})", fontsize=12)
    ax.invert_yaxis()
    ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    args    = parse_args()
    out_dir = args.outdir
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    metrics_to_plot = [m.strip() for m in args.metrics.split(",")]

    # ── Collect data ───────────────────────────────────────────────────────
    if args.dirs:
        print("Loading named eval directories ...")
        comparison = collect_from_dirs(args.dirs, args.metric_file)
    else:
        print(f"Collecting from stratify directory: {args.strat_dir}")
        comparison = collect_from_strat_dir(args.strat_dir, args.metric_file)

    print(f"\nTotal rows loaded: {len(comparison):,}")
    print(f"Strata: {comparison['stratum'].unique().tolist()}")

    # ── Save full comparison table ─────────────────────────────────────────
    comp_path = out_dir / "comparison_all_methods.tsv"
    comparison.to_csv(comp_path, sep="\t", index=False)
    print(f"\nSaved full comparison to {comp_path}")

    # ── Select methods to plot ─────────────────────────────────────────────
    methods = select_methods(comparison, args.methods, args.top_n)
    print(f"\nFocused methods ({len(methods)}): {methods}")

    # Save focused pivot table
    for metric in metrics_to_plot:
        piv = (comparison[comparison["column"].isin(methods)]
               .pivot_table(index="column", columns="stratum", values=metric, aggfunc="first")
               .reindex(index=methods))
        piv.to_csv(out_dir / f"pivot_{metric}.tsv", sep="\t")
        print(f"  Saved pivot_{metric}.tsv")

    # ── Plots ──────────────────────────────────────────────────────────────
    print("\n── Generating comparison plots ───────────────────────────────────")
    for metric in metrics_to_plot:
        plot_grouped_bar(comparison, methods, metric,
                         plots_dir / f"grouped_bar_{metric}.png")
        plot_method_heatmap(comparison, methods, metric,
                            plots_dir / f"heatmap_{metric}.png")
        plot_rank_chart(comparison, methods, metric,
                        plots_dir / f"rank_chart_{metric}.png")

    print(f"\n✓  Done.  Outputs in {out_dir}")


if __name__ == "__main__":
    main()
