#!/usr/bin/env python3
"""
Step 4 — Evaluate all methods on the full merged dataset.

Defines a 'shared evaluation subset' by requiring valid scores in anchor columns
(default: REVEL_score + AlphaMissense_score) so every method is compared on
the same variants.

Usage:
    # Full dataset (all four classes)
    python evaluation/04_evaluate_all.py \
        --merged    results/predictions/ClinVar.260309only.seq12k.pb/merged.tsv \
        --outdir    results/predictions/ClinVar.260309only.seq12k.pb/eval_all

    # Restrict to a variant subset (e.g. P+B only)
    python evaluation/04_evaluate_all.py \
        --merged    results/predictions/ClinVar.260309only.seq12k.pb/merged.tsv \
        --outdir    results/predictions/ClinVar.260309only.seq12k.pb/eval_pb \
        --subset    data/sequences/pb_variant_ids.tsv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Allow running from project root without installation
sys.path.insert(0, str(Path(__file__).parent))
from core import (
    evaluate_all_columns, build_metrics_df, build_summary,
    apply_anchor_filter,
    plot_roc_curves, plot_pr_curves, plot_auroc_barplot,
    plot_metrics_heatmap, plot_auroc_scatter,
)

SKIP_COLS = {"variant_id", "chromosome", "position", "ref_allele", "alt_allele",
             "predicted_label"}


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate all methods on merged variants.")
    p.add_argument("--merged",       required=True,  type=Path)
    p.add_argument("--outdir",       required=True,  type=Path)
    p.add_argument("--our_col",      default="pathogenicity_score")
    p.add_argument("--label_col",    default="true_label")
    p.add_argument("--anchor_cols",  default="REVEL_score,AlphaMissense_score",
                   help="Comma-separated columns required to be non-missing in the "
                        "shared evaluation subset (default: REVEL_score,AlphaMissense_score)")
    p.add_argument("--top_n",        default=20, type=int)
    p.add_argument("--subset",       default=None, type=Path,
                   help="Optional TSV with 'variant_id' column to restrict evaluation")
    return p.parse_args()


def load_and_filter_subset(merged: pd.DataFrame, subset_path: Path) -> pd.DataFrame:
    print(f"\n── Subset filter ─────────────────────────────────────────────────")
    sub = pd.read_csv(subset_path, sep="\t", low_memory=False)
    if "variant_id" not in sub.columns:
        raise ValueError(f"'variant_id' not in subset file. Got: {list(sub.columns)}")
    ids      = set(sub["variant_id"].dropna().astype(str))
    n_before = len(merged)
    merged   = merged[merged["variant_id"].astype(str).isin(ids)].copy().reset_index(drop=True)
    print(f"  Subset IDs:        {len(ids):,}")
    print(f"  Matched in merged: {len(merged):,} / {n_before:,} variants kept")
    if "source_label" in sub.columns:
        matched = sub[sub["variant_id"].astype(str).isin(merged["variant_id"].astype(str))]
        print(f"  Source labels: {matched['source_label'].value_counts().to_dict()}")
    return merged


def main():
    args        = parse_args()
    out_dir     = args.outdir
    plots_dir   = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(exist_ok=True)

    anchor_cols = [c.strip() for c in args.anchor_cols.split(",")]

    # ── Load ───────────────────────────────────────────────────────────────
    print(f"Loading {args.merged}")
    merged = pd.read_csv(args.merged, sep="\t", low_memory=False)
    print(f"  {merged.shape[0]:,} variants × {merged.shape[1]} columns")

    if args.subset:
        merged = load_and_filter_subset(merged, args.subset)

    # ── Build shared evaluation subset ────────────────────────────────────
    print(f"\n── Building shared evaluation subset ─────────────────────────────")
    print(f"  Requiring valid values in: {anchor_cols}")
    mask   = apply_anchor_filter(merged, anchor_cols, args.our_col)
    shared = merged[mask].copy().reset_index(drop=True)

    print(f"\n  Shared subset: {len(shared):,} variants")
    print(f"  Label dist:    {shared[args.label_col].value_counts().to_dict()}")

    if len(shared) < 50:
        print("ERROR: shared subset too small (<50). Check --anchor_cols match column names.")
        print("  Columns containing 'revel' or 'alpha':")
        for c in merged.columns:
            if "revel" in c.lower() or "alpha" in c.lower():
                print(f"    {c}")
        return

    # Save shared subset summary
    save_cols = (["variant_id", "chromosome", "position", "ref_allele", "alt_allele",
                  args.our_col, args.label_col] + anchor_cols)
    save_cols = [c for c in save_cols if c in shared.columns]
    shared[save_cols].to_csv(out_dir / "shared_subset_summary.tsv", sep="\t", index=False)

    labels = shared[args.label_col]

    # ── Evaluate ───────────────────────────────────────────────────────────
    skip = SKIP_COLS | {args.our_col, args.label_col}
    print(f"\n── Our model ─────────────────────────────────────────────────────")
    our_metrics, dbnsfp_metrics = evaluate_all_columns(shared, labels, skip, args.our_col)
    print(f"  AUROC={our_metrics['auroc']:.4f}  PRAUC={our_metrics['prauc']:.4f}  "
          f"n={our_metrics['n_variants']:,}")

    # ── Build results tables ───────────────────────────────────────────────
    metrics_df = build_metrics_df(our_metrics, dbnsfp_metrics)
    metrics_df.to_csv(out_dir / "all_metrics.tsv", sep="\t", index=False)

    summary = build_summary(metrics_df, anchor_cols, args.top_n)
    print("\n── Summary ───────────────────────────────────────────────────────")
    print(summary.to_string(index=False))
    summary.to_csv(out_dir / "summary_comparison.tsv", sep="\t", index=False)

    # ── Plots ──────────────────────────────────────────────────────────────
    print("\n── Generating plots ──────────────────────────────────────────────")
    top_cols = (metrics_df[metrics_df["column"] != "ours"]
                .dropna(subset=["auroc"])
                .nlargest(args.top_n, "auroc")["column"].tolist())
    our_auroc = our_metrics["auroc"]

    plot_roc_curves(shared, top_cols, args.our_col, args.label_col,
                    plots_dir / "roc_curves.png")
    plot_pr_curves(shared, top_cols, args.our_col, args.label_col,
                   plots_dir / "pr_curves.png")
    plot_auroc_barplot(metrics_df, our_auroc, anchor_cols,
                       plots_dir / "auroc_barplot.png")
    plot_metrics_heatmap(metrics_df, plots_dir / "metrics_heatmap.png")
    plot_auroc_scatter(metrics_df, our_auroc, plots_dir / "auroc_scatter.png")

    print(f"\n✓  Done.  Outputs in {out_dir}")
    print(f"   Shared subset:   {len(shared):,} variants")
    print(f"   Our AUROC:       {our_auroc:.4f}")
    for col in anchor_cols:
        row = metrics_df[metrics_df["column"] == col]
        if not row.empty:
            print(f"   {col}: {row['auroc'].values[0]:.4f}")


if __name__ == "__main__":
    main()
