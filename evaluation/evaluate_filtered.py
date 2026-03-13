#!/usr/bin/env python3
"""
Step 5 — Evaluate methods on filtered or stratified variant subsets.

Two modes:
  --mode filter    Keep variants passing a condition (e.g. AF < 1e-3).
                   Produces one eval output directory.

  --mode stratify  Bin variants into ranges and evaluate each bin separately.
                   Produces one subdirectory per stratum + a comparison table.

Supported filter/stratify columns:
  AF:           gnomAD4.1_joint_AF (default), or any AF column in merged.tsv
  Conservation: phyloP100way_vertebrate, GERP++_RS, phastCons100way_vertebrate, etc.
                (see evaluation/core/filters.py for full list)

Usage:
    # ── Filter mode ─────────────────────────────────────────────────────
    # Rare variants (AF < 1e-6, missing included)
    python evaluation/05_evaluate_filtered.py \
        --merged  results/predictions/ClinVar.260309only.seq12k.pb/merged.tsv \
        --outdir  results/predictions/ClinVar.260309only.seq12k.pb/eval_rare \
        --mode    filter \
        --col     gnomAD4.1_joint_AF \
        --threshold 1e-6

    # Highly conserved sites (phyloP >= 3)
    python evaluation/05_evaluate_filtered.py \
        --merged  results/predictions/ClinVar.260309only.seq12k.pb/merged.tsv \
        --outdir  results/predictions/ClinVar.260309only.seq12k.pb/eval_conserved \
        --mode    filter \
        --col     phyloP100way_vertebrate \
        --threshold 3.0 \
        --direction above

    # Multiple AF columns — rare in ALL of them
    python evaluation/05_evaluate_filtered.py \
        --merged  results/predictions/ClinVar.260309only.seq12k.pb/merged.tsv \
        --outdir  results/predictions/ClinVar.260309only.seq12k.pb/eval_rare_multi \
        --mode    filter \
        --col     gnomAD4.1_joint_AF,gnomAD2.1.1_exomes_controls_AF \
        --threshold 1e-3

    # ── Stratify mode ────────────────────────────────────────────────────
    # Stratify by AF using built-in bins
    python evaluation/05_evaluate_filtered.py \
        --merged  results/predictions/ClinVar.260309only.seq12k.pb/merged.tsv \
        --outdir  results/predictions/ClinVar.260309only.seq12k.pb/strat_af \
        --mode    stratify \
        --col     gnomAD4.1_joint_AF \
        --strata  builtin_af

    # Stratify by GERP++ RS using built-in bins
    python evaluation/05_evaluate_filtered.py \
        --merged  results/predictions/ClinVar.260309only.seq12k.pb/merged.tsv \
        --outdir  results/predictions/ClinVar.260309only.seq12k.pb/strat_gerp \
        --mode    stratify \
        --col     GERP++_RS \
        --strata  builtin_gerp

    # Stratify by custom bins:  --strata 'None:1e-6,1e-6:1e-4,1e-4:None'
    python evaluation/05_evaluate_filtered.py \
        --merged  results/predictions/ClinVar.260309only.seq12k.pb/merged.tsv \
        --outdir  results/predictions/ClinVar.260309only.seq12k.pb/strat_custom \
        --mode    stratify \
        --col     gnomAD4.1_joint_AF \
        --strata  'None:1e-6,1e-6:1e-3,1e-3:None'
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from core import (
    evaluate_all_columns, build_metrics_df, build_summary,
    apply_anchor_filter, apply_af_filter, apply_conservation_filter,
    stratify_by_column, parse_custom_strata,
    AF_STRATA_DEFAULT, GERP_STRATA_DEFAULT, PHYLOP_STRATA_DEFAULT,
    CONSERVATION_COLS,
    plot_roc_curves, plot_pr_curves, plot_auroc_barplot,
    plot_metrics_heatmap, plot_af_distribution,
    plot_comparison_across_strata,
)

SKIP_COLS = {"variant_id", "chromosome", "position", "ref_allele", "alt_allele",
             "predicted_label"}

BUILTIN_STRATA = {
    "builtin_af":     AF_STRATA_DEFAULT,
    "builtin_gerp":   GERP_STRATA_DEFAULT,
    "builtin_phylop": PHYLOP_STRATA_DEFAULT,
}


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate with AF/conservation filtering or stratification.")
    p.add_argument("--merged",       required=True,  type=Path)
    p.add_argument("--outdir",       required=True,  type=Path)
    p.add_argument("--mode",         required=True,  choices=["filter", "stratify"],
                   help="'filter': keep passing variants only. "
                        "'stratify': evaluate each bin separately.")
    p.add_argument("--col",          required=True,
                   help="Column(s) to filter/stratify on. Comma-separated for multi-AF filter. "
                        "For conservation use a single column name, e.g. GERP++_RS")
    p.add_argument("--threshold",    default=1e-3, type=float,
                   help="[filter mode] Numeric threshold (default: 1e-3)")
    p.add_argument("--direction",    default="below", choices=["below", "above"],
                   help="[filter mode] 'below': col < threshold (AF filter, default). "
                        "'above': col >= threshold (conservation filter).")
    p.add_argument("--include_missing", default=True,
                   action=argparse.BooleanOptionalAction,
                   help="[AF filter mode] Include variants absent from gnomAD (default: True)")
    p.add_argument("--strata",       default="builtin_af",
                   help="[stratify mode] Built-in preset (builtin_af, builtin_gerp, "
                        "builtin_phylop) or custom spec 'lo:hi,lo:hi,...' "
                        "where lo/hi are floats or None. (default: builtin_af)")
    p.add_argument("--our_col",      default="pathogenicity_score")
    p.add_argument("--label_col",    default="true_label")
    p.add_argument("--anchor_cols",  default="REVEL_score,AlphaMissense_score")
    p.add_argument("--top_n",        default=20, type=int)
    p.add_argument("--subset",       default=None, type=Path,
                   help="Optional variant_id TSV to pre-filter merged")
    return p.parse_args()


def run_evaluation(shared: pd.DataFrame, labels: pd.Series,
                   skip: set, our_col: str, anchor_cols: list,
                   out_dir: Path, top_n: int, subtitle: str = "") -> dict:
    """Run full evaluation on a shared subset; return our_metrics dict."""
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(exist_ok=True)

    if len(shared) < 20 or labels.nunique() < 2:
        print(f"  SKIP: too few variants ({len(shared)}) or only one class.")
        return {}

    our_metrics, dbnsfp_metrics = evaluate_all_columns(shared, labels, skip, our_col)
    print(f"  AUROC={our_metrics['auroc']:.4f}  n={our_metrics['n_variants']:,}")

    metrics_df = build_metrics_df(our_metrics, dbnsfp_metrics)
    metrics_df.to_csv(out_dir / "all_metrics.tsv", sep="\t", index=False)

    summary = build_summary(metrics_df, anchor_cols, top_n)
    summary.to_csv(out_dir / "summary_comparison.tsv", sep="\t", index=False)
    print(summary.to_string(index=False))

    top_cols  = (metrics_df[metrics_df["column"] != "ours"]
                 .dropna(subset=["auroc"])
                 .nlargest(top_n, "auroc")["column"].tolist())
    our_auroc = our_metrics["auroc"]

    plot_roc_curves(shared, top_cols, our_col, labels.name,
                    plots_dir / "roc_curves.png", subtitle=subtitle)
    plot_pr_curves(shared, top_cols, our_col, labels.name,
                   plots_dir / "pr_curves.png", subtitle=subtitle)
    plot_auroc_barplot(metrics_df, our_auroc, anchor_cols,
                       plots_dir / "auroc_barplot.png")
    plot_metrics_heatmap(metrics_df, plots_dir / "metrics_heatmap.png")

    return our_metrics


def main():
    args        = parse_args()
    out_dir     = args.outdir
    anchor_cols = [c.strip() for c in args.anchor_cols.split(",")]
    filter_cols = [c.strip() for c in args.col.split(",")]
    skip        = SKIP_COLS | {args.our_col, args.label_col}

    # ── Load ───────────────────────────────────────────────────────────────
    print(f"Loading {args.merged}")
    merged = pd.read_csv(args.merged, sep="\t", low_memory=False)
    print(f"  {len(merged):,} variants × {merged.shape[1]} columns")

    # Optional variant subset
    if args.subset:
        sub  = pd.read_csv(args.subset, sep="\t", low_memory=False)
        ids  = set(sub["variant_id"].dropna().astype(str))
        merged = merged[merged["variant_id"].astype(str).isin(ids)].copy().reset_index(drop=True)
        print(f"  After subset filter: {len(merged):,} variants")

    # ── Anchor filter (shared evaluation subset) ───────────────────────────
    print(f"\n── Anchor filter: {anchor_cols} ──────────────────────────────────")
    anchor_mask = apply_anchor_filter(merged, anchor_cols, args.our_col)
    shared_base = merged[anchor_mask].copy().reset_index(drop=True)
    print(f"  Shared base: {len(shared_base):,} variants")

    # ══════════════════════════════════════════════════════════════════════
    # FILTER MODE
    # ══════════════════════════════════════════════════════════════════════
    if args.mode == "filter":
        print(f"\n── Filter mode ───────────────────────────────────────────────────")
        is_conservation = filter_cols[0] in CONSERVATION_COLS or args.direction == "above"

        if is_conservation:
            assert len(filter_cols) == 1, "Conservation filter supports only one column."
            col  = filter_cols[0]
            mask = apply_conservation_filter(shared_base, col, args.threshold, args.direction)
            subtitle = f"{col} {args.direction} {args.threshold}  n={mask.sum():,}"
        else:
            print(f"  AF threshold: {args.threshold}  include_missing={args.include_missing}")
            mask     = apply_af_filter(shared_base, filter_cols, args.threshold,
                                       include_missing=args.include_missing)
            subtitle = (f"AF < {args.threshold} ({', '.join(filter_cols)})  "
                        f"n={mask.sum():,}")

        filtered = shared_base[mask].copy().reset_index(drop=True)
        labels   = filtered[args.label_col]
        labels.name = args.label_col

        print(f"\n  Filtered subset: {len(filtered):,} variants")
        print(f"  Label dist: {labels.value_counts().to_dict()}")

        # Diagnostic: AF distribution
        af_out = out_dir / "plots"
        af_out.mkdir(parents=True, exist_ok=True)
        if not is_conservation:
            plot_af_distribution(filtered, filter_cols, af_out / "af_distribution.png")

        # Save subset
        out_dir.mkdir(parents=True, exist_ok=True)
        save_cols = (["variant_id", "chromosome", "position", "ref_allele", "alt_allele",
                      args.our_col, args.label_col] + filter_cols + anchor_cols)
        save_cols = [c for c in save_cols if c in filtered.columns]
        filtered[save_cols].to_csv(out_dir / "filtered_subset_summary.tsv",
                                   sep="\t", index=False)

        run_evaluation(filtered, labels, skip, args.our_col, anchor_cols,
                       out_dir, args.top_n, subtitle=subtitle)

        print(f"\n✓  Done.  Outputs in {out_dir}")

    # ══════════════════════════════════════════════════════════════════════
    # STRATIFY MODE
    # ══════════════════════════════════════════════════════════════════════
    elif args.mode == "stratify":
        assert len(filter_cols) == 1, "Stratify supports only one column."
        col = filter_cols[0]

        # Resolve strata definition
        if args.strata in BUILTIN_STRATA:
            strata = BUILTIN_STRATA[args.strata]
        else:
            strata = parse_custom_strata(args.strata)

        print(f"\n── Stratify mode  —  column: {col} ──────────────────────────────")
        print(f"  Strata: {[s[0] for s in strata]}")

        stratum_dfs = stratify_by_column(shared_base, col, strata)

        # Evaluate each stratum
        all_summaries = []
        for stratum_name, sub_df in stratum_dfs.items():
            print(f"\n──── Stratum: '{stratum_name}'  ({len(sub_df):,} variants) ────")
            if len(sub_df) < 20:
                print(f"  SKIP: too few variants.")
                continue

            labels     = sub_df[args.label_col]
            labels.name = args.label_col
            stratum_dir = out_dir / stratum_name.replace(" ", "_").replace("/", "_")

            our_m = run_evaluation(sub_df, labels, skip, args.our_col, anchor_cols,
                                   stratum_dir, args.top_n,
                                   subtitle=f"{col} = '{stratum_name}' | n={len(sub_df):,}")

            if our_m:
                # Collect top-N + anchor summary for cross-stratum comparison
                metrics_path = stratum_dir / "all_metrics.tsv"
                if metrics_path.exists():
                    mdf = pd.read_csv(metrics_path, sep="\t")
                    mdf["stratum"] = stratum_name
                    all_summaries.append(mdf)

        # ── Cross-stratum comparison ────────────────────────────────────
        if all_summaries:
            print("\n── Cross-stratum comparison ──────────────────────────────────")
            comparison = pd.concat(all_summaries, ignore_index=True)
            comp_path  = out_dir / "stratification_comparison.tsv"
            comparison.to_csv(comp_path, sep="\t", index=False)
            print(f"  Saved comparison table to {comp_path}")

            # Pivot: strata × top methods
            top_methods = (comparison[comparison["column"] != "ours"]
                           .dropna(subset=["auroc"])
                           .groupby("column")["auroc"].mean()
                           .nlargest(args.top_n).index.tolist())
            focus = comparison[comparison["column"].isin(["ours"] + top_methods)]

            for metric in ["auroc", "prauc"]:
                plot_path = out_dir / f"stratification_{metric}.png"
                plot_comparison_across_strata(focus, plot_path, metric=metric)

        print(f"\n✓  Done.  Outputs in {out_dir}")


if __name__ == "__main__":
    main()
