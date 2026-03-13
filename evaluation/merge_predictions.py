#!/usr/bin/env python3
"""
Step 3 — Merge our model predictions with dbNSFP annotations.

Reports missing-variant statistics and per-column dbNSFP coverage before
writing the final merged TSV used by all downstream evaluation scripts.

Usage:
    python evaluation/03_merge_predictions.py \
        --ours    results/predictions/ClinVar.260309only.seq12k.pb/ours.tsv \
        --dbnsfp  results/predictions/ClinVar.260309only.seq12k.pb/dbnsfp.tsv \
        --outdir  results/predictions/ClinVar.260309only.seq12k.pb
"""

import argparse
from pathlib import Path

import pandas as pd


KEY_COLS = ["variant_id", "chromosome", "position", "ref_allele", "alt_allele"]


def parse_args():
    p = argparse.ArgumentParser(description="Merge model predictions with dbNSFP.")
    p.add_argument("--ours",   required=True, type=Path,
                   help="Our model predictions TSV. "
                        "Must include: variant_id, true_label, pathogenicity_score")
    p.add_argument("--dbnsfp", required=True, type=Path,
                   help="dbNSFP annotations TSV (output of 02_annotate_dbnsfp.py)")
    p.add_argument("--outdir", required=True, type=Path,
                   help="Output directory")
    p.add_argument("--top_coverage", default=30, type=int,
                   help="Print top-N best-covered dbNSFP columns (default: 30)")
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = args.outdir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load ───────────────────────────────────────────────────────────────
    print(f"Loading {args.ours}")
    ours = pd.read_csv(args.ours, sep="\t")
    print(f"  {len(ours):,} variants,  columns: {list(ours.columns)}")

    print(f"\nLoading {args.dbnsfp}")
    dbnsfp = pd.read_csv(args.dbnsfp, sep="\t", low_memory=False)
    print(f"  {len(dbnsfp):,} variants,  {dbnsfp.shape[1]} columns")

    # ── Missing variant analysis ───────────────────────────────────────────
    print("\n── Missing variant analysis ──────────────────────────────────────")
    ours_ids   = set(ours["variant_id"])
    dbnsfp_ids = set(dbnsfp["variant_id"])

    in_both        = ours_ids & dbnsfp_ids
    only_in_ours   = ours_ids - dbnsfp_ids
    only_in_dbnsfp = dbnsfp_ids - ours_ids

    print(f"  In ours:                    {len(ours_ids):>6,}")
    print(f"  In dbnsfp:                  {len(dbnsfp_ids):>6,}")
    print(f"  In both:                    {len(in_both):>6,}")
    print(f"  Only in ours (no dbnsfp):   {len(only_in_ours):>6,}")
    print(f"  Only in dbnsfp (not ours):  {len(only_in_dbnsfp):>6,}")

    # Rows where every dbNSFP score is '.'
    score_cols          = [c for c in dbnsfp.columns if c not in KEY_COLS + ["label"]]
    dbnsfp["_n_ann"]    = (dbnsfp[score_cols] != ".").sum(axis=1)
    fully_missing       = (dbnsfp["_n_ann"] == 0).sum()
    print(f"\n  dbNSFP rows with zero annotations: {fully_missing:,}")
    print(f"  dbNSFP rows with ≥1 score:          {len(dbnsfp) - fully_missing:,}")

    if only_in_ours:
        label_col = "true_label" if "true_label" in ours.columns else ours.columns[-1]
        missing_df = ours[ours["variant_id"].isin(only_in_ours)][KEY_COLS + [label_col]]
        missing_path = out_dir / "missing_in_dbnsfp.tsv"
        missing_df.to_csv(missing_path, sep="\t", index=False)
        print(f"\n  Missing variants saved to {missing_path}")

    # ── Merge ──────────────────────────────────────────────────────────────
    print("\n── Merging ───────────────────────────────────────────────────────")
    dbnsfp_for_merge = dbnsfp.drop(columns=["label", "_n_ann"], errors="ignore")

    merged = ours.merge(
        dbnsfp_for_merge,
        on=KEY_COLS,
        how="left",
        suffixes=("", "_dbnsfp"),
    )
    print(f"  Merged shape: {merged.shape[0]:,} rows × {merged.shape[1]} columns")

    # ── Coverage summary ───────────────────────────────────────────────────
    print("\n── dbNSFP column coverage ────────────────────────────────────────")
    extra_cols = [c for c in merged.columns if c not in list(ours.columns) and c not in KEY_COLS]

    cov_rows = []
    for col in extra_cols:
        n_total   = len(merged)
        n_missing = merged[col].isna().sum() + (merged[col] == ".").sum()
        n_present = n_total - n_missing
        cov_rows.append({
            "column":    col,
            "n_present": n_present,
            "n_missing": n_missing,
            "coverage":  round(n_present / n_total, 4),
        })

    cov_df = pd.DataFrame(cov_rows).sort_values("coverage", ascending=False)
    cov_path = out_dir / "dbnsfp_column_coverage.tsv"
    cov_df.to_csv(cov_path, sep="\t", index=False)
    print(f"  Coverage table saved to {cov_path}")
    print(f"\n  Top {args.top_coverage} best-covered columns:")
    print(cov_df.head(args.top_coverage).to_string(index=False))

    # ── Save ───────────────────────────────────────────────────────────────
    merged_path = out_dir / "merged.tsv"
    merged.to_csv(merged_path, sep="\t", index=False)
    print(f"\n  Merged table saved to {merged_path}")
    print(f"  Shape: {merged.shape[0]:,} rows × {merged.shape[1]} columns")


if __name__ == "__main__":
    main()
