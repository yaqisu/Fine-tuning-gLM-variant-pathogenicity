#!/usr/bin/env python3
"""
Step 1 — Concatenate all per-class sequence files into a single dataset TSV.

Finds all files matching the pattern (all four ClinVar classes) and concatenates
them into one TSV named {dataset}.seq12k.tsv for downstream scoring.

Usage:
    python evaluation/01_concat_sequences.py \
        --dataset ClinVar.260309only \
        --datadir data/sequences
    # Output: data/sequences/ClinVar.260309only.seq12k.tsv

    # Custom glob pattern
    python evaluation/01_concat_sequences.py \
        --dataset ClinVar.260309only \
        --datadir data/sequences \
        --pattern "data/sequences/ClinVar.260309only.missense.hg38.*.bed.seq12k.tsv"
"""

import argparse
import glob
import re
from pathlib import Path

import pandas as pd


DEFAULT_PATTERN = "{datadir}/{dataset}.missense.hg38.*.bed.seq12k.tsv"


def parse_args():
    p = argparse.ArgumentParser(description="Concatenate per-class sequence files.")
    p.add_argument("--dataset",  required=True,
                   help="Dataset name, e.g. ClinVar.260309only")
    p.add_argument("--datadir",  default="data/sequences",
                   help="Directory containing the per-class TSV files (default: data/sequences)")
    p.add_argument("--pattern",  default=None,
                   help="Override glob pattern (default uses --datadir and --dataset)")
    p.add_argument("--outdir",   default=None,
                   help="Output directory (default: same as --datadir)")
    p.add_argument("--sep",      default=None,
                   help="Column separator in input files. "
                        "Auto-detected from extension if not specified.")
    return p.parse_args()


def infer_source_label(filepath: Path, pattern: str) -> str:
    pat_re = re.escape(pattern).replace(r"\*", r"([^/]+)")
    match  = re.search(pat_re, str(filepath))
    return match.group(1) if match else filepath.stem


def main():
    args   = parse_args()
    outdir = Path(args.outdir) if args.outdir else Path(args.datadir)
    outdir.mkdir(parents=True, exist_ok=True)

    pattern = args.pattern or DEFAULT_PATTERN.format(
        datadir=args.datadir, dataset=args.dataset)

    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found matching: {pattern}")

    print(f"Dataset:  {args.dataset}")
    print(f"Pattern:  {pattern}")
    print(f"Found {len(files)} file(s):")
    for f in files:
        print(f"  {f}")

    dfs = []
    for fpath in files:
        src_label = infer_source_label(Path(fpath), pattern)
        sep = args.sep or ("\t" if fpath.endswith(".tsv") else ",")

        print(f"\n  Loading '{src_label}': {fpath}")
        df = pd.read_csv(fpath, sep=sep, low_memory=False)
        print(f"    {len(df):,} rows  x  {df.shape[1]} columns")
        dfs.append(df)

    if not dfs:
        raise RuntimeError("No files loaded.")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nCombined: {len(combined):,} rows  x  {combined.shape[1]} columns")
    if "label" in combined.columns:
        print(f"Label distribution: {combined['label'].value_counts().to_dict()}")

    out_path = outdir / f"{args.dataset}.seq12k.tsv"
    combined.to_csv(out_path, sep="\t", index=False)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()