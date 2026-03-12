#!/usr/bin/env python3
"""
Split a combined labeled sequence TSV into train/val sets using fixed
chromosome assignments. The input TSV must already contain a 'label' column
(produced by extract_variant_sequences.py --label).
"""

import pandas as pd
import argparse

# Chromosome split established from initial data split using split_data.py
# with a random chromosome shuffle. Hardcoded for reproducibility across
# all experiments.
TRAIN_CHROMS = {'1','2','3','4','5','6','9','10','12','13','14',
                '16','17','18','19','21','22','MT','X','Y'}
VAL_CHROMS   = {'7','8','11','15','20'}


def split_and_save(df, train_out, val_out):
    df['chromosome'] = df['chromosome'].astype(str)

    unknown = set(df['chromosome'].unique()) - (TRAIN_CHROMS | VAL_CHROMS)
    if unknown:
        print(f"  WARNING: Unknown chromosomes will be excluded: {sorted(unknown)}")

    train_df = df[df['chromosome'].isin(TRAIN_CHROMS)]
    val_df   = df[df['chromosome'].isin(VAL_CHROMS)]

    print(f"  Total:  {len(df):>6} variants")
    print(f"  Train:  {len(train_df):>6} ({len(train_df)/len(df)*100:.1f}%)"
          f" | label counts: {dict(train_df['label'].value_counts().sort_index())}")
    print(f"  Val:    {len(val_df):>6} ({len(val_df)/len(df)*100:.1f}%)"
          f" | label counts: {dict(val_df['label'].value_counts().sort_index())}")

    train_df.to_csv(train_out, sep='\t', index=False)
    val_df.to_csv(val_out,     sep='\t', index=False)
    print(f"  Saved:  {train_out}")
    print(f"  Saved:  {val_out}")


def main():
    parser = argparse.ArgumentParser(
        description='Split a labeled sequence TSV into train/val using fixed chromosome assignments.',
        epilog="""
Chromosome assignments (hardcoded for reproducibility):
  Train: chr 1-6, 9-10, 12-14, 16-19, 21-22, MT, X, Y  (~80%)
  Val:   chr 7, 8, 11, 15, 20                            (~20%)

The input TSV must already contain a 'label' column.
To combine multiple classes before splitting, concatenate the TSVs first:
  cat benign.tsv <(tail -n +2 pathogenic.tsv) > combined.tsv
        """
    )
    parser.add_argument('--input',  '-i', required=True, help='Combined labeled sequence TSV')
    parser.add_argument('--train',  '-t', required=True, help='Output train TSV')
    parser.add_argument('--val',    '-v', required=True, help='Output val TSV')

    args = parser.parse_args()

    df = pd.read_csv(args.input, sep='\t')

    if 'label' not in df.columns:
        raise ValueError("Input TSV must contain a 'label' column. "
                         "Generate sequences with --label flag first.")

    split_and_save(df, args.train, args.val)


if __name__ == '__main__':
    main()
