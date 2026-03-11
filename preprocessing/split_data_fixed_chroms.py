import pandas as pd
import argparse

# Chromosome split established from initial data split using split_data.py
# with a random chromosome shuffle. To ensure all subsequent datasets use
# the exact same train/validation split, the chromosome assignments are
# hardcoded here for reproducibility across experiments.
TRAIN_CHROMS = {'1','2','3','4','5','6','9','10','12','13','14',
                '16','17','18','19','21','22','MT','X','Y'}
VAL_CHROMS   = {'7','8','11','15','20'}

def load_and_label(path, label):
    df = pd.read_csv(path, sep='\t')
    df['label'] = int(label)   # ensure integer label
    return df

def split_and_save(df, train_out, val_out):
    df['chromosome'] = df['chromosome'].astype(str)

    unknown = set(df['chromosome'].unique()) - (TRAIN_CHROMS | VAL_CHROMS)
    if unknown:
        print(f"  WARNING: Unknown chromosomes will be excluded: {sorted(unknown)}")

    train_df = df[df['chromosome'].isin(TRAIN_CHROMS)]
    val_df   = df[df['chromosome'].isin(VAL_CHROMS)]

    print(f"  Total:  {len(df):>6} variants")
    print(f"  Train:  {len(train_df):>6} ({len(train_df)/len(df)*100:.1f}%) | label counts: {dict(train_df['label'].value_counts().sort_index())}")
    print(f"  Val:    {len(val_df):>6} ({len(val_df)/len(df)*100:.1f}%) | label counts: {dict(val_df['label'].value_counts().sort_index())}")

    train_df.to_csv(train_out, sep='\t', index=False)
    val_df.to_csv(val_out,     sep='\t', index=False)
    print(f"  Saved:  {train_out}")
    print(f"  Saved:  {val_out}")

def main():
    parser = argparse.ArgumentParser(
        description='Split ClinVar variant sequences into train/val using fixed chromosome assignments.',
        epilog="""
Chromosome assignments (established from initial random shuffle, then hardcoded):
  Train: chr 1-6, 9-10, 12-14, 16-19, 21-22, MT, X, Y
  Val:   chr 7, 8, 11, 15, 20

Two dataset modes:
  --mode BvsP     : benign (0) + pathogenic (1) only
  --mode BLBvsPLP : benign (0) + likely_benign (0) + likely_pathogenic (1) + pathogenic (1)
        """
    )
    parser.add_argument('--benign',            '-b',  required=True, help='Benign sequence file')
    parser.add_argument('--pathogenic',        '-p',  required=True, help='Pathogenic sequence file')
    parser.add_argument('--likely_benign',     '-lb', default=None,  help='Likely benign sequence file (BLBvsPLP mode only)')
    parser.add_argument('--likely_pathogenic', '-lp', default=None,  help='Likely pathogenic sequence file (BLBvsPLP mode only)')
    parser.add_argument('--mode',              '-m',  default='BvsP', choices=['BvsP', 'BLBvsPLP'],
                        help='BvsP: benign+pathogenic only; BLBvsPLP: include likely classes (default: BvsP)')
    parser.add_argument('--train',             '-t',  required=True, help='Output train file path')
    parser.add_argument('--val',               '-v',  required=True, help='Output val file path')

    args = parser.parse_args()

    # Load and label
    dfs = [
        load_and_label(args.benign,     0),
        load_and_label(args.pathogenic, 1),
    ]

    if args.mode == 'BLBvsPLP':
        if not args.likely_benign or not args.likely_pathogenic:
            parser.error('--mode BLBvsPLP requires --likely_benign and --likely_pathogenic')
        dfs += [
            load_and_label(args.likely_benign,     0),
            load_and_label(args.likely_pathogenic, 1),
        ]

    df = pd.concat(dfs, ignore_index=True)
    split_and_save(df, args.train, args.val)

if __name__ == '__main__':
    main()