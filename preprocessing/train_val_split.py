import pandas as pd
import numpy as np
import argparse

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Split variant data by chromosome into train/validation sets')
    parser.add_argument('--input', '-i', required=True, help='Input file path')
    parser.add_argument('--train', '-t', default='train.txt', help='Output train file path (default: train.txt)')
    parser.add_argument('--val', '-v', default='validation.txt', help='Output validation file path (default: validation.txt)')
    parser.add_argument('--seed', '-s', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--split', type=float, default=0.8, help='Train split ratio (default: 0.8)')
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    np.random.seed(args.seed)
    
    # Read the data
    df = pd.read_csv(args.input, sep="\t")
    
    # Get unique chromosomes
    chromosomes = df['chromosome'].unique()
    print(f"Total chromosomes: {len(chromosomes)}")
    print(f"Chromosomes: {sorted(chromosomes)}")
    
    # Shuffle chromosomes
    np.random.shuffle(chromosomes)
    
    # Split chromosomes into train and validation
    n_train = int(len(chromosomes) * args.split)
    train_chroms = chromosomes[:n_train]
    val_chroms = chromosomes[n_train:]
    
    print(f"\nTrain chromosomes ({len(train_chroms)}): {sorted(train_chroms)}")
    print(f"Validation chromosomes ({len(val_chroms)}): {sorted(val_chroms)}")
    
    # Split data based on chromosome assignment
    train_df = df[df['chromosome'].isin(train_chroms)]
    val_df = df[df['chromosome'].isin(val_chroms)]
    
    print(f"\nTrain samples: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Validation samples: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    
    # Save splits
    train_df.to_csv(args.train, sep="\t", index=False)
    val_df.to_csv(args.val, sep="\t", index=False)
    
    print(f"\nSaved {args.train} and {args.val}")

if __name__ == "__main__":
    main()
