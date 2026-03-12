import glob
import pandas as pd

# Adjust this pattern to match your actual file location
pattern = "data/sequences/ClinVar.260309only.missense.hg38.*.bed.seq12k.tsv"
files = sorted(glob.glob(pattern))

if not files:
    raise FileNotFoundError(f"No files found matching pattern: {pattern}")

print(f"Found {len(files)} files:")
for f in files:
    print(f"  {f}")

# Combine all files, skipping the header for non-first files
dfs = []
for i, filepath in enumerate(files):
    df = pd.read_csv(filepath)  # adjust sep="\t" if TSV
    dfs.append(df)
    print(f"  {filepath}: {len(df)} rows")

combined = pd.concat(dfs, ignore_index=True)
print(f"\nTotal rows combined: {len(combined)}")

output_path = "data/sequences/ClinVar.260309only.seq12k_all.tsv"
combined.to_csv(output_path, index=False)
print(f"Saved to: {output_path}")