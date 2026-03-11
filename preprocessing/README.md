# Preprocessing

This directory contains all scripts for generating the training data from raw ClinVar variant annotations and the GRCh38 reference genome. All scripts are run from the **repo root**.

**Design principle:** This folder contains code only. All data files (inputs and outputs) live in `data/`.

---

## Pipeline Overview

```
data/reference/    +    data/bed/
        │
        ▼
preprocessing/generate_sequences.sh
        │
        ▼
data/sequences/
        │
        ▼
preprocessing/generate_splits.sh
        │
        ▼
data/splits/
```

---

## Scripts

| Script | Description |
|--------|-------------|
| `extract_variant_sequences.py` | Extracts flanking sequences around each variant from the reference genome |
| `generate_sequences.sh` | Batch wrapper — runs extraction for all 4 variant classes and 5 window sizes, outputs to `data/sequences/` |
| `split_data_fixed_chroms.py` | Splits a combined sequence file into train/val by chromosome assignment |
| `generate_splits.sh` | Batch wrapper — runs splitting for all window sizes and both dataset modes, outputs to `data/splits/` |

---

## Input Data

### Reference Genome
- **File:** `data/reference/Homo_sapiens.GRCh38.dna.primary_assembly.fa`
- **Genome build:** GRCh38 (hg38), Ensembl release 104 (May 2021)
- **Source:** [Ensembl release 104](https://ftp.ensembl.org/pub/release-104/fasta/homo_sapiens/dna/)
- Not tracked in git due to file size (~3 GB). Download and place manually:

```bash
wget https://ftp.ensembl.org/pub/release-104/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz
gunzip Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz
mv Homo_sapiens.GRCh38.dna.primary_assembly.fa data/reference/
```

> **Note:** GRCh38.104 refers to Ensembl release 104 built on the GRCh38 assembly. The `.104` is the Ensembl release number — the underlying DNA sequence is standard GRCh38 (hg38).

### ClinVar BED Files
- **Location:** `data/bed/`
- **Tracked in git** (small files)
- Missense variants from ClinVar accessed November 2025 (`251103` = YYMMDD date stamp), filtered by pathogenicity label and aligned to hg38

| File | Label |
|------|-------|
| `ClinVar.251103.missense.hg38.benign.bed` | Benign |
| `ClinVar.251103.missense.hg38.likely_benign.bed` | Likely benign |
| `ClinVar.251103.missense.hg38.likely_pathogenic.bed` | Likely pathogenic |
| `ClinVar.251103.missense.hg38.pathogenic.bed` | Pathogenic |

BED files use **1-based coordinates**.

---

## Output Data

### Sequences (`data/sequences/`)

Running `generate_sequences.sh` produces one `.tsv` file per variant class per window size, named:

```
{bed_filename}.seq{window_size}.tsv
```

The `-l` parameter in `extract_variant_sequences.py` specifies flanking bp on **each side** of the variant. Total sequence length = `(2 × l) + 1`.

| Output suffix | `-l` value | Total window |
|---------------|-----------|--------------|
| `seq6k`       | 2,999     | 5,999 bp     |
| `seq12k`      | 5,999     | 11,999 bp    |
| `seq30k`      | 14,999    | 29,999 bp    |
| `seq60k`      | 29,999    | 59,999 bp    |
| `seq130k`     | 64,999    | 129,999 bp   |

These window sizes correspond to the context lengths of the genomic language models used (Nucleotide Transformer, Caduceus).



### Splits (`data/splits/`)

Running `generate_splits.sh` produces train/val `.tsv` files for each window size and dataset mode, named:

```
ClinVar.251103.missense.hg38.seq{size}.{mode}_{training|validation}.tsv
```

Two dataset modes are supported:

| Mode | Classes included | Label encoding | Window sizes |
|------|-----------------|----------------|--------------|
| `BvsP` | Benign + Pathogenic | benign=0, pathogenic=1 | 6k, 12k, 30k, 60k, 130k |
| `BLBvsPLP` | Benign + Likely benign + Likely pathogenic + Pathogenic | benign/likely_benign=0, pathogenic/likely_pathogenic=1 | 6k, 12k, 30k, 60k, 130k |

Each split file contains the following columns:

```
variant_id  chromosome  position  ref_allele  alt_allele  upstream_flank  downstream_flank  ref_sequence  alt_sequence  label
```

#### Chromosome split

Data was split by chromosome to avoid data leakage between train and validation sets. The chromosome assignments were established from the initial dataset using a random shuffle, then hardcoded in `split_data_fixed_chroms.py` to ensure consistency across all experiments:

- **Train:** chr 1–6, 9–10, 12–14, 16–19, 21–22, MT, X, Y (~80%)
- **Val:** chr 7, 8, 11, 15, 20 (~20%)

---

## Running the Pipeline

**Step 1 — Generate sequences** (requires reference genome in `data/reference/`):

```bash
bash preprocessing/generate_sequences.sh
```

Generates all 20 files (4 classes × 5 window sizes) into `data/sequences/`. Already-existing files are skipped, so it is safe to re-run.

**Step 2 — Generate train/val splits:**

```bash
bash preprocessing/generate_splits.sh
```

Generates 20 split files into `data/splits/` — 10 for `BvsP` (5 window sizes × 2) and 10 for `BLBvsPLP` (5 window sizes × 2).

---

## Notes

- `data/bed/` — tracked in git (small ClinVar BED files, ~few MB)
- `data/reference/` — not tracked in git (reference genome ~3 GB); download instructions above
- `data/sequences/` — not tracked in git (large generated files); regenerate using `generate_sequences.sh`
- `data/splits/` — not tracked in git (generated from sequences); regenerate using `generate_splits.sh`