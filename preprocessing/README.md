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
preprocessing/split_data.py
        │
        ▼
data/splits/
```

---

## Scripts

| Script | Description |
|--------|-------------|
| `extract_variant_sequences.py` | Extracts flanking sequences around each variant from the reference genome |
| `generate_sequences.sh` | Batch wrapper — runs extraction for all 4 variant classes and 5 window sizes |
| `split_data.py` | Splits sequences into train/val sets and writes to `data/splits/` |

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

Running `generate_sequences.sh` produces one file per variant class per window size, named:
```
{bed_filename}.seq{window_size}.txt
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

`split_data.py` produces:
- `train.txt` — training sequences
- `val.txt` — validation sequences
- `split_info.txt` — records random seed, split ratio, and date for reproducibility

---

## Running the Pipeline

**Step 1 — Generate sequences** (requires reference genome in `data/reference/`):
```bash
bash preprocessing/generate_sequences.sh
```
Generates all 20 files (4 classes × 5 window sizes) into `data/sequences/`. Already-existing files are skipped, so it is safe to re-run.

**Step 2 — Generate train/val split:**
```bash
python preprocessing/split_data.py
```

**Or run a single file manually:**
```bash
python preprocessing/extract_variant_sequences.py \
    -b data/bed/ClinVar.251103.missense.hg38.pathogenic.bed \
    -f data/reference/Homo_sapiens.GRCh38.dna.primary_assembly.fa \
    -l 5999 \
    -o data/sequences/ClinVar.251103.missense.hg38.pathogenic.bed.seq12k.txt
```

---

## Notes

- `data/sequences/` and `data/splits/` are excluded from git via `.gitignore` — download from Google Drive instead of regenerating if possible
- `data/reference/` is also excluded from git due to file size
- `data/bed/` is tracked in git since BED files are small