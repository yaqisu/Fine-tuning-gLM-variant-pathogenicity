# Preprocessing

This directory contains scripts and data for extracting genomic sequences around ClinVar missense variants, to be used as input for genomic language model fine-tuning.

---

## Directory Structure

```
preprocessing/
├── extract_variant_sequences.py     # Core script for sequence extraction
├── generate_sequences.sh            # Batch script to generate all sequence files
├── reference/
│   └── Homo_sapiens.GRCh38.dna.primary_assembly.fa   # GRCh38 reference genome (not tracked in git)
└── ClinVar_BED_files/
    ├── ClinVar.251103.missense.hg38.benign.bed
    ├── ClinVar.251103.missense.hg38.likely_benign.bed
    ├── ClinVar.251103.missense.hg38.likely_pathogenic.bed
    ├── ClinVar.251103.missense.hg38.pathogenic.bed
data/
├── *.seq{6k,12k,30k,60k,130k}.txt   # Extracted sequences (generated, not tracked in git)
```

---

## Input Files

### Reference Genome
- **File:** `reference/Homo_sapiens.GRCh38.dna.primary_assembly.fa`
- **Genome build:** GRCh38 (hg38)
- **Source:** [Ensembl](https://ftp.ensembl.org/pub/release-109/fasta/homo_sapiens/dna/)
- Not tracked in git due to file size. Download manually and place at the path above.

### ClinVar BED Files
Located in `ClinVar_BED_files/`. Each file contains missense variants from ClinVar (accessed November 2025, `251103` = date stamp) filtered by pathogenicity label:

| File | Label | Description |
|------|-------|-------------|
| `ClinVar.251103.missense.hg38.benign.bed` | Benign | ClinVar-reviewed benign missense variants |
| `ClinVar.251103.missense.hg38.likely_benign.bed` | Likely benign | ClinVar-reviewed likely benign variants |
| `ClinVar.251103.missense.hg38.likely_pathogenic.bed` | Likely pathogenic | ClinVar-reviewed likely pathogenic variants |
| `ClinVar.251103.missense.hg38.pathogenic.bed` | Pathogenic | ClinVar-reviewed pathogenic variants |

BED files use **1-based coordinates** and are aligned to hg38.

---

## Output Files

Running the batch script produces one sequence file per BED file per flanking window size. Output files are named:

```
{bed_filename}.seq{window_size}.txt
```

### Flanking Window Sizes

The `-l` parameter in `extract_variant_sequences.py` specifies the number of flanking base pairs on **each side** of the variant. Total sequence length = `(2 × l) + 1`.

| Output suffix | `-l` value | Total sequence length |
|---------------|-----------|----------------------|
| `seq6k`       | 2,999     | 5,999 bp             |
| `seq12k`      | 5,999     | 11,999 bp            |
| `seq30k`      | 14,999    | 29,999 bp            |
| `seq60k`      | 29,999    | 59,999 bp            |
| `seq130k`     | 64,999    | 129,999 bp           |

These window sizes correspond to the context lengths supported by the genomic language models used in this project (Nucleotide Transformer, Caduceus).

---

## Generating Sequence Files

Run from the **repo root**:

```bash
bash preprocessing/generate_sequences.sh
```

This will generate all 20 output files (4 BED files × 5 window sizes). Files that already exist are skipped, so it is safe to re-run if a job is interrupted.

You can also run `extract_variant_sequences.py` directly for a single file:

```bash
python preprocessing/extract_variant_sequences.py \
    -b preprocessing/ClinVar_BED_files/ClinVar.251103.missense.hg38.pathogenic.bed \
    -f preprocessing/reference/Homo_sapiens.GRCh38.dna.primary_assembly.fa \
    -l 5999 \
    -o data/my_output.txt
```

---

## Notes

- Sequence files (`.txt`) and the reference genome (`.fa`) are excluded from version control via `.gitignore` due to file size.
- BED files use 1-based coordinates as expected by `extract_variant_sequences.py`.