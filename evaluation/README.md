# Evaluation

## Overview

We evaluate our fine-tuned genomic language models against baseline methods on a held-out set of ClinVar variants. To ensure fair evaluation on truly unseen data, we construct this held-out set by extracting variants that are new to the most recent ClinVar release.

---

## ClinVar.260309only — Held-Out Variant Set

### 1. Constructing the Held-Out Set

We subtract all variants present in the **251103** (November 2024) ClinVar release from the **260309** (March 2026) release, retaining only missense variants that are new to the later release. This ensures that none of the evaluation variants were seen during training.

```bash
bash preprocessing/subtract_new_variants.sh
```

This yields new-only missense variants in hg38 coordinates across four pathogenicity classes:

| Class | 251103 | 260309 | New-only |
|---|---|---|---|
| Benign | 31,521 | 31,373 | 884 |
| Likely benign | 99,227 | 105,087 | 7,852 |
| Likely pathogenic | 32,055 | 32,893 | 2,044 |
| Pathogenic | 23,868 | 24,565 | 1,195 |

Output files:
- `ClinVar.260309only.missense.hg38.benign.bed`
- `ClinVar.260309only.missense.hg38.likely_benign.bed`
- `ClinVar.260309only.missense.hg38.likely_pathogenic.bed`
- `ClinVar.260309only.missense.hg38.pathogenic.bed`

### 2. Extracting Sequences

±6kb sequences around each variant are extracted and concatenated into a single TSV file:

```bash
python evaluation/concat_sequence_files.py
```

Output: `data/sequences/ClinVar.260309only.seq12k_all.tsv`

### 3. Scoring Variants

Variants are scored using our best fine-tuned model:

```bash
python scoring/score_variants.py \
  --input  data/sequences/ClinVar.260309only.seq12k_all.tsv \
  --model  scoring/model/best_model.pt \
  --output results/predictions/ClinVar.260309only.seq12k_all_scores.tsv
```

---
