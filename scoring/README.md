# Scoring

This directory contains the script and model for scoring new variants using our best fine-tuned model.

---

## Contents

```
scoring/
├── score_variants.py       # Main scoring script
└── model/                  # Model weights and config (not tracked in git)
    ├── best_model.pt        # Fine-tuned model weights
    └── model_config.json    # Model architecture config
```

---

## Model

Our best model is **Nucleotide Transformer v2 (500M)** fine-tuned with:

| Setting | Value |
|---------|-------|
| Fine-tuning | LoRA (rank=32) |
| Classifier head | CNN |
| Embedding strategy | full-variant_position |
| Training data | BvsP (benign + pathogenic) |
| Sequence length | 12k bp (5,999 bp flanking each side) |
| Validation AUC | **0.886** |

The model weights (`best_model.pt`) are not tracked in git due to file size. Download link will be provided upon publication.

---

## Input Format

The input TSV must have the following columns (same format as files in `data/splits/`):

```
variant_id  chromosome  position  ref_allele  alt_allele
upstream_flank  downstream_flank  ref_sequence  alt_sequence
```

An optional `label` column (0=benign, 1=pathogenic) can be included — if present, AUC will be computed and logged automatically.

To generate a properly formatted input from a BED file, see [`preprocessing/README.md`](../preprocessing/README.md).

---

## Output Format

The output TSV contains one row per input variant:

```
variant_id  chromosome  position  ref_allele  alt_allele  pathogenicity_score  predicted_label
```

| Column | Description |
|--------|-------------|
| `pathogenicity_score` | Sigmoid probability (0–1), higher = more pathogenic |
| `predicted_label` | Binary call: 0 (benign) or 1 (pathogenic) at default threshold of 0.5 |

---

## Usage

Run from the **repo root**:

```bash
python scoring/score_variants.py \
    --input  your_variants.tsv \
    --model  scoring/model/best_model.pt \
    --output results/predictions/scores.tsv
```

### All arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input`,  `-i` | required | Input TSV file |
| `--model`,  `-m` | required | Path to `best_model.pt` |
| `--output`, `-o` | required | Output TSV file path |
| `--batch_size`, `-b` | 16 | Batch size for inference |
| `--gpu`, `-g` | -1 | GPU id (-1 for CPU) |
| `--threshold`, `-t` | 0.5 | Threshold for `predicted_label` |
| `--k` | 6 | K-mer size for tokenization |

---

## Notes

- The script automatically creates the output directory if it does not exist
- Scoring order matches input order — output rows correspond 1:1 to input rows
- `best_model.pt` is gitignored; add them to `scoring/model/` after downloading