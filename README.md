# Fine-tuning Genomic Language Models for Variant Pathogenicity Prediction

This repository contains the code for *“Fine-tuning genomic language models for variant pathogenicity prediction”* (Su\*, Lin\*, et al.; \*co-first authors; in preparation).


We fine-tune genomic language models (Nucleotide Transformer v2, Caduceus) on ClinVar missense variants to predict pathogenicity. Our best model achieves **0.886 AUC** using NT2 with LoRA (rank=32) and a CNN classifier head.

---

## Quick Start — Score Your Variants

If you just want to score variants using our best pre-trained model:

**1. Download the model weights** (link TBD upon publication) and place at `scoring/model/best_model.pt`

**2. Prepare your input** — a TSV file with the same format as our split files (see [`preprocessing/README.md`](preprocessing/README.md) for how to generate this from a BED file). Required columns:
```
variant_id  chromosome  position  ref_allele  alt_allele
upstream_flank  downstream_flank  ref_sequence  alt_sequence
```

**3. Run scoring:**
```bash
python scoring/score_variants.py \
    --input  your_variants.tsv \
    --model  scoring/model/best_model.pt \
    --output results/predictions/scores.tsv \
```

**Output** — a TSV file with one row per variant:
```
variant_id  chromosome  position  ref_allele  alt_allele  pathogenicity_score  predicted_label
```
- `pathogenicity_score`: probability of pathogenicity (0–1), higher = more pathogenic
- `predicted_label`: binary call at 0.5 threshold (0=benign, 1=pathogenic)

If your input has a `label` column, AUC will also be computed and logged automatically.

---

## Repository Structure

```
Fine-tuning-gLM-variant-pathogenicity/
│
├── data/                               # All data files (mostly gitignored)
│   ├── bed/                            # ClinVar BED files (tracked in git)
│   ├── reference/                      # GRCh38 reference genome (gitignored, ~3GB)
│   ├── sequences/                      # Extracted sequences (gitignored)
│   └── splits/                         # Train/val splits (gitignored)
│
├── preprocessing/                      # Scripts to generate data/ from raw inputs
│   ├── extract_variant_sequences.py
│   ├── generate_sequences.sh
│   ├── split_data_fixed_chroms.py
│   ├── generate_splits.sh
│   └── README.md
│
├── training/                           # Model training scripts
│   ├── NT2_BLBvsPLP.py                 
│   ├── NT2_phase1_and_unfreezeAll.py
│   ├── NT2_phase2_and_phase3.py
│   ├── NT1_phase1.py
│   ├── caduceus_phase1.py
│   └── runs.sh                         # Exact commands used in the paper
│
├── scoring/                            # Score new variants using trained model
│   ├── score_variants.py
│   └── model/                          # Best model weights (gitignored)
│       ├── best_model.pt
│       └── model_config.json
│
├── evaluation/                         # Compute metrics from predictions
│   └── README.md
│
└── results/                            # All outputs from training and analysis
    ├── NT2_seq12k_BLBvsPLP_lr3e-5/     # Training outputs (gitignored)
    ├── figures/                         # Final paper figures
    ├── figures.ipynb                    # Figure generation notebook
    └── results.tsv                      # Combined results table
```

---

## Reproducing Paper Results

Follow these steps in order to reproduce our results from scratch:

### 1. Data
Download the reference genome and place ClinVar BED files (see [`preprocessing/README.md`](preprocessing/README.md)):
```bash
# BED files are already in data/bed/ (tracked in git)
# Download reference genome:
wget https://ftp.ensembl.org/pub/release-104/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz
gunzip Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz
mv Homo_sapiens.GRCh38.dna.primary_assembly.fa data/reference/
```

### 2. Preprocessing
Generate sequence files and train/val splits:
```bash
bash preprocessing/generate_sequences.sh
bash preprocessing/generate_splits.sh
```

### 3. Training

Training scripts are organized by model and experimental phase. All scripts are run from the repo root and read data from `data/splits/`.

| Script | Model | Description |
|--------|-------|-------------|
| `training/NT2_BLBvsPLP.py` | NT2 | LoRA (rank=32) + CNN on BLBvsPLP data |
| `training/NT2_phase1_and_unfreezeAll.py` | NT2 | Phase 1: frozen backbone; full fine-tuning (unfreeze all) |
| `training/NT2_phase2_and_phase3.py` | NT2 | Phase 2: LoRA fine-tuning; Phase 3: learning rate |
| `training/NT1_phase1.py` | NT1 | Phase 1: frozen backbone experiments |
| `training/caduceus_phase1.py` | Caduceus | Phase 1: frozen backbone experiments |

To reproduce the exact experiments from the paper, run the commands used in our paper:
```bash
bash training/runs.sh
```

### 4. Results

Training outputs are saved to `results/` (one subdirectory per experiment). The combined results table used for figure generation is at `results/results.tsv`.

Generate all paper figures using the notebook:
```bash
jupyter nbconvert --to notebook --execute results/figures.ipynb
```

Generated figures are saved to `results/figures/`:

| Figure | Description |
|--------|-------------|
| `fig2_classifier_embedding_comparison` | Classifier and embedding strategy comparison |
| `fig3_lora_rank_comparison` | LoRA rank ablation |
| `fig4_learning_rate_comparison` | Learning rate sweep |
| `fig5_lora_vs_full_finetuning` | LoRA vs full fine-tuning comparison |
| `fig6_model_performance_summary` | Summary of all model results |

### 5. Scoring
Score new variants using the best trained model:
```bash
python scoring/score_variants.py \
    --input  data/splits/ClinVar.251103.missense.hg38.seq12k.BvsP_validation.tsv \
    --model  scoring/model/best_model.pt \
    --config scoring/model/model_config.json \
    --output results/predictions/val_scores.tsv
```

### 6. Evaluation
Evaluate predictions against ground truth labels:
```bash
# See evaluation/README.md
```

---

## Models

| Model | Dataset | Seq length | AUC |
|-------|---------|------------|-----|
| NT2 + LoRA (rank=32) + CNN | BvsP | 12k | **0.886** |

---

## Requirements

```bash
conda env create -f environment.yml
conda activate glm-finetune
```

---

## Citation

If you use this code, please cite: Su\*, Lin\*, et al. *Fine-tuning genomic language models for variant pathogenicity prediction*. In preparation.  
\*Co-first authors