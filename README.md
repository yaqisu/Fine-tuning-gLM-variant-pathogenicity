# Fine tuning genomic language models for variant pathogenicity prediction
This repository contains codes for reproduction of results in the ISMB 2026 paper submission titled "Fine tuning genomic languague model for variant pathogenicity prediction"

## Repository Structure

### Data_preprocessing/
Codes for generating training and validation data. The generated dataset can be downloaded [here](https://drive.google.com/file/d/1srEASzSajo2EVu-0WANFGVbqXT68BhwC/view?usp=sharing)

### NT2_phase1_and_unfreezeAll.py
Script for phase 1 experiments (frozen model backbone) and full fine-tuning using Nucleotide Transformer 2.

### NT2_phase2_and_phase3.py
Script for phase 2 (LoRA fine-tuning) and phase 3 (learning rate) experiments using Nucleotide Transformer 2.

### NT1_phase1.py
Script for phase 1 experiments using Nucleotide Transformer 1.

### caduceus_phase1.py
Script for phase 1 experiments for Caduceus.

### figures.ipynb
Jupyter notebook for generating figures and visualizations from experimental results.

### output/
Directory containing experimental results and generated figures.
- `output/results.tsv`: Combined results table used for figure generation
- `output/figures/`: Generated figures from `figures.ipynb`
