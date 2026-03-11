#!/usr/bin/env python3
"""
score_variants.py
-----------------
Score variants from a pre-processed sequence TSV file using a fine-tuned
Nucleotide Transformer 2 (NT2) model with LoRA and CNN classifier.

The input TSV is expected to have the same format as the split files produced
by preprocessing/split_data_fixed_chroms.py:
    variant_id  chromosome  position  ref_allele  alt_allele
    upstream_flank  downstream_flank  ref_sequence  alt_sequence  [label]

The label column is optional — if present, AUC will also be computed.

Output TSV columns:
    variant_id  chromosome  position  ref_allele  alt_allele
    pathogenicity_score  predicted_label

    pathogenicity_score : sigmoid probability (0-1), higher = more pathogenic
    predicted_label     : 0 (benign) or 1 (pathogenic) at 0.5 threshold

Usage:
    python scoring/score_variants.py \
        --input  data/splits/ClinVar.251103.missense.hg38.seq12k.BvsP_validation.tsv \
        --model  scoring/model/best_model.pt \
        --output results/predictions/my_scores.tsv
"""

import os
import sys
import logging
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForMaskedLM
from peft import LoraConfig, get_peft_model, TaskType

# ============================================================================
# Hardcoded config for the released best model
# (architecture must match the weights in best_model.pt exactly)
# ============================================================================

MODEL_CONFIG = {
    "classifier_type":    "cnn",
    "embedding_strategy": "full-variant_position",
    "lora_rank":          32,
    "num_layers":         2,
    "base_model":         "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('score_variants')


# ============================================================================
# Model Architecture (must match training code exactly)
# ============================================================================

class CNNClassifier(nn.Module):
    """CNN classifier — must match architecture used during training"""
    def __init__(self, input_dim=1024, dropout=0.1, pooling_strategy='mean_pool'):
        super(CNNClassifier, self).__init__()
        self.pooling_strategy = pooling_strategy
        self.conv1 = nn.Conv1d(input_dim, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(256)
        self.conv2 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(128, 1)

    def forward(self, x, variant_position=None, attention_mask=None):
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)

        if self.pooling_strategy == 'mean_pool':
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                pooled = torch.sum(mask * x, dim=1) / torch.sum(mask, dim=1)
            else:
                pooled = torch.mean(x, dim=1)
        elif self.pooling_strategy == 'variant_position':
            if variant_position is None:
                raise ValueError("variant_position required for variant_position pooling")
            pooled = x[:, variant_position, :]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

        return self.fc(pooled).squeeze(-1)


class MLPClassifier(nn.Module):
    """MLP classifier"""
    def __init__(self, input_dim=1024, num_layers=2, dropout=0.1):
        super(MLPClassifier, self).__init__()
        if num_layers == 2:
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, 512), nn.ReLU(inplace=True), nn.Dropout(p=dropout),
                nn.Linear(512, 256),       nn.ReLU(inplace=True), nn.Dropout(p=dropout),
                nn.Linear(256, 1)
            )
        elif num_layers == 3:
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, 512), nn.ReLU(inplace=True), nn.Dropout(p=dropout),
                nn.Linear(512, 256),       nn.ReLU(inplace=True), nn.Dropout(p=dropout),
                nn.Linear(256, 128),       nn.ReLU(inplace=True), nn.Dropout(p=dropout),
                nn.Linear(128, 1)
            )
        else:
            raise ValueError(f"num_layers must be 2 or 3, got {num_layers}")

    def forward(self, x):
        return self.classifier(x).squeeze(-1)


class TransformerClassifier(nn.Module):
    """Transformer classifier"""
    def __init__(self, input_dim=1024, embed_dim=128, nhead=2,
                 dim_feedforward=512, dropout=0.1, pooling_strategy='mean_pool'):
        super(TransformerClassifier, self).__init__()
        self.pooling_strategy = pooling_strategy
        self.input_projection = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, x, variant_position=None, attention_mask=None):
        x = self.input_projection(x)
        x = self.transformer(x)
        if self.pooling_strategy == 'mean_pool':
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                pooled = torch.sum(mask * x, dim=1) / torch.sum(mask, dim=1)
            else:
                pooled = torch.mean(x, dim=1)
        elif self.pooling_strategy == 'variant_position':
            pooled = x[:, variant_position, :]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
        return self.fc(self.dropout(pooled)).squeeze(-1)


class NT2_FineTune(nn.Module):
    """NT2 fine-tuning wrapper with LoRA"""
    def __init__(self, base_model, classifier_type, num_layers,
                 embedding_strategy, lora_rank):
        super(NT2_FineTune, self).__init__()
        self.bert = base_model
        self.embedding_strategy = embedding_strategy
        self.classifier_type = classifier_type

        # Freeze base model and apply LoRA
        for param in self.bert.parameters():
            param.requires_grad = False

        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            lora_dropout=0.1,
            target_modules=["query", "value"],
            bias="none"
        )
        self.bert = get_peft_model(self.bert, lora_config)
        self.bert.enable_input_require_grads()

        input_dim = 1024
        if embedding_strategy.startswith('full-'):
            pooling_strategy = embedding_strategy.split('-', 1)[1]
        else:
            pooling_strategy = 'mean_pool'

        if classifier_type == "mlp":
            self.classifier = MLPClassifier(input_dim, num_layers)
        elif classifier_type == "cnn":
            self.classifier = CNNClassifier(input_dim, pooling_strategy=pooling_strategy)
        elif classifier_type == "transformer":
            self.classifier = TransformerClassifier(input_dim, pooling_strategy=pooling_strategy)
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")

    def forward(self, input_ids, attention_mask=None):
        attention_mask = input_ids != 1

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            encoder_attention_mask=attention_mask,
            output_hidden_states=True
        )
        embed = outputs['hidden_states'][-1]

        if self.embedding_strategy == "variant_position":
            variant_position = min(1000, embed.shape[1] - 1)
            pooled_embed = embed[:, variant_position, :]
            logits = self.classifier(pooled_embed)

        elif self.embedding_strategy == "mean_pool":
            mask = attention_mask.unsqueeze(-1).float()
            pooled_embed = torch.sum(mask * embed, dim=1) / torch.sum(mask, dim=1)
            logits = self.classifier(pooled_embed)

        elif self.embedding_strategy == "full":
            logits = self.classifier(embed)

        elif self.embedding_strategy == "full-mean_pool":
            logits = self.classifier(embed, attention_mask=attention_mask)

        elif self.embedding_strategy == "full-variant_position":
            variant_position = min(1000, embed.shape[1] - 1)
            logits = self.classifier(embed, variant_position=variant_position,
                                     attention_mask=attention_mask)
        else:
            raise ValueError(f"Unknown embedding strategy: {self.embedding_strategy}")

        return logits


# ============================================================================
# Dataset
# ============================================================================

class ScoringDataset(Dataset):
    """Dataset for scoring — no label required"""
    def __init__(self, sequences, tokenizer, max_length=2048):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences.iloc[idx]
        encoding = self.tokenizer.encode_plus(
            sequence,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        return {
            'input_ids':      encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }


# ============================================================================
# Main Scoring Logic
# ============================================================================

def load_model(model_path, device):
    """Load fine-tuned model from .pt file using hardcoded MODEL_CONFIG"""
    config = MODEL_CONFIG
    logger.info(f"Model config: {config}")

    logger.info(f"Loading base NT2 model and tokenizer from {config['base_model']}...")
    tokenizer = AutoTokenizer.from_pretrained(
        config['base_model'], trust_remote_code=True
    )
    base_model = AutoModelForMaskedLM.from_pretrained(
        config['base_model'], trust_remote_code=True
    )

    model = NT2_FineTune(
        base_model,
        classifier_type=config['classifier_type'],
        num_layers=config['num_layers'],
        embedding_strategy=config['embedding_strategy'],
        lora_rank=config['lora_rank']
    )

    logger.info(f"Loading model weights from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    logger.info(f"Model loaded. Checkpoint val AUC: {checkpoint.get('val_auc', 'N/A'):.4f}")
    return model, tokenizer


def score(model, dataloader, device):
    """Run inference and return pathogenicity scores"""
    all_scores = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            logits = model(input_ids, attention_mask)
            scores = torch.sigmoid(logits).cpu().numpy()
            all_scores.extend(scores)
            if (i + 1) % 10 == 0:
                logger.info(f"  Scored {(i+1) * dataloader.batch_size} / {len(dataloader.dataset)} variants")
    return np.array(all_scores)


def main():
    parser = argparse.ArgumentParser(
        description='Score variants using a fine-tuned NT2 model.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python scoring/score_variants.py \\
      --input  data/splits/ClinVar.251103.missense.hg38.seq12k.BvsP_validation.tsv \\
      --model  scoring/model/best_model.pt \\
      --output results/predictions/scores.tsv
        """
    )
    parser.add_argument('--input',      '-i', required=True,
                        help='Input TSV file (from preprocessing pipeline)')
    parser.add_argument('--model',      '-m', required=True,
                        help='Path to best_model.pt')
    parser.add_argument('--output',     '-o', required=True,
                        help='Output TSV file path')
    parser.add_argument('--batch_size', '-b', type=int, default=16,
                        help='Batch size for inference (default: 16)')
    parser.add_argument('--gpu',        '-g', type=int, default=0,
                        help='GPU id to use, -1 for CPU (default: 0)')
    parser.add_argument('--threshold',  '-t', type=float, default=0.5,
                        help='Threshold for predicted_label (default: 0.5)')
    parser.add_argument('--k',          type=int, default=6,
                        help='K-mer size for tokenization (default: 6)')

    args = parser.parse_args()

    # Device setup
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        logger.info(f"Using GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")

    # Load input data
    logger.info(f"Loading input data from {args.input}")
    df = pd.read_csv(args.input, sep='\t')
    logger.info(f"Loaded {len(df)} variants")

    # Apply k-mer tokenization (same as training)
    sequences = df['alt_sequence'].apply(
        lambda x: ' '.join([x[i:i+args.k] for i in range(0, len(x), args.k)])
    )

    # Load model
    model, tokenizer = load_model(args.model, device)

    # Create dataloader
    dataset = ScoringDataset(sequences, tokenizer, max_length=2048)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Score
    logger.info(f"Scoring {len(df)} variants...")
    scores = score(model, dataloader, device)

    # Build output
    output_df = df[['variant_id', 'chromosome', 'position', 'ref_allele', 'alt_allele']].copy()
    output_df['pathogenicity_score'] = scores
    output_df['predicted_label']     = (scores >= args.threshold).astype(int)

    # If labels present, compute AUC
    if 'label' in df.columns:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(df['label'], scores)
        logger.info(f"AUC on labeled data: {auc:.4f}")
        output_df['true_label'] = df['label']

    # Save output
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output, sep='\t', index=False)
    logger.info(f"Saved {len(output_df)} scored variants to {args.output}")

    # Print summary
    n_patho  = (output_df['predicted_label'] == 1).sum()
    n_benign = (output_df['predicted_label'] == 0).sum()
    logger.info(f"Summary: {n_patho} predicted pathogenic, {n_benign} predicted benign "
                f"(threshold={args.threshold})")


if __name__ == '__main__':
    main()