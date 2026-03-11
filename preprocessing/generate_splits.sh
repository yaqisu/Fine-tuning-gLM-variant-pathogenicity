#!/bin/bash
# generate_splits.sh
# Generates train/val splits for all sequence lengths in two modes:
#   BvsP    — benign (0) + pathogenic (1) only, all 5 window sizes
#   BLBvsPLP — benign (0) + likely_benign (0) + likely_pathogenic (1) + pathogenic (1)
#              only seq6k and seq12k (likely_* files not available for larger windows)
#
# Output files named: ClinVar.251103.missense.hg38.seq{size}.{mode}_{training|validation}.tsv
#
# Usage: bash preprocessing/generate_splits.sh (run from repo root)

SEQ_DIR="data/sequences"
OUT_DIR="data/splits"
SCRIPT="preprocessing/split_data_fixed_chroms.py"

mkdir -p "$OUT_DIR"

# ── BvsP splits (all 5 window sizes) ──────────────────────────────────────────
echo "========================================"
echo "  BvsP splits (benign vs pathogenic)"
echo "========================================"
for SIZE in "6k" "12k" "30k" "60k" "130k"; do
    echo "--- seq${SIZE} ---"
    python "$SCRIPT" \
        --mode BvsP \
        --benign     "$SEQ_DIR/ClinVar.251103.missense.hg38.benign.bed.seq${SIZE}.tsv" \
        --pathogenic "$SEQ_DIR/ClinVar.251103.missense.hg38.pathogenic.bed.seq${SIZE}.tsv" \
        --train "$OUT_DIR/ClinVar.251103.missense.hg38.seq${SIZE}.BvsP_training.tsv" \
        --val   "$OUT_DIR/ClinVar.251103.missense.hg38.seq${SIZE}.BvsP_validation.tsv"
    echo ""
done

# ── BLBvsPLP splits (all 5 window sizes) ─────────────────────────────────────
echo "========================================"
echo "  BLBvsPLP splits (B+LB vs P+LP)"
echo "========================================"
for SIZE in "6k" "12k" "30k" "60k" "130k"; do
    echo "--- seq${SIZE} ---"
    python "$SCRIPT" \
        --mode BLBvsPLP \
        --benign            "$SEQ_DIR/ClinVar.251103.missense.hg38.benign.bed.seq${SIZE}.tsv" \
        --pathogenic        "$SEQ_DIR/ClinVar.251103.missense.hg38.pathogenic.bed.seq${SIZE}.tsv" \
        --likely_benign     "$SEQ_DIR/ClinVar.251103.missense.hg38.likely_benign.bed.seq${SIZE}.tsv" \
        --likely_pathogenic "$SEQ_DIR/ClinVar.251103.missense.hg38.likely_pathogenic.bed.seq${SIZE}.tsv" \
        --train "$OUT_DIR/ClinVar.251103.missense.hg38.seq${SIZE}.BLBvsPLP_training.tsv" \
        --val   "$OUT_DIR/ClinVar.251103.missense.hg38.seq${SIZE}.BLBvsPLP_validation.tsv"
    echo ""
done

echo "Done! All splits generated in $OUT_DIR"
echo ""
echo "Output files:"
ls "$OUT_DIR"/ClinVar.251103.missense.hg38.seq*.tsv 2>/dev/null | sort