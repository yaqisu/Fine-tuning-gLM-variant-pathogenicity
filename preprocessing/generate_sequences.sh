#!/bin/bash
# generate_sequences.sh
# Generates all sequence txt files from ClinVar BED files
# Usage: bash preprocessing/generate_sequences.sh (run from repo root)
#
# Note: -l specifies flanking bp on EACH SIDE of the variant.
# Total window = (2 * l) + 1. Mappings used:
#   seq6k   -> -l 2999   (5999 bp total)
#   seq12k  -> -l 5999   (11999 bp total)
#   seq30k  -> -l 14999  (29999 bp total)
#   seq60k  -> -l 29999  (59999 bp total)
#   seq130k -> -l 64999  (129999 bp total)

GENOME="preprocessing/reference/Homo_sapiens.GRCh38.dna.primary_assembly.fa"
BED_DIR="preprocessing/ClinVar_BED_files"
OUT_DIR="data"
SCRIPT="preprocessing/extract_variant_sequences.py"

# Create output directory if it doesn't exist
mkdir -p "$OUT_DIR"

# Format: suffix -> flank length
declare -A LENGTHS=(
    ["6k"]=2999
    ["12k"]=5999
    ["30k"]=14999
    ["60k"]=29999
    ["130k"]=64999
)

BED_FILES=(
    "ClinVar.251103.missense.hg38.benign.bed"
    "ClinVar.251103.missense.hg38.likely_benign.bed"
    "ClinVar.251103.missense.hg38.likely_pathogenic.bed"
    "ClinVar.251103.missense.hg38.pathogenic.bed"
)

for BED in "${BED_FILES[@]}"; do
    for SUFFIX in "${!LENGTHS[@]}"; do
        FLANK="${LENGTHS[$SUFFIX]}"
        INPUT="$BED_DIR/$BED"
        OUTPUT="$OUT_DIR/${BED}.seq${SUFFIX}.txt"

        # Skip if output already exists
        if [ -f "$OUTPUT" ]; then
            echo "Skipping (already exists): $OUTPUT"
            continue
        fi

        echo "Generating: $OUTPUT  (flank=-l $FLANK)"
        python "$SCRIPT" -b "$INPUT" -f "$GENOME" -l "$FLANK" -o "$OUTPUT"

        if [ $? -ne 0 ]; then
            echo "ERROR: Failed on $OUTPUT"
            exit 1
        fi
    done
done

echo "Done! All sequence files generated."