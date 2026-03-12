#!/bin/bash
# subtract_new_variants.sh
# Extracts variants present in the NEW timestamp but NOT in the OLD timestamp,
# per clinical significance class. These are truly unseen variants for benchmarking
# models trained on the old timestamp.
#
# Uses variant_id (column 4 of BED) as the unique key for comparison.
#
# Output BED files named: ClinVar.{new_tag}only.missense.hg38.{class}.bed
#
# Usage: bash preprocessing/subtract_new_variants.sh (run from repo root)

BED_DIR="data/bed"
OLD_TAG="251103"
NEW_TAG="260309"

CLASSES=(
    "benign"
    "likely_benign"
    "likely_pathogenic"
    "pathogenic"
)

echo "Subtracting ${OLD_TAG} variants from ${NEW_TAG} to find new-only variants"
echo ""

for CLASS in "${CLASSES[@]}"; do
    OLD_BED="$BED_DIR/ClinVar.${OLD_TAG}.missense.hg38.${CLASS}.bed"
    NEW_BED="$BED_DIR/ClinVar.${NEW_TAG}.missense.hg38.${CLASS}.bed"
    OUT_BED="$BED_DIR/ClinVar.${NEW_TAG}only.missense.hg38.${CLASS}.bed"

    if [ ! -f "$OLD_BED" ]; then
        echo "ERROR: Old BED not found: $OLD_BED"
        exit 1
    fi
    if [ ! -f "$NEW_BED" ]; then
        echo "ERROR: New BED not found: $NEW_BED"
        exit 1
    fi

    if [ -f "$OUT_BED" ]; then
        echo "Skipping (already exists): $OUT_BED"
        continue
    fi

    # Build a key from chr + pos + ref + alt (columns 1, 2, 5, 6) for each
    # variant in the old file, then keep only rows from the new file whose
    # key is NOT present in the old file.
    OLD_KEYS=$(mktemp)
    awk '{print $1"_"$2"_"$5"_"$6}' "$OLD_BED" > "$OLD_KEYS"

    awk 'NR==FNR { old[$1]=1; next } !(($1"_"$2"_"$5"_"$6) in old)'         "$OLD_KEYS" "$NEW_BED" > "$OUT_BED"

    OLD_COUNT=$(wc -l < "$OLD_BED")
    NEW_COUNT=$(wc -l < "$NEW_BED")
    OUT_COUNT=$(wc -l < "$OUT_BED")

    echo "${CLASS}:"
    echo "  ${OLD_TAG}: ${OLD_COUNT} variants"
    echo "  ${NEW_TAG}: ${NEW_COUNT} variants"
    echo "  New-only:  ${OUT_COUNT} variants -> $(basename "$OUT_BED")"
    echo ""

    rm "$OLD_KEYS"
done

echo "Done! New-only BED files written to $BED_DIR"