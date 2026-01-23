#!/bin/bash
# Prepare V4 Dataset with No-Hemorrhage Image Downloads
#
# This script automates the entire workflow:
# 1. Downloads missing no-hemorrhage images from CSV (prioritizing FP)
# 2. Creates COCO annotations for downloaded images
# 3. Runs the v4 dataset preparation pipeline with all images
#
# Usage:
#   bash scripts/prepare_v4_with_no_hem_downloads.sh
#   bash scripts/prepare_v4_with_no_hem_downloads.sh --max-images 2000
#   bash scripts/prepare_v4_with_no_hem_downloads.sh --fp-only

set -e  # Exit on error

echo "================================================================================"
echo "V4 DATASET PREPARATION WITH NO-HEMORRHAGE IMAGE DOWNLOADS"
echo "================================================================================"
echo ""

# Parse arguments
MAX_IMAGES=""
FP_ONLY=""
SKIP_DOWNLOAD=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --max-images)
            MAX_IMAGES="--max-images $2"
            shift 2
            ;;
        --fp-only)
            FP_ONLY="--fp-only"
            shift
            ;;
        --skip-download)
            SKIP_DOWNLOAD=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--max-images N] [--fp-only] [--skip-download]"
            exit 1
            ;;
    esac
done

# Step 1: Download missing no-hemorrhage images
if [ "$SKIP_DOWNLOAD" = false ]; then
    echo "[STEP 1/3] Downloading missing no-hemorrhage images from CSV..."
    echo "  Priority: False Positives (FP) > True Negatives (TN)"
    echo ""
    python scripts/download_missing_no_hemorrhage_images.py $MAX_IMAGES $FP_ONLY

    if [ $? -ne 0 ]; then
        echo "❌ Error: Image download failed"
        exit 1
    fi
else
    echo "[STEP 1/3] Skipping download (--skip-download flag set)"
fi

echo ""
echo "================================================================================"
echo ""

# Step 2: Create COCO annotations for downloaded images
echo "[STEP 2/3] Creating COCO annotations for downloaded images..."
echo ""
python scripts/create_no_hemorrhage_coco_annotations.py

if [ $? -ne 0 ]; then
    echo "❌ Error: COCO annotation creation failed"
    exit 1
fi

echo ""
echo "================================================================================"
echo ""

# Step 3: Run v4 dataset preparation with all images
echo "[STEP 3/3] Running v4 dataset preparation pipeline..."
echo "  This will merge:"
echo "    - V3 filtered 4-class dataset"
echo "    - Roboflow 8-category dataset"
echo "    - Downloaded no-hemorrhage images"
echo ""
python scripts/prepare_v4_dataset.py

if [ $? -ne 0 ]; then
    echo "❌ Error: V4 dataset preparation failed"
    exit 1
fi

echo ""
echo "================================================================================"
echo "✅ COMPLETE! V4 dataset prepared with no-hemorrhage balance optimization"
echo "================================================================================"
echo ""
echo "Next steps:"
echo "  1. Validate: python scripts/validate_v4_dataset.py"
echo "  2. Analyze: python scripts/analyze_v4_dataset.py"
echo "  3. Train: modal run train.py"
