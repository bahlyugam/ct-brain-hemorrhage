#!/bin/bash
# Upload train folder in batches to avoid Modal timeout

VOLUME="medium-v5_rfdetr-brain-ct-hemorrhage"
SOURCE="/Users/yugambahl/Desktop/brain_ct/data/training_datasets/v5/roboflow_augmented_3x/train"
DEST="/v5_augmented/train"
BATCH_SIZE=10000
TEMP_DIR="/tmp/modal_upload_batch"

echo "=============================================="
echo "BATCHED UPLOAD TO MODAL"
echo "=============================================="
echo "Volume: $VOLUME"
echo "Source: $SOURCE"
echo "Batch size: $BATCH_SIZE files"
echo "=============================================="

# Get all files
cd "$SOURCE"
FILES=(*)
TOTAL=${#FILES[@]}
echo "Total files: $TOTAL"

# Calculate number of batches
NUM_BATCHES=$(( (TOTAL + BATCH_SIZE - 1) / BATCH_SIZE ))
echo "Number of batches: $NUM_BATCHES"
echo ""

# First upload the annotation file
echo "Uploading _annotations.coco.json..."
modal volume put "$VOLUME" "$SOURCE/_annotations.coco.json" "$DEST/_annotations.coco.json"
echo ""

# Upload in batches
for ((batch=0; batch<NUM_BATCHES; batch++)); do
    START=$((batch * BATCH_SIZE))
    END=$((START + BATCH_SIZE))
    if [ $END -gt $TOTAL ]; then
        END=$TOTAL
    fi

    BATCH_NUM=$((batch + 1))
    echo "=============================================="
    echo "Batch $BATCH_NUM/$NUM_BATCHES (files $START to $END)"
    echo "=============================================="

    # Create temp directory
    rm -rf "$TEMP_DIR"
    mkdir -p "$TEMP_DIR"

    # Copy files to temp directory (excluding annotation file already uploaded)
    COUNT=0
    for ((i=START; i<END; i++)); do
        FILE="${FILES[$i]}"
        if [[ "$FILE" != "_annotations.coco.json" ]]; then
            cp "$SOURCE/$FILE" "$TEMP_DIR/"
            COUNT=$((COUNT + 1))
        fi
    done

    echo "Files in batch: $COUNT"

    # Upload batch
    if [ $COUNT -gt 0 ]; then
        modal volume put "$VOLUME" "$TEMP_DIR" "$DEST" --force

        if [ $? -eq 0 ]; then
            echo "✓ Batch $BATCH_NUM completed"
        else
            echo "✗ Batch $BATCH_NUM failed!"
            echo "Retrying in 10 seconds..."
            sleep 10
            modal volume put "$VOLUME" "$TEMP_DIR" "$DEST" --force
            if [ $? -ne 0 ]; then
                echo "Batch $BATCH_NUM failed after retry. Stopping."
                rm -rf "$TEMP_DIR"
                exit 1
            fi
        fi
    fi

    # Cleanup
    rm -rf "$TEMP_DIR"

    # Small delay between batches
    sleep 2
done

echo ""
echo "=============================================="
echo "✓ UPLOAD COMPLETE!"
echo "=============================================="
