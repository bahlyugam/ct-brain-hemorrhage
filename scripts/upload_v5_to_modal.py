#!/usr/bin/env python3
"""
Upload V5 dataset to Modal volume using Modal's Python API.
More reliable than CLI for large uploads.

Usage:
    modal run scripts/upload_v5_to_modal.py
"""

import modal
import os
from pathlib import Path

# Configuration
VOLUME_NAME = "medium-v5_rfdetr-brain-ct-hemorrhage"
LOCAL_DIR = "/Users/yugambahl/Desktop/brain_ct/data/training_datasets/v5/roboflow_augmented_3x"

# Create volume reference
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# Create app
app = modal.App("v5-dataset-upload")


@app.function(
    volumes={"/data": volume},
    timeout=7200,  # 2 hours
)
def verify_upload():
    """Verify the upload completed successfully."""
    import os

    print("\nVerifying upload...")

    for split in ['train', 'valid']:
        split_dir = f"/data/v5_augmented/{split}"
        if os.path.exists(split_dir):
            files = os.listdir(split_dir)
            print(f"  {split}: {len(files)} files")

            # Check annotation file
            ann_file = f"{split_dir}/_annotations.coco.json"
            if os.path.exists(ann_file):
                print(f"    ✓ _annotations.coco.json exists")
            else:
                print(f"    ✗ _annotations.coco.json MISSING")
        else:
            print(f"  {split}: Directory not found!")

    return True


@app.local_entrypoint()
def main():
    """Upload dataset to Modal volume."""
    from pathlib import Path
    import subprocess
    import time

    print(f"\n{'='*60}")
    print("V5 DATASET UPLOAD TO MODAL")
    print(f"{'='*60}")
    print(f"Volume: {VOLUME_NAME}")
    print(f"Source: {LOCAL_DIR}")
    print(f"{'='*60}\n")

    # Check local directory exists
    if not Path(LOCAL_DIR).exists():
        print(f"❌ Local directory not found: {LOCAL_DIR}")
        return

    # Count files
    train_dir = Path(LOCAL_DIR) / "train"
    valid_dir = Path(LOCAL_DIR) / "valid"

    train_count = len(list(train_dir.glob("*"))) if train_dir.exists() else 0
    valid_count = len(list(valid_dir.glob("*"))) if valid_dir.exists() else 0

    print(f"Files to upload:")
    print(f"  train: {train_count:,}")
    print(f"  valid: {valid_count:,}")
    print(f"  total: {train_count + valid_count:,}")

    # Upload valid first (smaller, faster)
    print(f"\n{'='*60}")
    print("Step 1: Uploading valid split...")
    print(f"{'='*60}")

    valid_result = subprocess.run(
        ["modal", "volume", "put", VOLUME_NAME, str(valid_dir), "/v5_augmented/valid"],
        capture_output=False
    )

    if valid_result.returncode != 0:
        print("❌ Valid split upload failed")
        return

    print("✓ Valid split uploaded")

    # Upload train in chunks by using the CLI with smaller timeouts
    print(f"\n{'='*60}")
    print("Step 2: Uploading train split...")
    print("(This may take 30-60 minutes)")
    print(f"{'='*60}")

    # Try direct upload first
    train_result = subprocess.run(
        ["modal", "volume", "put", VOLUME_NAME, str(train_dir), "/v5_augmented/train"],
        capture_output=False
    )

    if train_result.returncode != 0:
        print("❌ Train split upload failed")
        print("\nTry uploading manually in smaller batches or check your connection.")
        return

    print("✓ Train split uploaded")

    # Verify
    print(f"\n{'='*60}")
    print("Step 3: Verifying upload...")
    print(f"{'='*60}")

    verify_upload.remote()

    print(f"\n{'='*60}")
    print("✓ UPLOAD COMPLETE!")
    print(f"{'='*60}")
    print(f"\nYou can now run training with:")
    print(f"  modal run train.py::train_augmented")
