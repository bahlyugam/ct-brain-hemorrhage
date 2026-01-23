#!/usr/bin/env python3
"""
Upload V5 dataset to Modal volume in batches to avoid timeout issues.

Usage:
    python scripts/upload_to_modal_batched.py
    python scripts/upload_to_modal_batched.py --batch-size 5000
    python scripts/upload_to_modal_batched.py --resume  # Resume from last batch
"""

import subprocess
import os
import sys
import argparse
import json
from pathlib import Path
import time

# Configuration
VOLUME_NAME = "medium-v5_rfdetr-brain-ct-hemorrhage"
LOCAL_DIR = "/Users/yugambahl/Desktop/brain_ct/data/training_datasets/v5/roboflow_augmented_3x"
REMOTE_DIR = "/v5_augmented"
BATCH_SIZE = 5000  # Files per batch
PROGRESS_FILE = "/Users/yugambahl/Desktop/brain_ct/.modal_upload_progress.json"


def get_all_files(local_dir: str) -> list:
    """Get all files in directory recursively."""
    files = []
    for root, dirs, filenames in os.walk(local_dir):
        for filename in filenames:
            filepath = os.path.join(root, filename)
            relpath = os.path.relpath(filepath, local_dir)
            files.append((filepath, relpath))
    return sorted(files)


def load_progress() -> dict:
    """Load upload progress from file."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {'completed_batches': [], 'failed_batches': []}


def save_progress(progress: dict):
    """Save upload progress to file."""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


def upload_batch(files: list, batch_num: int, total_batches: int) -> bool:
    """Upload a batch of files to Modal volume."""
    print(f"\n{'='*60}")
    print(f"Uploading batch {batch_num}/{total_batches} ({len(files)} files)")
    print(f"{'='*60}")

    # Create a temporary directory structure for this batch
    temp_dir = f"/tmp/modal_batch_{batch_num}"
    os.makedirs(temp_dir, exist_ok=True)

    try:
        # Create symlinks to original files in temp directory
        for filepath, relpath in files:
            dest = os.path.join(temp_dir, relpath)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            if os.path.exists(dest):
                os.remove(dest)
            os.symlink(filepath, dest)

        # Upload the batch
        cmd = [
            "modal", "volume", "put",
            VOLUME_NAME,
            temp_dir,
            REMOTE_DIR,
            "--force"  # Overwrite existing files
        ]

        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=False, text=True)

        if result.returncode == 0:
            print(f"✓ Batch {batch_num} uploaded successfully")
            return True
        else:
            print(f"✗ Batch {batch_num} failed")
            return False

    except Exception as e:
        print(f"✗ Batch {batch_num} failed with error: {e}")
        return False
    finally:
        # Cleanup temp directory
        subprocess.run(["rm", "-rf", temp_dir], capture_output=True)


def upload_split(split_name: str, local_split_dir: str, batch_size: int, resume: bool = False):
    """Upload a single split (train/valid) in batches."""
    print(f"\n{'='*60}")
    print(f"Processing {split_name} split")
    print(f"{'='*60}")

    if not os.path.exists(local_split_dir):
        print(f"Directory not found: {local_split_dir}")
        return

    # Get all files
    files = get_all_files(local_split_dir)
    print(f"Total files in {split_name}: {len(files)}")

    # Create batches
    batches = []
    for i in range(0, len(files), batch_size):
        batch_files = files[i:i+batch_size]
        batches.append(batch_files)

    print(f"Split into {len(batches)} batches of ~{batch_size} files each")

    # Load progress
    progress = load_progress()
    completed_key = f"{split_name}_completed"
    if completed_key not in progress:
        progress[completed_key] = []

    # Upload each batch
    for batch_idx, batch_files in enumerate(batches):
        batch_num = batch_idx + 1

        if resume and batch_num in progress[completed_key]:
            print(f"Skipping batch {batch_num} (already completed)")
            continue

        # Prepare files with correct relative paths for this split
        batch_with_split = [
            (fp, f"{split_name}/{rp}") for fp, rp in batch_files
        ]

        success = upload_batch(batch_with_split, batch_num, len(batches))

        if success:
            progress[completed_key].append(batch_num)
            save_progress(progress)
        else:
            print(f"\n⚠ Batch {batch_num} failed. Run with --resume to continue from here.")
            # Wait a bit before potentially retrying
            time.sleep(5)
            # Retry once
            print("Retrying...")
            success = upload_batch(batch_with_split, batch_num, len(batches))
            if success:
                progress[completed_key].append(batch_num)
                save_progress(progress)
            else:
                print(f"Batch {batch_num} failed after retry. Stopping.")
                return False

    return True


def main():
    parser = argparse.ArgumentParser(description='Upload V5 dataset to Modal in batches')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help=f'Files per batch (default: {BATCH_SIZE})')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from last successful batch')
    parser.add_argument('--split', type=str, choices=['train', 'valid', 'all'],
                        default='all', help='Which split to upload')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("MODAL VOLUME BATCHED UPLOAD")
    print(f"{'='*60}")
    print(f"Volume: {VOLUME_NAME}")
    print(f"Local: {LOCAL_DIR}")
    print(f"Remote: {REMOTE_DIR}")
    print(f"Batch size: {args.batch_size}")
    print(f"Resume: {args.resume}")
    print(f"{'='*60}")

    # Upload annotation files first (they're small)
    print("\nUploading annotation files...")
    for split in ['train', 'valid']:
        ann_file = f"{LOCAL_DIR}/{split}/_annotations.coco.json"
        if os.path.exists(ann_file):
            cmd = [
                "modal", "volume", "put",
                VOLUME_NAME,
                ann_file,
                f"{REMOTE_DIR}/{split}/_annotations.coco.json"
            ]
            print(f"  Uploading {split}/_annotations.coco.json...")
            subprocess.run(cmd, capture_output=True)

    # Upload splits
    if args.split in ['train', 'all']:
        success = upload_split('train', f"{LOCAL_DIR}/train", args.batch_size, args.resume)
        if not success:
            print("Train upload incomplete. Run with --resume to continue.")
            return

    if args.split in ['valid', 'all']:
        success = upload_split('valid', f"{LOCAL_DIR}/valid", args.batch_size, args.resume)
        if not success:
            print("Valid upload incomplete. Run with --resume to continue.")
            return

    print(f"\n{'='*60}")
    print("✓ UPLOAD COMPLETE!")
    print(f"{'='*60}")

    # Cleanup progress file
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
        print("Cleaned up progress file.")


if __name__ == '__main__':
    main()
