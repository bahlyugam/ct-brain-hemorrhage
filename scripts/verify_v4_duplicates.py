#!/usr/bin/env python3
"""
Verify duplicates in v4 dataset based on patientID_instanceNumber.
"""

import os
from pathlib import Path
from collections import defaultdict


def analyze_duplicates(v4_dataset_path):
    """Analyze duplicate images across train/valid/test splits."""
    v4_path = Path(v4_dataset_path)

    # Collect all filenames from each split
    train_files = set()
    valid_files = set()
    test_files = set()

    # Get train aug0 files (remove _aug0 suffix for comparison)
    train_folder = v4_path / "train"
    if train_folder.exists():
        for f in train_folder.iterdir():
            if f.suffix == '.png' and f.name.endswith('_aug0.png'):
                # Extract patientID_instanceNumber
                base_name = f.name.replace('_aug0.png', '.png')
                train_files.add(base_name)

    # Get valid files
    valid_folder = v4_path / "valid"
    if valid_folder.exists():
        for f in valid_folder.iterdir():
            if f.suffix == '.png':
                valid_files.add(f.name)

    # Get test files
    test_folder = v4_path / "test"
    if test_folder.exists():
        for f in test_folder.iterdir():
            if f.suffix == '.png':
                test_files.add(f.name)

    print("=" * 80)
    print("V4 DATASET DUPLICATE ANALYSIS")
    print("=" * 80)
    print(f"\nTotal images per split (based on patientID_instanceNumber):")
    print(f"  Train (aug0 only):  {len(train_files):,}")
    print(f"  Valid:              {len(valid_files):,}")
    print(f"  Test:               {len(test_files):,}")
    print(f"  Total:              {len(train_files) + len(valid_files) + len(test_files):,}")

    # Find duplicates
    train_valid_overlap = train_files & valid_files
    train_test_overlap = train_files & test_files
    valid_test_overlap = valid_files & test_files
    all_three_overlap = train_files & valid_files & test_files

    print(f"\n" + "-" * 80)
    print("DUPLICATE ANALYSIS:")
    print(f"  Train ∩ Valid:      {len(train_valid_overlap):,} images")
    print(f"  Train ∩ Test:       {len(train_test_overlap):,} images")
    print(f"  Valid ∩ Test:       {len(valid_test_overlap):,} images")
    print(f"  All three splits:   {len(all_three_overlap):,} images")

    # Calculate unique images
    all_unique = train_files | valid_files | test_files
    total_with_duplicates = len(train_files) + len(valid_files) + len(test_files)
    total_duplicates = total_with_duplicates - len(all_unique)

    print(f"\n" + "-" * 80)
    print("SUMMARY:")
    print(f"  Unique images (patientID_instanceNumber):  {len(all_unique):,}")
    print(f"  Total duplicate occurrences:                {total_duplicates:,}")
    print(f"  Deduplication rate:                         {total_duplicates / total_with_duplicates * 100:.2f}%")

    # Show some example duplicates
    if train_valid_overlap:
        print(f"\n" + "-" * 80)
        print("Example duplicates (Train ∩ Valid) - first 10:")
        for i, fname in enumerate(sorted(train_valid_overlap)[:10], 1):
            print(f"  {i:2}. {fname}")

    if train_test_overlap:
        print(f"\n" + "-" * 80)
        print("Example duplicates (Train ∩ Test) - first 10:")
        for i, fname in enumerate(sorted(train_test_overlap)[:10], 1):
            print(f"  {i:2}. {fname}")

    print("\n" + "=" * 80)

    return {
        'train': len(train_files),
        'valid': len(valid_files),
        'test': len(test_files),
        'unique': len(all_unique),
        'duplicates': total_duplicates,
        'train_valid_overlap': len(train_valid_overlap),
        'train_test_overlap': len(train_test_overlap),
        'valid_test_overlap': len(valid_test_overlap),
        'all_three_overlap': len(all_three_overlap)
    }


if __name__ == "__main__":
    V4_DATASET_PATH = "/Users/yugambahl/Desktop/brain_ct/data/training_datasets/v4/6class_coco"
    stats = analyze_duplicates(V4_DATASET_PATH)
