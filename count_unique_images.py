#!/usr/bin/env python3
"""
Count unique patient_id + instance_no combinations across different dataset versions.
"""

import os
import re
from pathlib import Path
from collections import defaultdict

def extract_patient_instance(filename: str) -> tuple:
    """
    Extract (patient_id, instance_no) from various filename formats.

    Formats:
    - YOLO: 520892552_14.jpg → (520892552, 14)
    - Roboflow: 525190930_29_png.rf.xxx.jpg → (525190930, 29)
    - COCO: v3_520892552_14.jpg → (520892552, 14)
    - Augmented: 520892552_14_aug0.jpg → (520892552, 14)
    """
    # Remove extension
    base = os.path.splitext(filename)[0]

    # Remove Roboflow hash if present
    if '.rf.' in base:
        base = base.split('.rf.')[0]

    # Remove augmentation suffix
    if '_aug' in base:
        base = base.rsplit('_aug', 1)[0]

    # Remove common prefixes (loop until no more prefixes found)
    # This handles cases like "v3_thick_12345" where there are multiple prefixes
    prefixes = ['v3_', 'roboflow_', 'thin_', 'thick_', 'no_hem_']
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if base.startswith(prefix):
                base = base[len(prefix):]
                changed = True
                break

    # Extract two consecutive numeric parts
    parts = base.split('_')

    # Look for pattern: number_number
    for i in range(len(parts) - 1):
        if parts[i].isdigit() and parts[i+1].isdigit():
            return (parts[i], parts[i+1])

    # Fallback: first number is patient_id
    if len(parts) >= 1 and parts[0].isdigit():
        instance = parts[1] if len(parts) > 1 and parts[1].isdigit() else "0"
        return (parts[0], instance)

    return (base, "0")


def count_unique_images_in_folder(folder_path: str) -> set:
    """Count unique (patient_id, instance_no) combinations in a folder."""
    unique_combos = set()

    if not os.path.exists(folder_path):
        print(f"  ⚠️ Folder not found: {folder_path}")
        return unique_combos

    # Walk through all subdirectories
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            # Skip non-image files and annotations
            if filename.startswith('.') or filename.startswith('_'):
                continue

            ext = os.path.splitext(filename)[1].lower()
            if ext not in ['.jpg', '.jpeg', '.png']:
                continue

            combo = extract_patient_instance(filename)
            unique_combos.add(combo)

    return unique_combos


def analyze_dataset(name: str, base_path: str, has_splits: bool = True):
    """Analyze a dataset and print statistics."""
    print(f"\n{'='*80}")
    print(f"{name}")
    print(f"{'='*80}")
    print(f"Path: {base_path}")

    if not os.path.exists(base_path):
        print(f"❌ Dataset not found!")
        return

    all_unique = set()
    split_counts = {}

    if has_splits:
        # Check common split names
        splits = ['train', 'valid', 'test', 'val']

        for split in splits:
            split_path = os.path.join(base_path, split)
            if os.path.exists(split_path):
                unique = count_unique_images_in_folder(split_path)
                split_counts[split] = len(unique)
                all_unique.update(unique)
                print(f"  {split:8s}: {len(unique):,} unique images")
    else:
        # Single folder
        all_unique = count_unique_images_in_folder(base_path)
        split_counts['all'] = len(all_unique)
        print(f"  Total: {len(all_unique):,} unique images")

    print(f"\n  {'─'*40}")
    print(f"  TOTAL UNIQUE: {len(all_unique):,} patient-instance combinations")

    return all_unique, split_counts


def main():
    print("="*80)
    print("UNIQUE IMAGE COUNTER - Brain CT Hemorrhage Datasets")
    print("="*80)
    print("\nCounting unique (patient_id, instance_no) combinations...")

    datasets = {
        'V1': {
            'path': '/Users/yugambahl/Desktop/brain_ct/data/processed/roboflow_versions/ct_brain_hemorrhage.v6i.yolov8',
            'has_splits': True
        },
        'V2': {
            'path': '/Users/yugambahl/Desktop/brain_ct/data/training_datasets/archived/current_20251105/UAT_CT_BRAIN_HEMORRHAGE.v2i.yolov8_combined',
            'has_splits': True
        },
        'V3': {
            'path': '/Users/yugambahl/Desktop/brain_ct/data/training_datasets/v3/filtered_4class/coco',
            'has_splits': True
        },
        'V4': {
            'path': '/Users/yugambahl/Desktop/brain_ct/data/training_datasets/v4/temp_combined',
            'has_splits': True
        }
    }

    results = {}

    for name, config in datasets.items():
        unique_set, split_counts = analyze_dataset(name, config['path'], config['has_splits'])
        results[name] = {
            'unique_set': unique_set,
            'total': len(unique_set) if unique_set else 0,
            'split_counts': split_counts
        }

    # Summary comparison
    print(f"\n\n{'='*80}")
    print("SUMMARY COMPARISON")
    print(f"{'='*80}")

    for name in ['V1', 'V2', 'V3', 'V4']:
        if name in results and results[name]:
            total = results[name]['total']
            print(f"{name}: {total:,} unique images")

    # Check if V4 contains all previous datasets
    if all(results.get(name, {}).get('unique_set') for name in ['V1', 'V2', 'V3', 'V4']):
        print(f"\n{'='*80}")
        print("V4 CONTAINMENT ANALYSIS")
        print(f"{'='*80}")

        v1_set = results['V1']['unique_set']
        v2_set = results['V2']['unique_set']
        v3_set = results['V3']['unique_set']
        v4_set = results['V4']['unique_set']

        v1_in_v4 = v1_set & v4_set
        v2_in_v4 = v2_set & v4_set
        v3_in_v4 = v3_set & v4_set

        print(f"\nV1 images in V4: {len(v1_in_v4):,} / {len(v1_set):,} ({len(v1_in_v4)/len(v1_set)*100:.1f}%)")
        print(f"V2 images in V4: {len(v2_in_v4):,} / {len(v2_set):,} ({len(v2_in_v4)/len(v2_set)*100:.1f}%)")
        print(f"V3 images in V4: {len(v3_in_v4):,} / {len(v3_set):,} ({len(v3_in_v4)/len(v3_set)*100:.1f}%)")

        # Check what's missing from each
        v1_missing = v1_set - v4_set
        v2_missing = v2_set - v4_set
        v3_missing = v3_set - v4_set

        if v1_missing:
            print(f"\n⚠️ V1 images NOT in V4: {len(v1_missing):,}")
        else:
            print(f"\n✓ All V1 images are in V4")

        if v2_missing:
            print(f"⚠️ V2 images NOT in V4: {len(v2_missing):,}")
        else:
            print(f"✓ All V2 images are in V4")

        if v3_missing:
            print(f"⚠️ V3 images NOT in V4: {len(v3_missing):,}")
        else:
            print(f"✓ All V3 images are in V4")

        # What's new in V4?
        v4_new = v4_set - v1_set - v2_set - v3_set
        print(f"\n📥 NEW images in V4 (not in V1/V2/V3): {len(v4_new):,}")

    # Original overlap analysis
    if all(results.get(name, {}).get('unique_set') for name in ['V1', 'V2', 'V3']):
        print(f"\n\n{'='*80}")
        print("DATASET OVERLAPS (V1/V2/V3)")
        print(f"{'='*80}")

        v1_set = results['V1']['unique_set']
        v2_set = results['V2']['unique_set']
        v3_set = results['V3']['unique_set']

        v1_v2_overlap = v1_set & v2_set
        v1_v3_overlap = v1_set & v3_set
        v2_v3_overlap = v2_set & v3_set
        all_overlap = v1_set & v2_set & v3_set

        print(f"V1 ∩ V2: {len(v1_v2_overlap):,} images")
        print(f"V1 ∩ V3: {len(v1_v3_overlap):,} images")
        print(f"V2 ∩ V3: {len(v2_v3_overlap):,} images")
        print(f"V1 ∩ V2 ∩ V3: {len(all_overlap):,} images")

        # Unique to each
        v1_only = v1_set - v2_set - v3_set
        v2_only = v2_set - v1_set - v3_set
        v3_only = v3_set - v1_set - v2_set

        print(f"\nUnique to V1: {len(v1_only):,} images")
        print(f"Unique to V2: {len(v2_only):,} images")
        print(f"Unique to V3: {len(v3_only):,} images")


if __name__ == "__main__":
    main()
