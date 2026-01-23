#!/usr/bin/env python3
"""
Verify that V4 dataset contains all annotated data from V2 and V3 datasets.

V2: YOLO format (train/valid/test splits with labels/*.txt)
V3: COCO format (train/_annotations.coco.json)
V4: COCO format (train/valid/test splits with _annotations.coco.json)
"""

import os
import json
from pathlib import Path
from collections import defaultdict


def get_v2_annotated_images(base_path):
    """
    Extract annotated images from V2 dataset (YOLO format).
    Returns dict: {split: set of image basenames with annotations}
    """
    annotated_images = defaultdict(set)

    for split in ['train', 'valid', 'test']:
        labels_dir = Path(base_path) / split / 'labels'
        if not labels_dir.exists():
            continue

        # Get all label files
        for label_file in labels_dir.glob('*.txt'):
            if label_file.stat().st_size > 0:  # Only non-empty labels
                # Get corresponding image basename
                img_basename = label_file.stem
                annotated_images[split].add(img_basename)

    return annotated_images


def get_coco_annotated_images(json_path):
    """
    Extract annotated images from COCO format JSON.
    Returns set of image filenames that have annotations.
    """
    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    # Get images with annotations
    annotated_image_ids = set()
    for ann in coco_data.get('annotations', []):
        annotated_image_ids.add(ann['image_id'])

    # Map image IDs to filenames
    annotated_images = set()
    for img in coco_data.get('images', []):
        if img['id'] in annotated_image_ids:
            # Store filename without extension for comparison
            filename = Path(img['file_name']).stem
            annotated_images.add(filename)

    return annotated_images


def get_v3_annotated_images(base_path):
    """
    Extract annotated images from V3 dataset (COCO format).
    Returns dict: {split: set of image basenames with annotations}
    """
    annotated_images = {}

    # V3 only has train split
    json_path = Path(base_path) / '_annotations.coco.json'
    if json_path.exists():
        annotated_images['train'] = get_coco_annotated_images(json_path)

    return annotated_images


def get_v4_annotated_images(base_path):
    """
    Extract annotated images from V4 dataset (COCO format).
    Returns dict: {split: set of image basenames with annotations}
    """
    annotated_images = {}

    for split in ['train', 'valid', 'test']:
        json_path = Path(base_path) / split / '_annotations.coco.json'
        if json_path.exists():
            annotated_images[split] = get_coco_annotated_images(json_path)

    return annotated_images


def normalize_filename(filename):
    """
    Extract the core identifier: patientID_instanceNo

    This extracts the first occurrence of the pattern NUMBERS_NUMBERS from the filename,
    ignoring all prefixes (thick_, thin_, v2_, v3_, roboflow_) and suffixes
    (_aug0, _aug1, .rf.xxx, _png, etc.)

    Examples:
        thick_520892552_14.png -> 520892552_14
        v3_thick_520892552_14_aug0.png -> 520892552_14
        roboflow_525190930_29_png.rf.xxx_aug0.jpg -> 525190930_29
    """
    import re

    # Extract the patientID_instanceNo pattern (first occurrence of numbers_numbers)
    match = re.search(r'(\d+)_(\d+)', filename)
    if match:
        return f'{match.group(1)}_{match.group(2)}'

    return filename  # Return as-is if pattern not found


def find_matches(source_images, target_all_images):
    """
    Find which source images are present in target dataset (any split).
    Returns (matched, missing) sets with normalized filenames.
    """
    # Normalize all target images
    normalized_target = set()
    for split_images in target_all_images.values():
        for img in split_images:
            normalized_target.add(normalize_filename(img))

    matched = set()
    missing = set()

    for img in source_images:
        normalized = normalize_filename(img)
        if normalized in normalized_target:
            matched.add(normalized)
        else:
            missing.add(img)  # Keep original for reporting

    return matched, missing


def main():
    print("="*80)
    print("V4 DATASET COMPLETENESS VERIFICATION")
    print("="*80)
    print("\nChecking if V4 dataset contains all annotated data from V2 and V3...")

    # Dataset paths
    v2_path = "/Users/yugambahl/Desktop/brain_ct/data/training_datasets/archived/current_20251105/UAT_CT_BRAIN_HEMORRHAGE.v2i.yolov8_combined"
    v3_path = "/Users/yugambahl/Desktop/brain_ct/data/training_datasets/v3/filtered_4class/coco/train"
    v4_path = "/Users/yugambahl/Desktop/brain_ct/data/training_datasets/v4/6class_coco"

    # Extract annotated images from each dataset
    print("\n" + "-"*80)
    print("EXTRACTING ANNOTATED IMAGES FROM DATASETS")
    print("-"*80)

    print("\nV2 Dataset (YOLO format)...")
    v2_annotated = get_v2_annotated_images(v2_path)
    v2_total = sum(len(imgs) for imgs in v2_annotated.values())
    print(f"  Found {v2_total:,} annotated images across {len(v2_annotated)} splits")
    for split, imgs in v2_annotated.items():
        print(f"    {split}: {len(imgs):,} images")

    print("\nV3 Dataset (COCO format)...")
    v3_annotated = get_v3_annotated_images(v3_path)
    v3_total = sum(len(imgs) for imgs in v3_annotated.values())
    print(f"  Found {v3_total:,} annotated images across {len(v3_annotated)} splits")
    for split, imgs in v3_annotated.items():
        print(f"    {split}: {len(imgs):,} images")

    print("\nV4 Dataset (COCO format)...")
    v4_annotated = get_v4_annotated_images(v4_path)
    v4_total = sum(len(imgs) for imgs in v4_annotated.values())
    print(f"  Found {v4_total:,} annotated images across {len(v4_annotated)} splits")
    for split, imgs in v4_annotated.items():
        print(f"    {split}: {len(imgs):,} images")

    # Compare V2 -> V4
    print("\n" + "="*80)
    print("V2 -> V4 COMPARISON")
    print("="*80)

    for split, v2_images in v2_annotated.items():
        print(f"\n{split.upper()} split:")
        matched, missing = find_matches(v2_images, v4_annotated)

        print(f"  V2 annotated images: {len(v2_images):,}")
        print(f"  Found in V4: {len(matched):,} ({len(matched)/len(v2_images)*100:.1f}%)")
        print(f"  Missing from V4: {len(missing):,} ({len(missing)/len(v2_images)*100:.1f}%)")

        if missing and len(missing) <= 20:
            print(f"\n  Missing images:")
            for img in sorted(missing)[:20]:
                print(f"    - {img}")

    # Overall V2 -> V4
    all_v2_images = set()
    for imgs in v2_annotated.values():
        all_v2_images.update(imgs)

    matched, missing = find_matches(all_v2_images, v4_annotated)
    print(f"\nOVERALL V2 -> V4:")
    print(f"  V2 total annotated: {len(all_v2_images):,}")
    print(f"  Found in V4: {len(matched):,} ({len(matched)/len(all_v2_images)*100:.1f}%)")
    print(f"  Missing from V4: {len(missing):,} ({len(missing)/len(all_v2_images)*100:.1f}%)")

    # Compare V3 -> V4
    print("\n" + "="*80)
    print("V3 -> V4 COMPARISON")
    print("="*80)

    all_v3_images = set()
    for imgs in v3_annotated.values():
        all_v3_images.update(imgs)

    matched, missing = find_matches(all_v3_images, v4_annotated)
    print(f"\n  V3 total annotated: {len(all_v3_images):,}")
    print(f"  Found in V4: {len(matched):,} ({len(matched)/len(all_v3_images)*100:.1f}%)")
    print(f"  Missing from V4: {len(missing):,} ({len(missing)/len(all_v3_images)*100:.1f}%)")

    if missing and len(missing) <= 50:
        print(f"\n  Missing V3 images from V4 (showing up to 50):")
        for img in sorted(missing)[:50]:
            print(f"    - {img}")
    elif missing:
        print(f"\n  Missing V3 images from V4 (showing first 50 of {len(missing):,}):")
        for img in sorted(missing)[:50]:
            print(f"    - {img}")

    # Combined V2 + V3 -> V4
    print("\n" + "="*80)
    print("COMBINED V2 + V3 -> V4 COMPARISON")
    print("="*80)

    all_v2_v3_images = all_v2_images | all_v3_images
    matched, missing = find_matches(all_v2_v3_images, v4_annotated)

    print(f"\n  V2 unique annotated: {len(all_v2_images):,}")
    print(f"  V3 unique annotated: {len(all_v3_images):,}")
    print(f"  V2 + V3 combined (unique): {len(all_v2_v3_images):,}")
    print(f"  Found in V4: {len(matched):,} ({len(matched)/len(all_v2_v3_images)*100:.1f}%)")
    print(f"  Missing from V4: {len(missing):,} ({len(missing)/len(all_v2_v3_images)*100:.1f}%)")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    if len(missing) == 0:
        print("\n✓ SUCCESS: V4 contains ALL annotated images from V2 and V3!")
    else:
        print(f"\n✗ WARNING: V4 is missing {len(missing):,} annotated images from V2/V3")
        print(f"\n  This represents {len(missing)/len(all_v2_v3_images)*100:.2f}% of the combined V2+V3 dataset")

        if missing:
            print(f"\n  Sample of missing images (showing up to 30):")
            for img in sorted(missing)[:30]:
                print(f"    - {img}")

            # Save full list to file
            output_file = "/Users/yugambahl/Desktop/brain_ct/v4_missing_images.txt"
            with open(output_file, 'w') as f:
                f.write("="*80 + "\n")
                f.write(f"V4 MISSING IMAGES REPORT\n")
                f.write(f"Total missing: {len(missing):,} images\n")
                f.write("="*80 + "\n\n")
                for img in sorted(missing):
                    # Extract patientID_instanceNo for reference
                    normalized = normalize_filename(img)
                    f.write(f"{normalized:20s} | {img}\n")

            print(f"\n  Full list of missing images saved to: {output_file}")

    print("\nV4 Dataset Statistics:")
    print(f"  Total annotated images: {v4_total:,}")
    print(f"  Growth over V2+V3: {v4_total - len(all_v2_v3_images):,} images")

    # Check for overlap between V2 and V3
    print("\n" + "="*80)
    print("V2 vs V3 OVERLAP")
    print("="*80)

    # Normalize both sets for comparison
    normalized_v2 = {normalize_filename(img) for img in all_v2_images}
    normalized_v3 = {normalize_filename(img) for img in all_v3_images}

    overlap = normalized_v2 & normalized_v3
    v2_only = normalized_v2 - normalized_v3
    v3_only = normalized_v3 - normalized_v2

    print(f"\n  V2 only: {len(v2_only):,}")
    print(f"  V3 only: {len(v3_only):,}")
    print(f"  Overlap (in both): {len(overlap):,}")
    print(f"  Combined unique: {len(normalized_v2 | normalized_v3):,}")

    # Check if V2 is entirely contained in V3
    print("\n" + "="*80)
    print("V2 CONTAINMENT IN V3")
    print("="*80)

    v2_coverage_in_v3 = len(overlap) / len(normalized_v2) * 100 if len(normalized_v2) > 0 else 0
    v3_coverage_in_v2 = len(overlap) / len(normalized_v3) * 100 if len(normalized_v3) > 0 else 0

    print(f"\n  V2 images found in V3: {len(overlap):,} / {len(normalized_v2):,} ({v2_coverage_in_v3:.1f}%)")
    print(f"  V3 images found in V2: {len(overlap):,} / {len(normalized_v3):,} ({v3_coverage_in_v2:.1f}%)")

    if len(v2_only) == 0:
        print("\n  ✓ V2 is ENTIRELY contained in V3")
    else:
        print(f"\n  ✗ V2 is NOT entirely contained in V3")
        print(f"    V2 has {len(v2_only):,} unique images not in V3")

        # Save V2-only images to file
        v2_only_file = "/Users/yugambahl/Desktop/brain_ct/v2_only_images.txt"
        with open(v2_only_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"V2-ONLY IMAGES (not in V3)\n")
            f.write(f"Total: {len(v2_only):,} images\n")
            f.write("="*80 + "\n\n")

            # Get original filenames for V2-only images
            v2_only_original = {}
            for img in all_v2_images:
                normalized = normalize_filename(img)
                if normalized in v2_only:
                    if normalized not in v2_only_original:
                        v2_only_original[normalized] = []
                    v2_only_original[normalized].append(img)

            for normalized in sorted(v2_only):
                original_names = v2_only_original.get(normalized, [normalized])
                for orig in original_names:
                    f.write(f"{normalized:20s} | {orig}\n")

        print(f"    Full list saved to: {v2_only_file}")

        # Show sample of V2-only images
        print(f"\n    Sample V2-only images (showing up to 20):")
        for img in sorted(list(v2_only)[:20]):
            print(f"      - {img}")

    if len(v3_only) == 0:
        print("\n  ✓ V3 has no unique images (V3 ⊆ V2 or V3 = overlap)")
    else:
        print(f"\n  V3 has {len(v3_only):,} unique images not in V2")


if __name__ == '__main__':
    main()
