#!/usr/bin/env python3
"""
Export V4 Dataset to CSV

Generates a CSV file listing all images in the v4 dataset with metadata,
matching the format of negative_feedback_v3.csv.

Usage:
    python scripts/export_v4_dataset_csv.py
    python scripts/export_v4_dataset_csv.py --dataset-dir /path/to/v4/4class_coco --output data/metadata/v4_dataset_4class.csv
"""

import os
import json
import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict


def extract_patient_info(filename: str) -> tuple:
    """
    Extract patient_id and instance_no from filename.

    Formats handled:
    - v3_thick_520892552_14.png → (520892552, 14)
    - roboflow_525190930_29_png.rf.xxx.jpg → (525190930, 29)
    - v3_12345_6_aug0.jpg → (12345, 6)

    Returns:
        (patient_id, instance_no) as strings
    """
    # Remove extension
    base = os.path.splitext(filename)[0]

    # Remove Roboflow hash if present
    if '.rf.' in base:
        base = base.split('.rf.')[0]

    # Remove augmentation suffix (_aug0, _aug1, etc.)
    aug_match = re.search(r'_aug\d+$', base)
    if aug_match:
        base = base[:aug_match.start()]

    # Remove prefixes
    for prefix in ['v3_', 'roboflow_', 'thin_', 'thick_']:
        if base.startswith(prefix):
            base = base[len(prefix):]

    # Extract patient_id and instance_no
    # Pattern: patientid_instanceno or just patientid_number
    parts = base.split('_')

    if len(parts) >= 2:
        # Try to find two consecutive numeric parts
        for i in range(len(parts) - 1):
            if parts[i].isdigit() and parts[i+1].isdigit():
                return (parts[i], parts[i+1])

        # Fallback: first two parts
        if parts[0].isdigit():
            instance = parts[1] if parts[1].isdigit() else parts[-1]
            return (parts[0], instance)
    elif len(parts) == 1 and parts[0].isdigit():
        return (parts[0], "0")

    # Last resort: return the whole base name
    return (base, "0")


def load_coco_annotation(json_path: str) -> Dict:
    """Load COCO JSON annotation file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def determine_source(filename: str) -> str:
    """Determine if image is from v3 or roboflow."""
    if filename.startswith('v3_'):
        return 'v3'
    elif filename.startswith('roboflow_'):
        return 'roboflow'
    else:
        return 'unknown'


def export_dataset_to_csv(dataset_dir: str, output_path: str):
    """
    Export v4 dataset to CSV matching negative_feedback_v3.csv format.

    Args:
        dataset_dir: Path to v4 dataset directory (e.g., v4/6class_coco)
        output_path: Output CSV file path
    """
    print(f"Exporting dataset from: {dataset_dir}")
    print(f"Output CSV: {output_path}")

    # Create category name to ID mapping
    # Standard v4 6-class: EDH=0, HC=1, IPH=2, IVH=3, SAH=4, SDH=5
    # Standard v4 4-class: IPH=0, IVH=1, SAH=2, SDH=3

    # We'll read the actual categories from the first annotation file
    first_ann_file = os.path.join(dataset_dir, 'train', '_annotations.coco.json')
    if not os.path.exists(first_ann_file):
        first_ann_file = os.path.join(dataset_dir, 'valid', '_annotations.coco.json')
    if not os.path.exists(first_ann_file):
        first_ann_file = os.path.join(dataset_dir, 'test', '_annotations.coco.json')

    if not os.path.exists(first_ann_file):
        print(f"Error: No annotation files found in {dataset_dir}")
        return

    with open(first_ann_file, 'r') as f:
        sample_data = json.load(f)
        categories = {cat['id']: cat['name'] for cat in sample_data['categories']}

    print(f"  Dataset classes: {categories}")

    # Class name to column mapping
    class_to_column = {
        'EDH': 'has_epidural_hemorrhage',
        'HC': 'has_hemorrhage_contusion',
        'IPH': 'has_intraparenchymal_hemorrhage',
        'IVH': 'has_intraventricular_hemorrhage',
        'SAH': 'has_subarachnoid_hemorrhage',
        'SDH': 'has_subdural_hemorrhage'
    }

    # Collect all rows
    rows = []

    # Process each split
    for split in ['train', 'valid', 'test']:
        json_path = os.path.join(dataset_dir, split, '_annotations.coco.json')

        if not os.path.exists(json_path):
            print(f"  Skipping {split} (not found)")
            continue

        data = load_coco_annotation(json_path)
        print(f"  Processing {split}: {len(data['images'])} images")

        # Build image_id to annotations mapping
        image_annotations = defaultdict(list)
        for ann in data['annotations']:
            image_annotations[ann['image_id']].append(ann)

        # Process each image
        for img in data['images']:
            filename = img['file_name']
            patient_id, instance_no = extract_patient_info(filename)
            source = determine_source(filename)

            # Get annotations for this image
            anns = image_annotations.get(img['id'], [])

            # Determine which classes are present
            present_classes = set()
            for ann in anns:
                class_id = ann['category_id']
                if class_id in categories:
                    present_classes.add(categories[class_id])

            # Create row
            row = {
                'patient_id': patient_id,
                'instance_no': instance_no,
                'filename': filename,
                'split': split,
                'source': source,
                'has_intraparenchymal_hemorrhage': 1 if 'IPH' in present_classes else 0,
                'has_intraventricular_hemorrhage': 1 if 'IVH' in present_classes else 0,
                'has_subarachnoid_hemorrhage': 1 if 'SAH' in present_classes else 0,
                'has_epidural_hemorrhage': 1 if 'EDH' in present_classes else 0,
                'has_subdural_hemorrhage': 1 if 'SDH' in present_classes else 0,
                'has_hemorrhage_contusion': 1 if 'HC' in present_classes else 0,
                'has_no_hemorrhage': 1 if len(present_classes) == 0 else 0,
                'total_hemorrhages_on_image': len(anns),
                'modelVersion': 'v4'
            }

            rows.append(row)

    # Sort rows by patient_id and instance_no
    rows.sort(key=lambda x: (int(x['patient_id']) if x['patient_id'].isdigit() else x['patient_id'],
                             int(x['instance_no']) if x['instance_no'].isdigit() else x['instance_no']))

    # Write CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fieldnames = [
        'patient_id',
        'instance_no',
        'filename',
        'split',
        'source',
        'has_intraparenchymal_hemorrhage',
        'has_intraventricular_hemorrhage',
        'has_subarachnoid_hemorrhage',
        'has_epidural_hemorrhage',
        'has_subdural_hemorrhage',
        'has_hemorrhage_contusion',
        'has_no_hemorrhage',
        'total_hemorrhages_on_image',
        'modelVersion'
    ]

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✓ Exported {len(rows)} images to {output_path}")

    # Print summary
    print(f"\nSummary:")
    print(f"  Total images: {len(rows)}")

    split_counts = defaultdict(int)
    source_counts = defaultdict(int)
    hemorrhage_counts = defaultdict(int)

    for row in rows:
        split_counts[row['split']] += 1
        source_counts[row['source']] += 1

        for class_name in ['IPH', 'IVH', 'SAH', 'EDH', 'SDH', 'HC']:
            column = class_to_column[class_name]
            if row[column] == 1:
                hemorrhage_counts[class_name] += 1

    print(f"\n  By split:")
    for split, count in sorted(split_counts.items()):
        print(f"    {split}: {count}")

    print(f"\n  By source:")
    for source, count in sorted(source_counts.items()):
        print(f"    {source}: {count}")

    print(f"\n  By hemorrhage type:")
    for hem_type, count in sorted(hemorrhage_counts.items()):
        print(f"    {hem_type}: {count} images")

    no_hemorrhage = sum(1 for row in rows if row['has_no_hemorrhage'] == 1)
    print(f"    No hemorrhage: {no_hemorrhage} images")


def main():
    parser = argparse.ArgumentParser(description='Export V4 dataset to CSV')
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default='/Users/yugambahl/Desktop/brain_ct/data/training_datasets/v4/6class_coco',
        help='Path to v4 dataset directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='/Users/yugambahl/Desktop/brain_ct/data/metadata/v4_dataset_6class.csv',
        help='Output CSV file path'
    )
    args = parser.parse_args()

    export_dataset_to_csv(args.dataset_dir, args.output)


if __name__ == "__main__":
    main()
