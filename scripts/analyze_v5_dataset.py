#!/usr/bin/env python3
"""
V5 Dataset Analysis Script

Analyzes the V5 combined dataset (original and augmented versions).
Generates statistics, class distribution, and comparison reports.

Usage:
    python scripts/analyze_v5_dataset.py
    python scripts/analyze_v5_dataset.py --augmented-only
"""

import os
import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import yaml

# Dataset paths
V5_ORIGINAL_DIR = "/Users/yugambahl/Desktop/brain_ct/data/training_datasets/v5/roboflow_original_images"
V5_AUGMENTED_DIR = "/Users/yugambahl/Desktop/brain_ct/data/training_datasets/v5/roboflow_augmented_3x"

# Class names
CLASS_NAMES = ['EDH', 'HC', 'IPH', 'IVH', 'SAH', 'SDH']


def load_coco_annotations(json_path: str) -> Dict:
    """Load COCO annotations from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def analyze_coco_split(coco_data: Dict, split_name: str) -> Dict:
    """Analyze a single COCO split."""
    images = coco_data['images']
    annotations = coco_data['annotations']
    categories = coco_data.get('categories', [])

    # Basic counts
    num_images = len(images)
    num_annotations = len(annotations)

    # Images with/without annotations
    images_with_ann = set(ann['image_id'] for ann in annotations)
    num_positive = len(images_with_ann)
    num_negative = num_images - num_positive

    # Class distribution
    class_counts = Counter()
    for ann in annotations:
        cat_id = int(ann['category_id'])  # Convert to int in case it's a float
        if 0 <= cat_id < len(CLASS_NAMES):
            class_counts[CLASS_NAMES[cat_id]] += 1

    # Annotations per image statistics
    ann_per_image = Counter()
    for ann in annotations:
        ann_per_image[ann['image_id']] += 1

    if ann_per_image:
        avg_ann_per_positive = sum(ann_per_image.values()) / len(ann_per_image)
        max_ann_per_image = max(ann_per_image.values())
    else:
        avg_ann_per_positive = 0
        max_ann_per_image = 0

    return {
        'split': split_name,
        'images': num_images,
        'annotations': num_annotations,
        'positive_images': num_positive,
        'negative_images': num_negative,
        'negative_ratio': num_negative / num_images * 100 if num_images > 0 else 0,
        'class_distribution': dict(class_counts),
        'avg_annotations_per_positive': round(avg_ann_per_positive, 2),
        'max_annotations_per_image': max_ann_per_image,
    }


def analyze_dataset(dataset_dir: str, dataset_name: str) -> Dict:
    """Analyze entire dataset (all splits)."""
    print(f"\n{'='*60}")
    print(f"ANALYZING: {dataset_name}")
    print(f"Path: {dataset_dir}")
    print(f"{'='*60}")

    results = {
        'name': dataset_name,
        'path': dataset_dir,
        'splits': {},
        'total': {
            'images': 0,
            'annotations': 0,
            'positive_images': 0,
            'negative_images': 0,
            'class_distribution': Counter(),
        }
    }

    for split in ['train', 'valid', 'test']:
        split_dir = Path(dataset_dir) / split
        ann_file = split_dir / '_annotations.coco.json'

        if not ann_file.exists():
            print(f"  {split}: Not found")
            continue

        coco_data = load_coco_annotations(str(ann_file))
        split_stats = analyze_coco_split(coco_data, split)
        results['splits'][split] = split_stats

        # Accumulate totals
        results['total']['images'] += split_stats['images']
        results['total']['annotations'] += split_stats['annotations']
        results['total']['positive_images'] += split_stats['positive_images']
        results['total']['negative_images'] += split_stats['negative_images']
        results['total']['class_distribution'].update(split_stats['class_distribution'])

        # Print split stats
        print(f"\n  {split.upper()} Split:")
        print(f"    Images: {split_stats['images']:,}")
        print(f"    Annotations: {split_stats['annotations']:,}")
        print(f"    Positive (with hemorrhage): {split_stats['positive_images']:,}")
        print(f"    Negative (no hemorrhage): {split_stats['negative_images']:,} ({split_stats['negative_ratio']:.1f}%)")
        print(f"    Avg annotations/positive: {split_stats['avg_annotations_per_positive']:.2f}")
        print(f"    Class distribution: {split_stats['class_distribution']}")

    # Convert Counter to dict for total
    results['total']['class_distribution'] = dict(results['total']['class_distribution'])

    # Print totals
    total = results['total']
    print(f"\n  TOTAL:")
    print(f"    Images: {total['images']:,}")
    print(f"    Annotations: {total['annotations']:,}")
    print(f"    Positive: {total['positive_images']:,}")
    print(f"    Negative: {total['negative_images']:,}")
    print(f"    Class distribution: {total['class_distribution']}")

    return results


def compare_datasets(original: Dict, augmented: Dict) -> None:
    """Compare original and augmented dataset statistics."""
    print(f"\n{'='*60}")
    print("DATASET COMPARISON: ORIGINAL vs AUGMENTED")
    print(f"{'='*60}")

    # Calculate multiplier
    if 'train' in original['splits'] and 'train' in augmented['splits']:
        orig_train = original['splits']['train']['images']
        aug_train = augmented['splits']['train']['images']
        actual_multiplier = aug_train / orig_train if orig_train > 0 else 0
        print(f"\n  Effective augmentation multiplier: {actual_multiplier:.2f}x")

    print(f"\n  {'Metric':<30} {'Original':>12} {'Augmented':>12} {'Change':>10}")
    print(f"  {'-'*64}")

    metrics = [
        ('Total Images', 'images'),
        ('Total Annotations', 'annotations'),
        ('Positive Images', 'positive_images'),
        ('Negative Images', 'negative_images'),
    ]

    for metric_name, key in metrics:
        orig_val = original['total'].get(key, 0)
        aug_val = augmented['total'].get(key, 0)
        change = aug_val - orig_val
        change_pct = (change / orig_val * 100) if orig_val > 0 else 0
        print(f"  {metric_name:<30} {orig_val:>12,} {aug_val:>12,} {change_pct:>+9.1f}%")

    print(f"\n  Class Distribution Comparison:")
    print(f"  {'Class':<10} {'Original':>12} {'Augmented':>12} {'Multiplier':>12}")
    print(f"  {'-'*46}")

    for cls in CLASS_NAMES:
        orig_count = original['total']['class_distribution'].get(cls, 0)
        aug_count = augmented['total']['class_distribution'].get(cls, 0)
        multiplier = aug_count / orig_count if orig_count > 0 else 0
        print(f"  {cls:<10} {orig_count:>12,} {aug_count:>12,} {multiplier:>11.2f}x")


def print_training_summary(augmented: Dict) -> None:
    """Print summary suitable for training configuration."""
    print(f"\n{'='*60}")
    print("TRAINING CONFIGURATION SUMMARY")
    print(f"{'='*60}")

    if 'train' in augmented['splits']:
        train = augmented['splits']['train']
        valid = augmented['splits'].get('valid', {})

        print(f"\n  Dataset: V5 Augmented 3x")
        print(f"  Classes: {len(CLASS_NAMES)} ({', '.join(CLASS_NAMES)})")
        print(f"\n  Training:")
        print(f"    Images: {train['images']:,}")
        print(f"    Annotations: {train['annotations']:,}")
        print(f"    Positive ratio: {100 - train['negative_ratio']:.1f}%")
        print(f"\n  Validation:")
        print(f"    Images: {valid.get('images', 0):,}")
        print(f"    Annotations: {valid.get('annotations', 0):,}")

        print(f"\n  Class imbalance (train):")
        class_dist = train['class_distribution']
        if class_dist:
            max_class = max(class_dist.values())
            for cls in CLASS_NAMES:
                count = class_dist.get(cls, 0)
                ratio = max_class / count if count > 0 else float('inf')
                bar = '#' * int(count / max_class * 30)
                print(f"    {cls}: {count:>6,} ({ratio:>5.1f}x imbalance) {bar}")


def main():
    parser = argparse.ArgumentParser(description='Analyze V5 dataset')
    parser.add_argument('--augmented-only', action='store_true',
                        help='Only analyze augmented dataset')
    parser.add_argument('--original-only', action='store_true',
                        help='Only analyze original dataset')
    args = parser.parse_args()

    print("\n" + "="*60)
    print("V5 DATASET ANALYSIS")
    print("="*60)

    original_results = None
    augmented_results = None

    # Analyze original dataset
    if not args.augmented_only:
        if Path(V5_ORIGINAL_DIR).exists():
            original_results = analyze_dataset(V5_ORIGINAL_DIR, "V5 Original")
        else:
            print(f"\n  Original dataset not found: {V5_ORIGINAL_DIR}")

    # Analyze augmented dataset
    if not args.original_only:
        if Path(V5_AUGMENTED_DIR).exists():
            augmented_results = analyze_dataset(V5_AUGMENTED_DIR, "V5 Augmented 3x")
        else:
            print(f"\n  Augmented dataset not found: {V5_AUGMENTED_DIR}")

    # Compare datasets
    if original_results and augmented_results:
        compare_datasets(original_results, augmented_results)

    # Print training summary
    if augmented_results:
        print_training_summary(augmented_results)

    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
