#!/usr/bin/env python3
"""
Comprehensive Dataset Analysis Script (V4/V5)

Analyzes:
- Train/valid/test split counts (original vs augmented images separately)
- Class distribution across splits
- Data sources (V3/Roboflow/V4/V5/no-hem downloads)
- Augmentation statistics
- No-hemorrhage balance per split
"""

import os
import sys
import json
import argparse
from collections import defaultdict
from typing import Dict, Tuple, Set

# Version-specific configuration
VERSION_CONFIG = {
    'v4': {
        'original': '/Users/yugambahl/Desktop/brain_ct/data/training_datasets/v4/6class_coco_original',
        'augmented': '/Users/yugambahl/Desktop/brain_ct/data/training_datasets/v4/6class_coco',
    },
    'v5': {
        'original': '/Users/yugambahl/Desktop/brain_ct/data/training_datasets/v5/6class_coco_original',
        'augmented': '/Users/yugambahl/Desktop/brain_ct/data/training_datasets/v5/6class_coco',
    }
}


def is_augmented_image(filename: str) -> bool:
    """
    Check if image is augmented based on filename.

    Note: aug0, aug1, aug2 are ALL augmented versions.
    The original images (before augmentation) don't have _aug suffix in temp_combined,
    but after augmentation all images have _aug suffix.
    """
    return '_aug' in filename


def extract_source_from_filename(filename: str) -> str:
    """
    Extract data source from filename prefix.

    Returns: 'v3', 'roboflow', 'no_hem', or 'unknown'
    """
    base = os.path.splitext(filename)[0]

    # Remove augmentation suffix first
    if '_aug' in base:
        base = base.rsplit('_aug', 1)[0]

    # Check explicit prefixes first
    if base.startswith('v3_') or base.startswith('thin_') or base.startswith('thick_'):
        return 'v3'
    elif base.startswith('roboflow_'):
        return 'roboflow'
    elif base.startswith('no_hem_'):
        return 'no_hem'

    # Check if it's a pure numeric pattern (downloaded no-hem images)
    # Pattern: {patient_id}_{instance_no} where both are digits
    # Example: 591645860_2.png (no prefix)
    parts = base.split('_')
    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
        # This is a downloaded no-hem image without prefix
        return 'no_hem'

    # Truly unknown
    return 'unknown'


def extract_augmentation_index(filename: str) -> int:
    """
    Extract augmentation index from filename.

    Examples:
    - 520892552_14_aug0.jpg -> 0
    - 520892552_14_aug1.jpg -> 1
    - 520892552_14.jpg -> -1 (original)
    """
    if '_aug' not in filename:
        return -1

    try:
        base = os.path.splitext(filename)[0]
        aug_part = base.split('_aug')[-1]
        return int(aug_part)
    except:
        return -1


def analyze_coco_split(coco_json_path: str, split_name: str) -> Dict:
    """Analyze a single COCO split."""

    if not os.path.exists(coco_json_path):
        print(f"  ⚠️ File not found: {coco_json_path}")
        return None

    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # Build category mapping
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}

    # Initialize counters
    total_images = len(coco_data['images'])
    original_images = []
    augmented_images = []

    source_counts = defaultdict(int)
    class_counts = defaultdict(int)
    images_with_annotations = set()

    aug_index_counts = defaultdict(int)  # Track aug0, aug1, etc.

    # Analyze images
    for img in coco_data['images']:
        filename = img['file_name']
        source = extract_source_from_filename(filename)

        if is_augmented_image(filename):
            augmented_images.append(filename)
            aug_idx = extract_augmentation_index(filename)
            aug_index_counts[aug_idx] += 1
        else:
            original_images.append(filename)

        source_counts[source] += 1

    # Analyze annotations
    for ann in coco_data['annotations']:
        class_id = ann['category_id']
        class_name = categories.get(class_id, f'Unknown_{class_id}')
        class_counts[class_name] += 1
        images_with_annotations.add(ann['image_id'])

    # Calculate no-hemorrhage images
    all_image_ids = set(img['id'] for img in coco_data['images'])
    no_hem_image_ids = all_image_ids - images_with_annotations
    no_hem_count = len(no_hem_image_ids)
    with_hem_count = len(images_with_annotations)

    # Count original vs augmented for each category
    original_with_hem = 0
    original_no_hem = 0
    augmented_with_hem = 0
    augmented_no_hem = 0

    for img in coco_data['images']:
        is_aug = is_augmented_image(img['file_name'])
        has_hem = img['id'] in images_with_annotations

        if is_aug:
            if has_hem:
                augmented_with_hem += 1
            else:
                augmented_no_hem += 1
        else:
            if has_hem:
                original_with_hem += 1
            else:
                original_no_hem += 1

    return {
        'split_name': split_name,
        'total_images': total_images,
        'original_images': len(original_images),
        'augmented_images': len(augmented_images),
        'source_counts': dict(source_counts),
        'class_counts': dict(class_counts),
        'with_hem_count': with_hem_count,
        'no_hem_count': no_hem_count,
        'original_with_hem': original_with_hem,
        'original_no_hem': original_no_hem,
        'augmented_with_hem': augmented_with_hem,
        'augmented_no_hem': augmented_no_hem,
        'aug_index_counts': dict(aug_index_counts),
        'total_annotations': len(coco_data['annotations'])
    }


def print_split_analysis(stats: Dict):
    """Print detailed analysis for a split."""
    if stats is None:
        return

    split = stats['split_name']

    print(f"\n{'='*80}")
    print(f"{split.upper()} SPLIT ANALYSIS")
    print(f"{'='*80}")

    # Image counts
    print(f"\n📊 IMAGE COUNTS:")
    print(f"  Total images: {stats['total_images']:,}")

    # For train split, calculate original count from augmented count
    if split == 'train' and stats['augmented_images'] > 0:
        # Each original generates 3 augmented versions (aug0, aug1, aug2)
        calculated_original = stats['augmented_images'] // 3
        print(f"    Original (before augmentation): {calculated_original:,}")
        print(f"    Augmented total: {stats['augmented_images']:,} (3x multiplier)")
    else:
        print(f"    Original: {stats['original_images']:,}")
        print(f"    Augmented: {stats['augmented_images']:,}")

    if stats['augmented_images'] > 0 and stats['original_images'] > 0:
        multiplier = stats['augmented_images'] / stats['original_images']
        print(f"    Augmentation multiplier: {multiplier:.2f}x")

    # Augmentation breakdown
    if stats['aug_index_counts']:
        print(f"\n  Augmentation breakdown:")
        for aug_idx in sorted(stats['aug_index_counts'].keys()):
            count = stats['aug_index_counts'][aug_idx]
            print(f"    aug{aug_idx}: {count:,} images")

    # Source breakdown
    print(f"\n📁 DATA SOURCES:")
    for source in ['v3', 'roboflow', 'no_hem', 'unknown']:
        if source in stats['source_counts']:
            count = stats['source_counts'][source]
            pct = count / stats['total_images'] * 100
            print(f"  {source:12s}: {count:5,} images ({pct:5.1f}%)")

    # No-hemorrhage balance
    print(f"\n🩺 HEMORRHAGE BALANCE:")
    print(f"  With hemorrhage:    {stats['with_hem_count']:,} images")
    print(f"  No hemorrhage:      {stats['no_hem_count']:,} images")
    total_imgs = stats['with_hem_count'] + stats['no_hem_count']
    if total_imgs > 0:
        no_hem_pct = stats['no_hem_count'] / total_imgs * 100
        print(f"  No-hem percentage:  {no_hem_pct:.1f}%")

    # Original vs augmented breakdown
    print(f"\n  Original images:")
    print(f"    With hemorrhage: {stats['original_with_hem']:,}")
    print(f"    No hemorrhage:   {stats['original_no_hem']:,}")
    if stats['original_images'] > 0:
        orig_no_hem_pct = stats['original_no_hem'] / stats['original_images'] * 100
        print(f"    No-hem %: {orig_no_hem_pct:.1f}%")

    if stats['augmented_images'] > 0:
        print(f"\n  Augmented images:")
        print(f"    With hemorrhage: {stats['augmented_with_hem']:,}")
        print(f"    No hemorrhage:   {stats['augmented_no_hem']:,}")
        aug_no_hem_pct = stats['augmented_no_hem'] / stats['augmented_images'] * 100
        print(f"    No-hem %: {aug_no_hem_pct:.1f}%")

    # Class distribution
    print(f"\n🔍 CLASS DISTRIBUTION ({stats['total_annotations']:,} total annotations):")

    # Sort classes by count (descending)
    sorted_classes = sorted(stats['class_counts'].items(), key=lambda x: x[1], reverse=True)

    for class_name, count in sorted_classes:
        pct = count / max(stats['total_annotations'], 1) * 100
        print(f"  {class_name:6s}: {count:5,} annotations ({pct:5.1f}%)")


def print_overall_summary(train_stats: Dict, val_stats: Dict, test_stats: Dict):
    """Print overall dataset summary."""

    print(f"\n\n{'='*80}")
    print("OVERALL DATASET SUMMARY")
    print(f"{'='*80}")

    # Combine stats
    all_stats = [s for s in [train_stats, val_stats, test_stats] if s is not None]

    if not all_stats:
        print("No data available")
        return

    total_images = sum(s['total_images'] for s in all_stats)

    # Calculate original count correctly - train images are all augmented
    train_original_count = train_stats['augmented_images'] // 3 if train_stats and train_stats['augmented_images'] > 0 else 0
    val_original_count = val_stats['original_images'] if val_stats else 0
    test_original_count = test_stats['original_images'] if test_stats else 0
    total_original = train_original_count + val_original_count + test_original_count

    total_augmented = sum(s['augmented_images'] for s in all_stats)

    print(f"\n📊 TOTAL IMAGES: {total_images:,}")
    print(f"  Original (before augmentation): {total_original:,}")
    print(f"  After augmentation: {total_images:,}")
    print(f"    Train (3x): {train_stats['total_images']:,}" if train_stats else "")
    print(f"    Valid (no aug): {val_stats['total_images']:,}" if val_stats else "")
    print(f"    Test (no aug): {test_stats['total_images']:,}" if test_stats else "")

    if total_original > 0:
        overall_multiplier = total_images / total_original
        print(f"  Overall expansion: {overall_multiplier:.2f}x")

    # Source totals (ORIGINAL IMAGES ONLY - exclude augmented train images)
    print(f"\n📁 DATA SOURCES (original images only, before augmentation):")
    source_totals_original = defaultdict(int)

    # For train: divide by 3 to get original count per source
    if train_stats:
        for source, count in train_stats['source_counts'].items():
            source_totals_original[source] += count // 3

    # For valid and test: use actual counts (no augmentation)
    for stats in [val_stats, test_stats]:
        if stats:
            for source, count in stats['source_counts'].items():
                source_totals_original[source] += count

    for source in ['v3', 'roboflow', 'no_hem', 'unknown']:
        if source in source_totals_original:
            count = source_totals_original[source]
            pct = count / total_original * 100
            print(f"  {source:12s}: {count:6,} images ({pct:5.1f}%)")

    # Class totals (ORIGINAL IMAGES ONLY - divide train by 3)
    print(f"\n🔍 CLASS DISTRIBUTION (original images only, before augmentation):")
    class_totals_original = defaultdict(int)
    total_annotations_original = 0

    # Train: divide by 3 to get original annotation count
    if train_stats:
        for class_name, count in train_stats['class_counts'].items():
            class_totals_original[class_name] += count // 3
        total_annotations_original += train_stats['total_annotations'] // 3

    # Valid and test: use actual counts
    for stats in [val_stats, test_stats]:
        if stats:
            for class_name, count in stats['class_counts'].items():
                class_totals_original[class_name] += count
            total_annotations_original += stats['total_annotations']

    sorted_classes = sorted(class_totals_original.items(), key=lambda x: x[1], reverse=True)

    print(f"  Total annotations: {total_annotations_original:,}")
    for class_name, count in sorted_classes:
        pct = count / max(total_annotations_original, 1) * 100
        print(f"  {class_name:6s}: {count:6,} annotations ({pct:5.1f}%)")

    # Hemorrhage balance (ORIGINAL IMAGES ONLY)
    # Calculate from original images in each split
    train_original_with_hem = train_stats['with_hem_count'] // 3 if train_stats else 0
    train_original_no_hem = train_stats['no_hem_count'] // 3 if train_stats else 0

    val_with_hem = val_stats['with_hem_count'] if val_stats else 0
    val_no_hem = val_stats['no_hem_count'] if val_stats else 0

    test_with_hem = test_stats['with_hem_count'] if test_stats else 0
    test_no_hem = test_stats['no_hem_count'] if test_stats else 0

    total_original_with_hem = train_original_with_hem + val_with_hem + test_with_hem
    total_original_no_hem = train_original_no_hem + val_no_hem + test_no_hem

    print(f"\n🩺 OVERALL HEMORRHAGE BALANCE (original images only):")
    print(f"  With hemorrhage: {total_original_with_hem:,} images")
    print(f"  No hemorrhage:   {total_original_no_hem:,} images")
    total_original_all = total_original_with_hem + total_original_no_hem
    if total_original_all > 0:
        no_hem_pct_original = total_original_no_hem / total_original_all * 100
        print(f"  No-hem %: {no_hem_pct_original:.1f}%")


    # Split ratios
    print(f"\n📈 SPLIT RATIOS (based on original images before augmentation):")
    if total_original > 0:
        train_pct = train_original_count / total_original * 100
        val_pct = val_original_count / total_original * 100
        test_pct = test_original_count / total_original * 100
        print(f"  Train: {train_pct:.1f}% ({train_original_count:,} original → {train_stats['total_images']:,} after 3x aug)" if train_stats else "")
        print(f"  Valid: {val_pct:.1f}% ({val_original_count:,} images)")
        print(f"  Test:  {test_pct:.1f}% ({test_original_count:,} images)")


def print_augmentation_info():
    """Print information about augmentations applied."""

    print(f"\n\n{'='*80}")
    print("AUGMENTATION DETAILS")
    print(f"{'='*80}")

    print("""
The following augmentations were applied to the TRAINING set only:

🔄 Geometric Transformations:
  - HorizontalFlip (p=0.5)
  - ShiftScaleRotate (shift=0.0625, scale=0.1, rotate=15°, p=0.5)
  - PadIfNeeded (min_height=512, min_width=512)

🎨 Color/Intensity Adjustments:
  - RandomBrightnessContrast (brightness=0.2, contrast=0.2, p=0.3)
  - CLAHE (clip_limit=4.0, p=0.3) - Adaptive histogram equalization
  - GaussNoise (var_limit=(10.0, 50.0), p=0.3)

💪 Multiplier: 3x (generates 2 augmented versions per original image)
  - aug0: First augmented version
  - aug1: Second augmented version

⚠️  Valid and Test splits: NO augmentation (original images only)

📋 Bounding Box Handling:
  - Format: COCO (x, y, width, height)
  - Min visibility: 30% (boxes <30% visible are filtered)
  - Post-augmentation clipping: Applied to ensure boxes stay within image bounds
""")


def main():
    """Main analysis function."""

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Analyze COCO format dataset with train/valid/test splits (V4/V5)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze V4 dataset (augmented)
  python scripts/analyze_v4_dataset.py --version v4 --dataset-type augmented

  # Analyze V5 dataset (original only)
  python scripts/analyze_v4_dataset.py --version v5 --dataset-type original

  # Use custom path
  python scripts/analyze_v4_dataset.py --dataset-path /path/to/custom/6class_coco
        """
    )
    parser.add_argument(
        '--version',
        type=str,
        default='v4',
        choices=['v4', 'v5'],
        help='Dataset version (default: v4)'
    )
    parser.add_argument(
        '--dataset-type',
        type=str,
        default='augmented',
        choices=['original', 'augmented'],
        help='Dataset type (default: augmented)'
    )
    parser.add_argument(
        '--dataset-path',
        type=str,
        default=None,
        help='Path to dataset directory (overrides version/dataset-type)'
    )

    args = parser.parse_args()

    # Determine dataset directory
    if args.dataset_path:
        dataset_dir = args.dataset_path
    else:
        dataset_dir = VERSION_CONFIG[args.version][args.dataset_type]

    # Validate dataset directory exists
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory not found: {dataset_dir}")
        sys.exit(1)

    # Extract dataset name from path for display
    dataset_name = os.path.basename(os.path.dirname(dataset_dir))
    if dataset_name == 'training_datasets':
        dataset_name = os.path.basename(dataset_dir)

    print("="*80)
    print(f"DATASET COMPREHENSIVE ANALYSIS: {dataset_name.upper()}")
    print(f"Path: {dataset_dir}")
    print("="*80)

    # Analyze each split
    train_stats = analyze_coco_split(
        os.path.join(dataset_dir, 'train', '_annotations.coco.json'),
        'train'
    )

    val_stats = analyze_coco_split(
        os.path.join(dataset_dir, 'valid', '_annotations.coco.json'),
        'valid'
    )

    test_stats = analyze_coco_split(
        os.path.join(dataset_dir, 'test', '_annotations.coco.json'),
        'test'
    )

    # Print individual split analyses
    print_split_analysis(train_stats)
    print_split_analysis(val_stats)
    print_split_analysis(test_stats)

    # Print overall summary
    print_overall_summary(train_stats, val_stats, test_stats)

    # Print augmentation info
    print_augmentation_info()

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}\n")

    # Return stats for potential programmatic use
    return {
        'train': train_stats,
        'valid': val_stats,
        'test': test_stats
    }


if __name__ == "__main__":
    main()
