#!/usr/bin/env python3
"""
V4/V5 Dataset Validation Script

Comprehensive validation checks for the brain CT hemorrhage detection dataset.

Usage:
    python scripts/validate_v4_dataset.py --version v4
    python scripts/validate_v4_dataset.py --version v5 --dataset-type original
    python scripts/validate_v4_dataset.py --dataset-dir /path/to/custom/6class_coco
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict, Counter
from PIL import Image
from tqdm import tqdm
import re

# Try importing pycocotools (optional)
try:
    from pycocotools.coco import COCO
    PYCOCOTOOLS_AVAILABLE = True
except ImportError:
    PYCOCOTOOLS_AVAILABLE = False
    print("⚠️ pycocotools not available - skipping COCO format validation")

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


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text.center(80)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.END}\n")


def print_success(text: str):
    """Print a success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def print_error(text: str):
    """Print an error message."""
    print(f"{Colors.RED}✗ {text}{Colors.END}")


def print_warning(text: str):
    """Print a warning message."""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")


def load_coco_annotation(json_path: str) -> Dict:
    """Load COCO JSON annotation file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def validate_coco_format(json_path: str) -> Tuple[bool, str]:
    """
    Validate COCO format using pycocotools.

    Returns:
        (is_valid, error_message)
    """
    if not PYCOCOTOOLS_AVAILABLE:
        return True, "Skipped (pycocotools not available)"

    try:
        coco = COCO(json_path)
        return True, "Valid COCO format"
    except Exception as e:
        return False, f"Invalid COCO format: {str(e)}"


def validate_class_distribution(dataset_dir: str, expected_classes: int) -> Tuple[bool, Dict]:
    """
    Validate that all classes are present in all splits.

    Returns:
        (all_valid, stats_dict)
    """
    stats = {}
    all_valid = True

    for split in ['train', 'valid', 'test']:
        json_path = os.path.join(dataset_dir, split, '_annotations.coco.json')

        if not os.path.exists(json_path):
            print_warning(f"Split {split} not found at {json_path}")
            continue

        data = load_coco_annotation(json_path)

        # Count instances per class
        class_counts = Counter()
        for ann in data['annotations']:
            class_counts[ann['category_id']] += 1

        # Get unique classes
        unique_classes = set(class_counts.keys())

        stats[split] = {
            'total_images': len(data['images']),
            'total_annotations': len(data['annotations']),
            'unique_classes': unique_classes,
            'class_counts': dict(class_counts),
            'categories': {cat['id']: cat['name'] for cat in data['categories']}
        }

        # Check if all expected classes are present
        if len(unique_classes) < expected_classes:
            print_error(f"{split} split: Only {len(unique_classes)}/{expected_classes} classes present")
            all_valid = False
        else:
            print_success(f"{split} split: All {expected_classes} classes present")

    return all_valid, stats


def validate_patient_leakage(dataset_dir: str) -> Tuple[bool, Dict]:
    """
    Validate that there is no patient overlap between train/val/test splits.

    Returns:
        (no_leakage, patient_stats)
    """
    def extract_patient_id(filename: str) -> str:
        """Extract patient ID from filename."""
        base = os.path.splitext(filename)[0]
        if '.rf.' in base:
            base = base.split('.rf.')[0]

        for prefix in ['thin_', 'thick_', 'v3_', 'roboflow_']:
            if base.startswith(prefix):
                base = base[len(prefix):]

        match = re.match(r'^(\d+)', base)
        if match:
            return match.group(1)
        return base

    patient_sets = {}

    for split in ['train', 'valid', 'test']:
        json_path = os.path.join(dataset_dir, split, '_annotations.coco.json')

        if not os.path.exists(json_path):
            continue

        data = load_coco_annotation(json_path)

        patients = set()
        for img in data['images']:
            patient_id = extract_patient_id(img['file_name'])
            patients.add(patient_id)

        patient_sets[split] = patients
        print(f"  {split}: {len(patients)} unique patients")

    # Check for overlaps
    no_leakage = True

    if 'train' in patient_sets and 'valid' in patient_sets:
        overlap = patient_sets['train'] & patient_sets['valid']
        if overlap:
            print_error(f"Train/Valid overlap: {len(overlap)} patients")
            no_leakage = False
        else:
            print_success("No Train/Valid overlap")

    if 'train' in patient_sets and 'test' in patient_sets:
        overlap = patient_sets['train'] & patient_sets['test']
        if overlap:
            print_error(f"Train/Test overlap: {len(overlap)} patients")
            no_leakage = False
        else:
            print_success("No Train/Test overlap")

    if 'valid' in patient_sets and 'test' in patient_sets:
        overlap = patient_sets['valid'] & patient_sets['test']
        if overlap:
            print_error(f"Valid/Test overlap: {len(overlap)} patients")
            no_leakage = False
        else:
            print_success("No Valid/Test overlap")

    return no_leakage, patient_sets


def validate_image_integrity(dataset_dir: str, sample_size: int = 100) -> Tuple[bool, Dict]:
    """
    Validate that images are readable and have correct dimensions.

    Args:
        sample_size: Number of images to check per split (0 = check all)

    Returns:
        (all_valid, stats)
    """
    stats = {}
    all_valid = True

    for split in ['train', 'valid', 'test']:
        json_path = os.path.join(dataset_dir, split, '_annotations.coco.json')

        if not os.path.exists(json_path):
            continue

        data = load_coco_annotation(json_path)
        images_to_check = data['images']

        if sample_size > 0 and len(images_to_check) > sample_size:
            import random
            images_to_check = random.sample(images_to_check, sample_size)

        corrupt_count = 0
        dimension_mismatch_count = 0
        missing_count = 0

        for img_info in tqdm(images_to_check, desc=f"  Validating {split} images"):
            img_path = os.path.join(dataset_dir, split, img_info['file_name'])

            # Check if file exists
            if not os.path.exists(img_path):
                missing_count += 1
                continue

            # Try to open image
            try:
                with Image.open(img_path) as img:
                    width, height = img.size

                    # Check dimensions match annotation
                    if width != img_info['width'] or height != img_info['height']:
                        dimension_mismatch_count += 1

            except Exception as e:
                corrupt_count += 1

        stats[split] = {
            'checked': len(images_to_check),
            'corrupt': corrupt_count,
            'dimension_mismatch': dimension_mismatch_count,
            'missing': missing_count
        }

        if corrupt_count > 0 or missing_count > 0:
            print_error(f"{split}: {corrupt_count} corrupt, {missing_count} missing images")
            all_valid = False
        elif dimension_mismatch_count > 0:
            print_warning(f"{split}: {dimension_mismatch_count} dimension mismatches")
        else:
            print_success(f"{split}: All {len(images_to_check)} images valid")

    return all_valid, stats


def validate_bounding_boxes(dataset_dir: str) -> Tuple[bool, Dict]:
    """
    Validate that bounding boxes are within image bounds and have non-zero area.

    Returns:
        (all_valid, stats)
    """
    stats = {}
    all_valid = True

    for split in ['train', 'valid', 'test']:
        json_path = os.path.join(dataset_dir, split, '_annotations.coco.json')

        if not os.path.exists(json_path):
            continue

        data = load_coco_annotation(json_path)

        # Create image ID to dimensions mapping
        img_dims = {img['id']: (img['width'], img['height']) for img in data['images']}

        out_of_bounds_count = 0
        zero_area_count = 0

        for ann in data['annotations']:
            img_id = ann['image_id']
            bbox = ann['bbox']  # [x, y, width, height]

            if img_id not in img_dims:
                continue

            img_width, img_height = img_dims[img_id]

            # Check if bbox is within bounds
            x, y, w, h = bbox
            if x < 0 or y < 0 or x + w > img_width or y + h > img_height:
                out_of_bounds_count += 1

            # Check for zero area
            if w <= 0 or h <= 0:
                zero_area_count += 1

        stats[split] = {
            'total_annotations': len(data['annotations']),
            'out_of_bounds': out_of_bounds_count,
            'zero_area': zero_area_count
        }

        if out_of_bounds_count > 0 or zero_area_count > 0:
            print_error(f"{split}: {out_of_bounds_count} out-of-bounds, {zero_area_count} zero-area bboxes")
            all_valid = False
        else:
            print_success(f"{split}: All {len(data['annotations'])} bounding boxes valid")

    return all_valid, stats


def validate_augmentation(dataset_dir: str, expected_multiplier: int = 3) -> Tuple[bool, Dict]:
    """
    Validate that augmentation was applied correctly to training set.

    Args:
        expected_multiplier: Expected augmentation multiplier

    Returns:
        (correctly_augmented, stats)
    """
    train_json_path = os.path.join(dataset_dir, 'train', '_annotations.coco.json')

    if not os.path.exists(train_json_path):
        return False, {'error': 'Training split not found'}

    data = load_coco_annotation(train_json_path)

    # Count original vs augmented images
    original_count = 0
    augmented_count = 0

    for img in data['images']:
        filename = img['file_name']
        if '_aug' in filename:
            augmented_count += 1
        else:
            original_count += 1

    total_count = len(data['images'])
    expected_total = original_count * expected_multiplier

    stats = {
        'total_images': total_count,
        'original_images': original_count,
        'augmented_images': augmented_count,
        'expected_total': expected_total,
        'multiplier': total_count / original_count if original_count > 0 else 0
    }

    # Check if augmentation multiplier is correct (allow some tolerance)
    actual_multiplier = total_count / original_count if original_count > 0 else 0
    tolerance = 0.1  # 10% tolerance

    if abs(actual_multiplier - expected_multiplier) <= tolerance:
        print_success(f"Augmentation: {actual_multiplier:.1f}x multiplier (expected {expected_multiplier}x)")
        return True, stats
    else:
        print_warning(f"Augmentation: {actual_multiplier:.1f}x multiplier (expected {expected_multiplier}x)")
        return False, stats


def calculate_class_imbalance(stats: Dict) -> Dict:
    """Calculate class imbalance ratios."""
    imbalance_stats = {}

    for split, split_stats in stats.items():
        if 'class_counts' not in split_stats:
            continue

        counts = list(split_stats['class_counts'].values())
        if not counts:
            continue

        max_count = max(counts)
        min_count = min(counts)
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')

        imbalance_stats[split] = {
            'max_count': max_count,
            'min_count': min_count,
            'imbalance_ratio': imbalance_ratio
        }

        print(f"  {split}: {imbalance_ratio:.2f}x imbalance (max={max_count}, min={min_count})")

    return imbalance_stats


def validate_symlinks(dataset_dir: str) -> Tuple[bool, Dict]:
    """
    Validate symlink integrity (for 4-class filtered dataset).

    Returns:
        (all_valid, stats)
    """
    stats = {}
    all_valid = True

    for split in ['train', 'valid', 'test']:
        split_dir = os.path.join(dataset_dir, split)

        if not os.path.exists(split_dir):
            continue

        symlink_count = 0
        broken_symlink_count = 0
        regular_file_count = 0

        for filename in os.listdir(split_dir):
            if filename == '_annotations.coco.json':
                continue

            filepath = os.path.join(split_dir, filename)

            if os.path.islink(filepath):
                symlink_count += 1
                if not os.path.exists(filepath):
                    broken_symlink_count += 1
            elif os.path.isfile(filepath):
                regular_file_count += 1

        stats[split] = {
            'symlinks': symlink_count,
            'broken_symlinks': broken_symlink_count,
            'regular_files': regular_file_count
        }

        if broken_symlink_count > 0:
            print_error(f"{split}: {broken_symlink_count} broken symlinks")
            all_valid = False
        elif symlink_count > 0:
            print_success(f"{split}: {symlink_count} valid symlinks")

    return all_valid, stats


def main():
    parser = argparse.ArgumentParser(description='Validate V4/V5 dataset')
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
        '--dataset-dir',
        type=str,
        default=None,
        help='Path to dataset directory (overrides version/dataset-type)'
    )
    parser.add_argument(
        '--expected-classes',
        type=int,
        default=6,
        help='Expected number of classes (default: 6)'
    )
    parser.add_argument(
        '--skip-image-validation',
        action='store_true',
        help='Skip image integrity validation (faster)'
    )
    args = parser.parse_args()

    # Determine dataset directory
    if args.dataset_dir:
        dataset_dir = args.dataset_dir
    else:
        dataset_dir = VERSION_CONFIG[args.version][args.dataset_type]

    print_header(f"{args.version.upper()} DATASET VALIDATION ({args.dataset_type.upper()})")

    results = {}

    # [1] COCO Format Validation
    print_header("1. COCO Format Validation")
    for split in ['train', 'valid', 'test']:
        json_path = os.path.join(dataset_dir, split, '_annotations.coco.json')
        if os.path.exists(json_path):
            is_valid, message = validate_coco_format(json_path)
            results[f'coco_format_{split}'] = is_valid
            if is_valid:
                print_success(f"{split}: {message}")
            else:
                print_error(f"{split}: {message}")

    # [2] Class Distribution Validation
    print_header("2. Class Distribution Validation")
    all_classes_present, class_stats = validate_class_distribution(dataset_dir, args.expected_classes)
    results['class_distribution'] = all_classes_present

    # [3] Patient Leakage Detection
    print_header("3. Patient Leakage Detection")
    no_leakage, patient_stats = validate_patient_leakage(dataset_dir)
    results['no_patient_leakage'] = no_leakage

    # [4] Image Integrity Validation
    if not args.skip_image_validation:
        print_header("4. Image Integrity Validation")
        images_valid, image_stats = validate_image_integrity(dataset_dir, sample_size=100)
        results['image_integrity'] = images_valid
    else:
        print_header("4. Image Integrity Validation (SKIPPED)")

    # [5] Bounding Box Validation
    print_header("5. Bounding Box Validation")
    bboxes_valid, bbox_stats = validate_bounding_boxes(dataset_dir)
    results['bounding_boxes'] = bboxes_valid

    # [6] Augmentation Verification
    print_header("6. Augmentation Verification")
    augmentation_correct, aug_stats = validate_augmentation(dataset_dir, expected_multiplier=3)
    results['augmentation'] = augmentation_correct

    # [7] Class Imbalance Analysis
    print_header("7. Class Imbalance Analysis")
    imbalance_stats = calculate_class_imbalance(class_stats)

    # [8] Symlink Integrity (if applicable)
    if '4class' in dataset_dir:
        print_header("8. Symlink Integrity Validation")
        symlinks_valid, symlink_stats = validate_symlinks(dataset_dir)
        results['symlinks'] = symlinks_valid

    # Final Summary
    print_header("VALIDATION SUMMARY")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    print(f"\nPassed: {passed}/{total} checks\n")

    for check_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        color = Colors.GREEN if passed else Colors.RED
        print(f"{color}{status:6}{Colors.END} {check_name}")

    if passed == total:
        print_success(f"\n✓ All validation checks passed!")
        return 0
    else:
        print_error(f"\n✗ {total - passed} validation check(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())
