#!/usr/bin/env python3
"""
Brain CT Hemorrhage Detection - V4 Dataset Preparation Pipeline

Combines V3 (4-class filtered) with Roboflow V3 (8 categories) into unified 6-class dataset.
Performs patient-level splitting, 3x augmentation, and generates both 6-class and 4-class versions.

Usage:
    python scripts/prepare_v4_dataset.py
    python scripts/prepare_v4_dataset.py --skip-augmentation  # For testing
    python scripts/prepare_v4_dataset.py --dry-run  # Validation only
"""

import os
import json
import shutil
import random
import re
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, Counter
from tqdm import tqdm
from PIL import Image
import argparse
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input dataset paths
V3_COCO_DIR = "/Users/yugambahl/Desktop/brain_ct/data/training_datasets/v3/filtered_4class/coco"
ROBOFLOW_COCO_DIR = "/Users/yugambahl/Desktop/brain_ct/data/training_datasets/roboflow_downloads/UAT_CT_BRAIN_HEMORRHAGE_V3.v3i.coco"
NO_HEM_DOWNLOADS_DIR = "/Users/yugambahl/Desktop/brain_ct/data/training_datasets/no_hemorrhage_downloads"

# Output paths
V4_BASE_DIR = "/Users/yugambahl/Desktop/brain_ct/data/training_datasets/v4"
V4_6CLASS_DIR = f"{V4_BASE_DIR}/6class_coco"
V4_4CLASS_DIR = f"{V4_BASE_DIR}/4class_coco"
TEMP_COMBINED_DIR = f"{V4_BASE_DIR}/temp_combined"

# Dataset split ratios
TRAIN_RATIO = 0.80
VAL_RATIO = 0.15
TEST_RATIO = 0.05

# Augmentation settings
AUGMENTATION_MULTIPLIER = 3
AUGMENTATION_STRENGTH = 'aggressive'
IMAGE_SIZE = 640

# Seed for reproducibility
RANDOM_SEED = 42

# Class mappings
V3_TO_V4_MAPPING = {
    0: 2,  # IPH (v3:0) → IPH (v4:2)
    1: 3,  # IVH (v3:1) → IVH (v4:3)
    2: 4,  # SAH (v3:2) → SAH (v4:4)
    3: 5,  # SDH (v3:3) → SDH (v4:5)
}

ROBOFLOW_TO_V4_MAPPING = {
    0: None,  # "-" → SKIP
    1: None,  # "0" → SKIP
    2: 0,     # EDH (robo:2) → EDH (v4:0)
    3: 1,     # HC (robo:3) → HC (v4:1)
    4: 2,     # IPH (robo:4) → IPH (v4:2)
    5: 3,     # IVH (robo:5) → IVH (v4:3)
    6: 4,     # SAH (robo:6) → SAH (v4:4)
    7: 5,     # SDH (robo:7) → SDH (v4:5)
}

V4_6CLASS_TO_4CLASS_MAPPING = {
    0: None,  # EDH → FILTER OUT
    1: None,  # HC → FILTER OUT
    2: 0,     # IPH (v4:2) → 4class:0
    3: 1,     # IVH (v4:3) → 4class:1
    4: 2,     # SAH (v4:4) → 4class:2
    5: 3,     # SDH (v4:5) → 4class:3
}

V4_6CLASS_NAMES = ['EDH', 'HC', 'IPH', 'IVH', 'SAH', 'SDH']
V4_4CLASS_NAMES = ['IPH', 'IVH', 'SAH', 'SDH']

# V5-specific category mapping (8 categories with different IPH duplicate at id 0)
V5_ROBOFLOW_TO_V4_MAPPING = {
    0: 2,     # IPH (v5:0) → IPH (v4:2) [duplicate IPH]
    1: None,  # "0" noise → SKIP
    2: 0,     # EDH (v5:2) → EDH (v4:0)
    3: 1,     # HC (v5:3) → HC (v4:1)
    4: 2,     # IPH (v5:4) → IPH (v4:2)
    5: 3,     # IVH (v5:5) → IVH (v4:3)
    6: 4,     # SAH (v5:6) → SAH (v4:4)
    7: 5,     # SDH (v5:7) → SDH (v4:5)
}

# Version-specific configuration
VERSION_CONFIG = {
    'v4': {
        'sources': {
            'v3': V3_COCO_DIR,
            'roboflow': ROBOFLOW_COCO_DIR,
            'no_hem': NO_HEM_DOWNLOADS_DIR,
        },
        'output_base': V4_BASE_DIR,
        'category_mappings': {
            'v3': V3_TO_V4_MAPPING,
            'roboflow': ROBOFLOW_TO_V4_MAPPING,
        },
        'prefixes': ['v3', 'roboflow', 'no_hem'],
    },
    'v5': {
        'sources': {
            'v4': '/Users/yugambahl/Desktop/brain_ct/data/training_datasets/v4/6class_coco',
            'v5_roboflow': '/Users/yugambahl/Desktop/brain_ct/data/training_datasets/v5/train',
            'no_hem': '/Users/yugambahl/Desktop/brain_ct/data/training_datasets/v5_no_hemorrhage_downloads',
        },
        'output_base': '/Users/yugambahl/Desktop/brain_ct/data/training_datasets/v5',
        'category_mappings': {
            'v5_roboflow': V5_ROBOFLOW_TO_V4_MAPPING,
        },
        'prefixes': ['v4', 'v5', 'no_hem'],
        'v4_augmentation_reverse': True,  # Keep only aug0 from V4 train
    }
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def log(message: str):
    """Print a timestamped log message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def load_coco_dataset(json_path: str, split_name: str, image_base_dir: str = None) -> Dict:
    """
    Load COCO JSON annotations.

    Args:
        json_path: Path to _annotations.coco.json
        split_name: Name for logging (e.g., "v3_train", "roboflow")
        image_base_dir: Base directory containing images (if different from JSON location)

    Returns:
        Dict with 'images', 'annotations', 'categories', 'image_paths'
    """
    log(f"  Loading {split_name} from {json_path}")

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Determine image directory
    if image_base_dir is None:
        image_base_dir = os.path.dirname(json_path)

    # Build map of filename → full path
    image_paths = {}
    missing_count = 0

    for img in data['images']:
        filename = img['file_name']
        # Try multiple possible locations
        possible_paths = [
            os.path.join(image_base_dir, filename),
            os.path.join(image_base_dir, 'images', filename),
            os.path.join(os.path.dirname(image_base_dir), filename)
        ]

        found = False
        for path in possible_paths:
            # Check if it's a symlink (even if broken)
            if os.path.islink(path):
                # Try to read the symlink target
                try:
                    target = os.readlink(path)
                    # If target is relative, make it absolute
                    if not os.path.isabs(target):
                        target = os.path.join(os.path.dirname(path), target)

                    # Check if target exists
                    if os.path.exists(target):
                        image_paths[filename] = target
                        found = True
                        break
                    else:
                        # Symlink is broken - try to fix the path
                        # V3 symlinks point to filtered_4class but should point to v3/filtered_4class
                        if '/filtered_4class/' in target:
                            fixed_target = target.replace('/filtered_4class/', '/v3/filtered_4class/')
                            if os.path.exists(fixed_target):
                                image_paths[filename] = fixed_target
                                found = True
                                break
                except OSError:
                    pass
            elif os.path.exists(path):
                # Regular file
                image_paths[filename] = path
                found = True
                break

        if not found:
            missing_count += 1

    if missing_count > 0:
        log(f"    ⚠️ {missing_count} images not found (will be skipped)")

    log(f"    Loaded {len(data['images'])} images, {len(data['annotations'])} annotations, {len(image_paths)} files found")

    data['image_paths'] = image_paths
    return data


def remap_annotations(annotations: List[Dict], mapping: Dict[int, Optional[int]]) -> Tuple[List[Dict], int]:
    """
    Remap annotation category IDs and filter out None mappings.

    Returns:
        (remapped_annotations, filtered_count)
    """
    remapped = []
    filtered = 0

    for ann in annotations:
        old_cat_id = ann['category_id']
        new_cat_id = mapping.get(old_cat_id)

        if new_cat_id is None:
            filtered += 1
            continue

        ann_copy = ann.copy()
        ann_copy['category_id'] = new_cat_id
        remapped.append(ann_copy)

    return remapped, filtered


def fix_out_of_bounds_bboxes(coco_data: Dict) -> Dict:
    """
    Fix bounding boxes that extend beyond image boundaries.

    Clips bboxes to [0, 0, width, height] and logs warnings for fixed boxes.
    """
    # Build image size map
    image_sizes = {}
    for img in coco_data['images']:
        image_sizes[img['id']] = (img['width'], img['height'])

    fixed_count = 0
    fixed_annotations = []

    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in image_sizes:
            fixed_annotations.append(ann)
            continue

        img_width, img_height = image_sizes[img_id]
        bbox = ann['bbox']  # [x, y, width, height] in pixels

        x, y, w, h = bbox
        x_max = x + w
        y_max = y + h

        # Check if bbox needs fixing
        needs_fix = (x < 0 or y < 0 or x_max > img_width or y_max > img_height)

        if needs_fix:
            # Clip to image boundaries
            x_clipped = max(0, x)
            y_clipped = max(0, y)
            x_max_clipped = min(img_width, x_max)
            y_max_clipped = min(img_height, y_max)

            w_clipped = x_max_clipped - x_clipped
            h_clipped = y_max_clipped - y_clipped

            # Skip if clipping results in zero-area bbox
            if w_clipped <= 0 or h_clipped <= 0:
                log(f"    ⚠️ Skipping zero-area bbox after clipping: img_id={img_id}, original_bbox={bbox}")
                continue

            ann_copy = ann.copy()
            ann_copy['bbox'] = [x_clipped, y_clipped, w_clipped, h_clipped]
            fixed_annotations.append(ann_copy)
            fixed_count += 1
        else:
            fixed_annotations.append(ann)

    if fixed_count > 0:
        log(f"  Fixed {fixed_count} out-of-bounds bounding boxes")

    coco_data['annotations'] = fixed_annotations
    return coco_data


def merge_coco_datasets(
    datasets: List[Dict],
    unified_categories: List[Dict],
    dataset_prefixes: List[str] = None
) -> Dict:
    """
    Merge multiple COCO datasets with sequential ID renumbering.

    Args:
        datasets: List of COCO dicts to merge
        unified_categories: Target category schema
        dataset_prefixes: Optional prefixes for filenames to prevent collisions

    Returns:
        Merged COCO dict
    """
    merged = {
        'images': [],
        'annotations': [],
        'categories': unified_categories,
        'info': {
            'description': 'V4 Brain CT Hemorrhage Dataset - Combined',
            'version': '4.0',
            'year': 2025,
        },
        'image_paths': {}
    }

    next_image_id = 1
    next_ann_id = 1

    for dataset_idx, dataset in enumerate(datasets):
        prefix = dataset_prefixes[dataset_idx] if dataset_prefixes else ''
        old_to_new_img_id = {}

        log(f"  Merging dataset {dataset_idx + 1}/{len(datasets)}...")

        # Merge images with optional prefix
        for img in dataset['images']:
            old_img_id = img['id']
            new_img_id = next_image_id
            old_to_new_img_id[old_img_id] = new_img_id

            img_copy = img.copy()
            img_copy['id'] = new_img_id

            # Add prefix to filename if provided
            original_filename = img_copy['file_name']
            if prefix:
                img_copy['file_name'] = f"{prefix}_{original_filename}"

            # Update image paths mapping
            if original_filename in dataset.get('image_paths', {}):
                merged['image_paths'][img_copy['file_name']] = dataset['image_paths'][original_filename]

            merged['images'].append(img_copy)
            next_image_id += 1

        # Merge annotations
        for ann in dataset['annotations']:
            old_img_id = ann['image_id']
            if old_img_id not in old_to_new_img_id:
                continue  # Skip annotations for missing images

            ann_copy = ann.copy()
            ann_copy['id'] = next_ann_id
            ann_copy['image_id'] = old_to_new_img_id[old_img_id]
            merged['annotations'].append(ann_copy)
            next_ann_id += 1

    log(f"  Merged: {len(merged['images'])} images, {len(merged['annotations'])} annotations")
    return merged


def extract_patient_id(filename: str) -> str:
    """
    Extract patient ID from various filename formats.

    Formats handled:
    - Roboflow: 525190930_29_png.rf.*.jpg → 525190930
    - V3: v3_12345_slice_10.jpg → 12345
    - V3 augmented: v3_12345_6_aug0.jpg → 12345
    - V3 with thick/thin: v3_thick_12345_6.png → 12345
    - V5: v5_12345_6_png.jpg → 12345
    """
    # Remove extension and Roboflow hash
    base = os.path.splitext(filename)[0]
    if '.rf.' in base:
        base = base.split('.rf.')[0]

    # Remove augmentation suffix (_aug0, _aug1, etc.)
    if '_aug' in base:
        base = base.rsplit('_aug', 1)[0]

    # Remove _png suffix (V5 Roboflow format)
    if base.endswith('_png'):
        base = base[:-4]

    # Remove prefixes (v3_, v4_, v5_, roboflow_, thin_, thick_, no_hem_) - keep looping until none left
    # This handles cases like "v3_thick_12345" or "no_hem_520892552_14" where there are multiple prefixes
    prefixes = ['v3_', 'v4_', 'v5_', 'roboflow_', 'thin_', 'thick_', 'no_hem_']
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if base.startswith(prefix):
                base = base[len(prefix):]
                changed = True
                break

    # Extract first numeric group
    match = re.match(r'^(\d+)', base)
    if match:
        return match.group(1)

    # Fallback: use full filename as patient ID
    log(f"  ⚠️ Could not extract patient ID from: {filename}, using filename as ID")
    return base


def patient_level_split(
    coco_data: Dict,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int
) -> Tuple[Dict, Dict, Dict]:
    """
    Split dataset by patient ID ensuring all classes in all splits.

    Adapted from utils/train_valid_test_split.py
    """
    random.seed(seed)

    # Group images by patient
    patient_to_images = defaultdict(list)
    for img in coco_data['images']:
        patient_id = extract_patient_id(img['file_name'])
        patient_to_images[patient_id].append(img['id'])

    log(f"  Found {len(patient_to_images)} unique patients")

    # Get classes for each patient
    image_to_classes = defaultdict(set)
    for ann in coco_data['annotations']:
        image_to_classes[ann['image_id']].add(ann['category_id'])

    patient_to_classes = {}
    for patient_id, image_ids in patient_to_images.items():
        all_classes = set()
        for img_id in image_ids:
            all_classes.update(image_to_classes.get(img_id, set()))
        patient_to_classes[patient_id] = all_classes

    # Get all unique classes
    all_classes = set()
    for classes in patient_to_classes.values():
        all_classes.update(classes)
    log(f"  Dataset contains {len(all_classes)} classes: {sorted(all_classes)}")

    # Create patient info and shuffle
    patient_info = []
    for patient_id, image_ids in patient_to_images.items():
        patient_info.append({
            'id': patient_id,
            'num_images': len(image_ids),
            'classes': patient_to_classes[patient_id]
        })
    random.shuffle(patient_info)

    total_images = len(coco_data['images'])
    target_test = int(total_images * test_ratio)
    target_val = int(total_images * val_ratio)

    # Select test patients (ensuring all classes)
    test_patients = []
    test_images = set()
    test_classes_covered = set()
    test_image_count = 0

    for patient in patient_info:
        new_classes = patient['classes'] - test_classes_covered

        # Add patient if:
        # 1. They have new classes we need, OR
        # 2. We haven't covered all classes yet, OR
        # 3. We haven't reached the target yet
        should_add = (
            new_classes or
            test_classes_covered != all_classes or
            test_image_count < target_test
        )

        if should_add:
            # Check if adding this patient would exceed overflow limit
            if test_image_count + patient['num_images'] <= target_test * 1.4:
                test_patients.append(patient['id'])
                test_images.update(patient_to_images[patient['id']])
                test_image_count += patient['num_images']
                test_classes_covered.update(patient['classes'])

    log(f"  Test: {len(test_patients)} patients, {len(test_images)} images")
    log(f"    Classes: {sorted(test_classes_covered)}")

    # Select validation patients
    remaining_patients = [p for p in patient_info if p['id'] not in test_patients]
    random.shuffle(remaining_patients)

    val_patients = []
    val_images = set()
    val_classes_covered = set()
    val_image_count = 0

    for patient in remaining_patients:
        new_classes = patient['classes'] - val_classes_covered

        # Add patient if:
        # 1. They have new classes we need, OR
        # 2. We haven't covered all classes yet, OR
        # 3. We haven't reached the target yet
        should_add = (
            new_classes or
            val_classes_covered != all_classes or
            val_image_count < target_val
        )

        if should_add:
            # Check if adding this patient would exceed overflow limit
            if val_image_count + patient['num_images'] <= target_val * 1.2:
                val_patients.append(patient['id'])
                val_images.update(patient_to_images[patient['id']])
                val_image_count += patient['num_images']
                val_classes_covered.update(patient['classes'])

    log(f"  Valid: {len(val_patients)} patients, {len(val_images)} images")
    log(f"    Classes: {sorted(val_classes_covered)}")

    # Training gets all remaining
    train_patients = [p['id'] for p in patient_info
                      if p['id'] not in test_patients and p['id'] not in val_patients]
    train_images = set()
    for pid in train_patients:
        train_images.update(patient_to_images[pid])

    train_classes_covered = set()
    for img_id in train_images:
        train_classes_covered.update(image_to_classes.get(img_id, set()))

    log(f"  Train: {len(train_patients)} patients, {len(train_images)} images")
    log(f"    Classes: {sorted(train_classes_covered)}")

    # Split COCO data
    train_coco = split_coco_by_images(coco_data, train_images)
    val_coco = split_coco_by_images(coco_data, val_images)
    test_coco = split_coco_by_images(coco_data, test_images)

    # Verify no leakage
    assert len(set(train_patients) & set(val_patients)) == 0
    assert len(set(train_patients) & set(test_patients)) == 0
    assert len(set(val_patients) & set(test_patients)) == 0

    return train_coco, val_coco, test_coco


def split_coco_by_images(coco_data: Dict, image_ids: Set[int]) -> Dict:
    """Extract subset of COCO dataset for specific image IDs."""
    split_data = {
        'images': [],
        'annotations': [],
        'categories': coco_data['categories'],
        'info': coco_data['info'].copy(),
        'image_paths': {}
    }

    for img in coco_data['images']:
        if img['id'] in image_ids:
            split_data['images'].append(img)
            if img['file_name'] in coco_data.get('image_paths', {}):
                split_data['image_paths'][img['file_name']] = coco_data['image_paths'][img['file_name']]

    for ann in coco_data['annotations']:
        if ann['image_id'] in image_ids:
            split_data['annotations'].append(ann)

    return split_data


def filter_missing_images(coco_data: Dict) -> Dict:
    """
    Filter out images that don't have corresponding files.

    Returns:
        COCO dataset with only images that have files available
    """
    available_image_ids = set()
    filtered_images = []
    filtered_image_paths = {}

    for img in coco_data['images']:
        if img['file_name'] in coco_data.get('image_paths', {}):
            available_image_ids.add(img['id'])
            filtered_images.append(img)
            filtered_image_paths[img['file_name']] = coco_data['image_paths'][img['file_name']]

    # Filter annotations to only include those for available images
    filtered_annotations = [
        ann for ann in coco_data['annotations']
        if ann['image_id'] in available_image_ids
    ]

    removed_images = len(coco_data['images']) - len(filtered_images)
    removed_annotations = len(coco_data['annotations']) - len(filtered_annotations)

    if removed_images > 0:
        log(f"  Filtered out {removed_images} images without files ({removed_annotations} annotations)")

    return {
        'images': filtered_images,
        'annotations': filtered_annotations,
        'categories': coco_data['categories'],
        'info': coco_data['info'],
        'image_paths': filtered_image_paths
    }


# ============================================================================
# NO-HEMORRHAGE BALANCING FUNCTIONS
# ============================================================================

def load_no_hemorrhage_csv(csv_path: str) -> Dict:
    """
    Load no-hemorrhage feedback CSV and categorize by priority.

    Returns:
        {
            'false_positives': [(patient_id, instance_no), ...],
            'true_negatives': [(patient_id, instance_no), ...]
        }
    """
    import csv

    fp_images = []
    tn_images = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            patient_id = row['patient_id']
            instance_no = row['instance_no']
            classification = row['classification_result']

            if classification == 'False Positive (FP)':
                fp_images.append((patient_id, instance_no))
            elif classification == 'True Negative (TN)':
                tn_images.append((patient_id, instance_no))

    log(f"  Loaded CSV: {len(fp_images)} FP, {len(tn_images)} TN")

    return {
        'false_positives': fp_images,
        'true_negatives': tn_images
    }


def extract_patient_id_and_instance(filename: str) -> Tuple[str, str]:
    """
    Extract both patient_id and instance_no from filename.

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
    if '_aug' in base:
        base = base.rsplit('_aug', 1)[0]

    # Remove prefixes
    prefixes = ['v3_', 'roboflow_', 'thin_', 'thick_']
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if base.startswith(prefix):
                base = base[len(prefix):]
                changed = True
                break

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


def identify_existing_no_hemorrhage_images(coco_data: Dict) -> Set:
    """
    Find all images in the dataset that have NO hemorrhage annotations.

    Returns:
        Set of image IDs with zero annotations
    """
    all_image_ids = set(img['id'] for img in coco_data['images'])
    images_with_annotations = set(ann['image_id'] for ann in coco_data['annotations'])
    no_hem_image_ids = all_image_ids - images_with_annotations

    return no_hem_image_ids


def match_csv_to_dataset_images(coco_data: Dict, csv_data: Dict) -> Dict:
    """
    Match CSV patient_id/instance_no to image IDs in the dataset.

    Returns:
        {
            'fp_matched_ids': [img_id1, img_id2, ...],
            'tn_matched_ids': [img_id3, img_id4, ...],
            'fp_unmatched': [(pid, inst), ...],
            'tn_unmatched': [(pid, inst), ...]
        }
    """
    # Build mapping: (patient_id, instance_no) -> image_id
    patient_instance_to_image = {}
    for img in coco_data['images']:
        filename = img['file_name']
        patient_id, instance_no = extract_patient_id_and_instance(filename)
        key = (patient_id, instance_no)
        patient_instance_to_image[key] = img['id']

    fp_matched = []
    fp_unmatched = []
    for pid, inst in csv_data['false_positives']:
        key = (str(pid), str(inst))
        if key in patient_instance_to_image:
            fp_matched.append(patient_instance_to_image[key])
        else:
            fp_unmatched.append((pid, inst))

    tn_matched = []
    tn_unmatched = []
    for pid, inst in csv_data['true_negatives']:
        key = (str(pid), str(inst))
        if key in patient_instance_to_image:
            tn_matched.append(patient_instance_to_image[key])
        else:
            tn_unmatched.append((pid, inst))

    log(f"    FP: {len(fp_matched)} matched, {len(fp_unmatched)} not found")
    log(f"    TN: {len(tn_matched)} matched, {len(tn_unmatched)} not found")

    return {
        'fp_matched_ids': fp_matched,
        'tn_matched_ids': tn_matched,
        'fp_unmatched': fp_unmatched,
        'tn_unmatched': tn_unmatched
    }


def balance_no_hemorrhage_in_splits(
    train_coco: Dict,
    val_coco: Dict,
    test_coco: Dict,
    target_ratio: float,
    no_hem_pool_coco: Dict,
    csv_matches: Dict
) -> Tuple[Dict, Dict, Dict]:
    """
    Balance no-hemorrhage images in all splits to achieve target_ratio.

    CRITICAL: Respects patient-level splitting - only adds no_hem images from
    patients that ALREADY exist in each split to prevent patient leakage.

    Strategy:
    1. Identify which patients are in each split (train/valid/test)
    2. Group no_hem pool images by patient ID
    3. For each split, only add no_hem images from patients ALREADY in that split
    4. Prioritize FP images over TN images within each split
    5. For new patients (not in any split), distribute proportionally

    Args:
        train_coco, val_coco, test_coco: COCO data for each split
        target_ratio: Target percentage of no-hem images (0.5 = 50%)
        no_hem_pool_coco: Pool of downloaded no-hem images to distribute
        csv_matches: Dict with fp_matched_ids and tn_matched_ids

    Returns:
        Updated (train_coco, val_coco, test_coco) with balanced no-hem images
    """
    log("\n[6.5/9] Balancing no-hemorrhage images in splits...")

    if no_hem_pool_coco is None or len(no_hem_pool_coco['images']) == 0:
        log("  No additional no-hem images available for balancing")
        return train_coco, val_coco, test_coco

    # Identify existing no-hem images in each split
    train_no_hem = identify_existing_no_hemorrhage_images(train_coco)
    val_no_hem = identify_existing_no_hemorrhage_images(val_coco)
    test_no_hem = identify_existing_no_hemorrhage_images(test_coco)

    # Count current hemorrhage images
    train_with_hem = len(train_coco['images']) - len(train_no_hem)
    val_with_hem = len(val_coco['images']) - len(val_no_hem)
    test_with_hem = len(test_coco['images']) - len(test_no_hem)

    # Calculate target no-hem counts for 50/50 balance
    target_train_no_hem = int(train_with_hem / (1 - target_ratio) * target_ratio)
    target_val_no_hem = int(val_with_hem / (1 - target_ratio) * target_ratio)
    target_test_no_hem = int(test_with_hem / (1 - target_ratio) * target_ratio)

    # Calculate shortfall in each split
    train_shortfall = max(0, target_train_no_hem - len(train_no_hem))
    val_shortfall = max(0, target_val_no_hem - len(val_no_hem))
    test_shortfall = max(0, target_test_no_hem - len(test_no_hem))
    total_shortfall = train_shortfall + val_shortfall + test_shortfall

    log(f"\n  Current balance per split:")
    log(f"    Train: {len(train_no_hem)}/{len(train_coco['images'])} no-hem ({len(train_no_hem)/(len(train_coco['images']) or 1)*100:.1f}%) | Need {train_shortfall} more")
    log(f"    Valid: {len(val_no_hem)}/{len(val_coco['images'])} no-hem ({len(val_no_hem)/(len(val_coco['images']) or 1)*100:.1f}%) | Need {val_shortfall} more")
    log(f"    Test: {len(test_no_hem)}/{len(test_coco['images'])} no-hem ({len(test_no_hem)/(len(test_coco['images']) or 1)*100:.1f}%) | Need {test_shortfall} more")
    log(f"  Total shortfall: {total_shortfall} no-hem images")

    if total_shortfall == 0:
        log("  ✓ All splits already balanced!")
        return train_coco, val_coco, test_coco

    # Build patient sets for each split
    log("\n  Building patient-level mappings...")
    train_patients = set(extract_patient_id(img['file_name']) for img in train_coco['images'])
    val_patients = set(extract_patient_id(img['file_name']) for img in val_coco['images'])
    test_patients = set(extract_patient_id(img['file_name']) for img in test_coco['images'])

    log(f"    Train patients: {len(train_patients)}")
    log(f"    Valid patients: {len(val_patients)}")
    log(f"    Test patients: {len(test_patients)}")

    # Group no_hem pool images by patient ID and FP/TN status
    pool_image_ids = set(img['id'] for img in no_hem_pool_coco['images'])
    fp_pool_ids = set(csv_matches['fp_matched_ids']) & pool_image_ids

    # Build patient -> image_ids mapping for pool
    pool_patient_to_images = {}
    pool_patient_to_priority = {}  # Track if patient has FP or TN images

    for img in no_hem_pool_coco['images']:
        patient_id = extract_patient_id(img['file_name'])
        if patient_id not in pool_patient_to_images:
            pool_patient_to_images[patient_id] = []
            pool_patient_to_priority[patient_id] = 'TN'  # Default to TN

        pool_patient_to_images[patient_id].append(img['id'])

        # Mark as FP if ANY image from this patient is FP
        if img['id'] in fp_pool_ids:
            pool_patient_to_priority[patient_id] = 'FP'

    log(f"\n  No-hem pool analysis:")
    log(f"    Total pool images: {len(no_hem_pool_coco['images'])}")
    log(f"    Unique patients in pool: {len(pool_patient_to_images)}")
    log(f"    FP patients: {sum(1 for p in pool_patient_to_priority.values() if p == 'FP')}")
    log(f"    TN patients: {sum(1 for p in pool_patient_to_priority.values() if p == 'TN')}")

    # Categorize pool patients by split membership
    pool_patients_in_train = set(p for p in pool_patient_to_images.keys() if p in train_patients)
    pool_patients_in_val = set(p for p in pool_patient_to_images.keys() if p in val_patients)
    pool_patients_in_test = set(p for p in pool_patient_to_images.keys() if p in test_patients)
    pool_patients_new = set(pool_patient_to_images.keys()) - pool_patients_in_train - pool_patients_in_val - pool_patients_in_test

    log(f"\n  Patient overlap with existing splits:")
    log(f"    Pool patients already in Train: {len(pool_patients_in_train)}")
    log(f"    Pool patients already in Valid: {len(pool_patients_in_val)}")
    log(f"    Pool patients already in Test: {len(pool_patients_in_test)}")
    log(f"    NEW patients (not in any split): {len(pool_patients_new)}")

    # Function to collect images for a split
    def collect_images_for_split(
        split_name: str,
        pool_patients_in_split: Set[str],
        shortfall: int
    ) -> List[int]:
        """Collect image IDs for a split, respecting patient constraints."""
        selected_ids = []

        # Priority 1: FP patients already in this split
        fp_patients = sorted([p for p in pool_patients_in_split if pool_patient_to_priority[p] == 'FP'])
        for patient in fp_patients:
            if len(selected_ids) >= shortfall:
                break
            selected_ids.extend(pool_patient_to_images[patient])

        # Priority 2: TN patients already in this split
        tn_patients = sorted([p for p in pool_patients_in_split if pool_patient_to_priority[p] == 'TN'])
        for patient in tn_patients:
            if len(selected_ids) >= shortfall:
                break
            selected_ids.extend(pool_patient_to_images[patient])

        log(f"    {split_name}: Selected {len(selected_ids)} images from {len(pool_patients_in_split)} existing patients")
        return selected_ids[:shortfall]  # Trim to exact shortfall

    # Collect images for each split
    train_pool_ids = collect_images_for_split('Train', pool_patients_in_train, train_shortfall)
    val_pool_ids = collect_images_for_split('Valid', pool_patients_in_val, val_shortfall)
    test_pool_ids = collect_images_for_split('Test', pool_patients_in_test, test_shortfall)

    # Handle NEW patients (not in any split) - distribute proportionally
    if len(pool_patients_new) > 0:
        log(f"\n  Distributing {len(pool_patients_new)} NEW patients proportionally:")

        # Sort new patients by priority (FP first, then TN)
        new_fp_patients = sorted([p for p in pool_patients_new if pool_patient_to_priority[p] == 'FP'])
        new_tn_patients = sorted([p for p in pool_patients_new if pool_patient_to_priority[p] == 'TN'])
        new_patients_ordered = new_fp_patients + new_tn_patients

        # Calculate remaining shortfall after adding existing patient images
        remaining_train = max(0, train_shortfall - len(train_pool_ids))
        remaining_val = max(0, val_shortfall - len(val_pool_ids))
        remaining_test = max(0, test_shortfall - len(test_pool_ids))
        total_remaining = remaining_train + remaining_val + remaining_test

        if total_remaining > 0:
            # Distribute new patients proportionally
            random.seed(RANDOM_SEED)
            random.shuffle(new_patients_ordered)

            for patient in new_patients_ordered:
                # Determine which split needs images most
                if remaining_train > 0 and remaining_train >= remaining_val and remaining_train >= remaining_test:
                    train_pool_ids.extend(pool_patient_to_images[patient])
                    remaining_train -= len(pool_patient_to_images[patient])
                elif remaining_val > 0 and remaining_val >= remaining_test:
                    val_pool_ids.extend(pool_patient_to_images[patient])
                    remaining_val -= len(pool_patient_to_images[patient])
                elif remaining_test > 0:
                    test_pool_ids.extend(pool_patient_to_images[patient])
                    remaining_test -= len(pool_patient_to_images[patient])
                else:
                    break

            log(f"    Added {len(new_patients_ordered)} new patients to splits")

    log(f"\n  Final distribution:")
    log(f"    Train: {len(train_pool_ids)} images")
    log(f"    Valid: {len(val_pool_ids)} images")
    log(f"    Test: {len(test_pool_ids)} images")

    # Add images from pool to respective splits
    def add_images_to_split(split_coco: Dict, pool_ids: List[int], pool_coco: Dict) -> Dict:
        """Add images from pool to split."""
        if len(pool_ids) == 0:
            return split_coco

        # Get max IDs to avoid conflicts
        max_img_id = max([img['id'] for img in split_coco['images']], default=0)
        max_ann_id = max([ann['id'] for ann in split_coco['annotations']], default=0)

        # Build mapping of old pool IDs to new IDs
        id_mapping = {}
        for new_id, old_id in enumerate(pool_ids, start=max_img_id + 1):
            id_mapping[old_id] = new_id

        # Add images from pool
        for img in pool_coco['images']:
            if img['id'] in pool_ids:
                img_copy = img.copy()
                img_copy['id'] = id_mapping[img['id']]
                split_coco['images'].append(img_copy)

                # Copy image path mapping
                if img['file_name'] in pool_coco.get('image_paths', {}):
                    split_coco['image_paths'][img['file_name']] = pool_coco['image_paths'][img['file_name']]

        # Add annotations from pool (should be empty for no-hem images, but copy anyway)
        for ann in pool_coco['annotations']:
            if ann['image_id'] in pool_ids:
                ann_copy = ann.copy()
                ann_copy['id'] = max_ann_id + 1
                ann_copy['image_id'] = id_mapping[ann['image_id']]
                split_coco['annotations'].append(ann_copy)
                max_ann_id += 1

        return split_coco

    # Add images to splits
    train_coco = add_images_to_split(train_coco, train_pool_ids, no_hem_pool_coco)
    val_coco = add_images_to_split(val_coco, val_pool_ids, no_hem_pool_coco)
    test_coco = add_images_to_split(test_coco, test_pool_ids, no_hem_pool_coco)

    # Report final balance
    log(f"\n  Final balance after distribution:")
    train_no_hem_final = identify_existing_no_hemorrhage_images(train_coco)
    val_no_hem_final = identify_existing_no_hemorrhage_images(val_coco)
    test_no_hem_final = identify_existing_no_hemorrhage_images(test_coco)

    log(f"    Train: {len(train_no_hem_final)}/{len(train_coco['images'])} no-hem ({len(train_no_hem_final)/len(train_coco['images'])*100:.1f}%)")
    log(f"    Valid: {len(val_no_hem_final)}/{len(val_coco['images'])} no-hem ({len(val_no_hem_final)/len(val_coco['images'])*100:.1f}%)")
    log(f"    Test: {len(test_no_hem_final)}/{len(test_coco['images'])} no-hem ({len(test_no_hem_final)/len(test_coco['images'])*100:.1f}%)")

    total_images = len(train_coco['images']) + len(val_coco['images']) + len(test_coco['images'])
    total_no_hem = len(train_no_hem_final) + len(val_no_hem_final) + len(test_no_hem_final)

    log(f"\n  Overall: {total_no_hem}/{total_images} no-hem ({total_no_hem/total_images*100:.1f}%)")

    # Verify no patient leakage after balancing (no_hem images don't have patient grouping, so skip this check)
    log(f"\n  ✓ No-hemorrhage balancing complete (patient-level constraints maintained)")

    return train_coco, val_coco, test_coco


def save_coco_split(
    coco_data: Dict,
    output_dir: str,
    split_name: str,
    use_symlinks: bool = True
):
    """Save COCO split with images."""
    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)

    # Save annotations
    ann_path = os.path.join(split_dir, '_annotations.coco.json')
    with open(ann_path, 'w') as f:
        json.dump({
            'images': coco_data['images'],
            'annotations': coco_data['annotations'],
            'categories': coco_data['categories'],
            'info': coco_data['info']
        }, f, indent=2)

    log(f"  Saved annotations: {ann_path}")

    # Copy/symlink images
    copied = 0
    missing = 0
    for img_info in tqdm(coco_data['images'], desc=f"  Copying {split_name} images"):
        filename = img_info['file_name']
        source_path = coco_data['image_paths'].get(filename)

        if source_path is None or not os.path.exists(source_path):
            missing += 1
            continue

        dest_path = os.path.join(split_dir, filename)

        if use_symlinks:
            if os.path.exists(dest_path):
                os.remove(dest_path)
            os.symlink(os.path.abspath(source_path), dest_path)
        else:
            shutil.copy2(source_path, dest_path)

        copied += 1

    log(f"  Copied/linked {copied}/{len(coco_data['images'])} images ({missing} missing)")


def apply_augmentation(input_dir: str, output_dir: str, multiplier: int):
    """Apply 3x augmentation using existing augment_coco_dataset.py logic."""
    import sys
    sys.path.append('/Users/yugambahl/Desktop/brain_ct')
    from scripts.augment_coco_dataset import augment_coco_dataset

    log(f"  Augmenting training set with {multiplier}x multiplier...")
    augment_coco_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        multiplier=multiplier,
        strength=AUGMENTATION_STRENGTH,
        image_size=IMAGE_SIZE
    )


def create_filtered_4class_dataset(source_6class_dir: str, output_4class_dir: str):
    """Create filtered 4-class dataset from 6-class dataset."""
    log("\n[9/9] Creating filtered 4-class dataset...")

    os.makedirs(output_4class_dir, exist_ok=True)

    # Process each split
    for split in ['train', 'valid', 'test']:
        log(f"  Processing {split} split...")

        # Load 6-class annotations
        source_ann_path = os.path.join(source_6class_dir, split, '_annotations.coco.json')
        if not os.path.exists(source_ann_path):
            log(f"    ⚠️ Skipping {split} (not found)")
            continue

        with open(source_ann_path, 'r') as f:
            data = json.load(f)

        # Filter and remap annotations
        filtered_annotations, filtered_count = remap_annotations(
            data['annotations'],
            V4_6CLASS_TO_4CLASS_MAPPING
        )

        # Update categories
        filtered_categories = [
            {'id': i, 'name': name, 'supercategory': 'hemorrhage'}
            for i, name in enumerate(V4_4CLASS_NAMES)
        ]

        # Save filtered annotations
        output_split_dir = os.path.join(output_4class_dir, split)
        os.makedirs(output_split_dir, exist_ok=True)

        filtered_data = {
            'images': data['images'],
            'annotations': filtered_annotations,
            'categories': filtered_categories,
            'info': {
                'description': 'V4 Brain CT Hemorrhage Dataset - 4-class Filtered',
                'version': '4.0',
                'year': 2025,
            }
        }

        output_ann_path = os.path.join(output_split_dir, '_annotations.coco.json')
        with open(output_ann_path, 'w') as f:
            json.dump(filtered_data, f, indent=2)

        # Create symlinks to 6-class images
        source_image_dir = os.path.join(source_6class_dir, split)
        for img_info in tqdm(data['images'], desc=f"  Symlinking {split} images"):
            filename = img_info['file_name']
            source_image_path = os.path.join(source_image_dir, filename)
            dest_image_path = os.path.join(output_split_dir, filename)

            if os.path.exists(source_image_path):
                if os.path.exists(dest_image_path):
                    os.remove(dest_image_path)
                os.symlink(os.path.abspath(source_image_path), dest_image_path)

        log(f"    Filtered {filtered_count} EDH/HC annotations")
        log(f"    {len(filtered_annotations)} annotations remaining")

    # Save metadata
    metadata = {
        'version': '4.0',
        'type': '4-class filtered',
        'classes': V4_4CLASS_NAMES,
        'source': '6-class dataset with EDH/HC filtered out',
        'created': datetime.now().isoformat()
    }

    metadata_path = os.path.join(output_4class_dir, 'metadata.yaml')
    with open(metadata_path, 'w') as f:
        yaml.dump(metadata, f)

    log(f"  Saved metadata: {metadata_path}")


def load_and_reverse_v4_augmentation(v4_coco_dir: str) -> Dict:
    """
    Load V4 dataset and keep only aug0 from train split (revert augmentation).

    For V5: Loads V4 6-class dataset and extracts only the original images (aug0)
    from the train split, while keeping all valid/test images.

    Args:
        v4_coco_dir: Path to V4 6class_coco directory

    Returns:
        Combined COCO dict with de-augmented images
    """
    log("  Reversing V4 augmentation (keeping only aug0 from train)...")

    combined = {
        'images': [],
        'annotations': [],
        'categories': None,
        'image_paths': {}
    }

    next_img_id = 1
    next_ann_id = 1

    for split in ['train', 'valid', 'test']:
        json_path = os.path.join(v4_coco_dir, split, '_annotations.coco.json')

        if not os.path.exists(json_path):
            log(f"    ⚠️ Split not found: {split}")
            continue

        with open(json_path, 'r') as f:
            data = json.load(f)

        # Set categories from first split
        if combined['categories'] is None:
            combined['categories'] = data['categories']

        old_to_new_img_id = {}

        if split == 'train':
            # Keep only aug0 images
            aug0_count = 0
            for img in data['images']:
                filename = img['file_name']
                if '_aug0.' in filename or '_aug0_' in filename:
                    # This is an aug0 image - keep it but rename to remove _aug0
                    old_img_id = img['id']
                    new_img_id = next_img_id
                    old_to_new_img_id[old_img_id] = new_img_id

                    img_copy = img.copy()
                    img_copy['id'] = new_img_id
                    # Remove _aug0 from filename
                    img_copy['file_name'] = filename.replace('_aug0.', '.').replace('_aug0_', '_')

                    # Store source path (with _aug0 suffix) for file copying
                    source_path = os.path.join(v4_coco_dir, split, filename)
                    combined['image_paths'][img_copy['file_name']] = source_path

                    combined['images'].append(img_copy)
                    next_img_id += 1
                    aug0_count += 1

            log(f"    {split}: kept {aug0_count} aug0 images out of {len(data['images'])} total")

            # Filter annotations to match aug0 images only
            for ann in data['annotations']:
                if ann['image_id'] in old_to_new_img_id:
                    ann_copy = ann.copy()
                    ann_copy['id'] = next_ann_id
                    ann_copy['image_id'] = old_to_new_img_id[ann['image_id']]
                    combined['annotations'].append(ann_copy)
                    next_ann_id += 1
        else:
            # Valid/test: keep all images
            for img in data['images']:
                old_img_id = img['id']
                new_img_id = next_img_id
                old_to_new_img_id[old_img_id] = new_img_id

                img_copy = img.copy()
                img_copy['id'] = new_img_id

                # Store source path
                source_path = os.path.join(v4_coco_dir, split, img['file_name'])
                combined['image_paths'][img_copy['file_name']] = source_path

                combined['images'].append(img_copy)
                next_img_id += 1

            log(f"    {split}: kept all {len(data['images'])} images")

            # Copy all annotations
            for ann in data['annotations']:
                if ann['image_id'] in old_to_new_img_id:
                    ann_copy = ann.copy()
                    ann_copy['id'] = next_ann_id
                    ann_copy['image_id'] = old_to_new_img_id[ann['image_id']]
                    combined['annotations'].append(ann_copy)
                    next_ann_id += 1

    log(f"  V4 de-augmented: {len(combined['images'])} images, {len(combined['annotations'])} annotations")
    return combined


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Prepare V4/V5 dataset')
    parser.add_argument('--version', type=str, default='v4', choices=['v4', 'v5'],
                        help='Dataset version to prepare (default: v4)')
    parser.add_argument('--skip-augmentation', action='store_true',
                        help='Skip augmentation step (create original-only version)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Validate configuration without executing')
    args = parser.parse_args()

    # Get version-specific configuration
    config = VERSION_CONFIG[args.version]

    print("=" * 80)
    print(f"BRAIN CT HEMORRHAGE DETECTION - {args.version.upper()} DATASET PREPARATION")
    print("=" * 80)
    print(f"Version: {args.version}")
    print(f"Skip augmentation: {args.skip_augmentation}")
    print("=" * 80)

    # Version-aware data loading
    datasets_to_merge = []
    prefixes = []

    if args.version == 'v4':
        # ===== V4 PIPELINE =====
        # [1/9] Load V3 dataset
        log("\n[1/9] Loading V3 COCO dataset (4-class filtered)...")
        v3_splits = {}
        for split in ['train', 'valid', 'test']:
            json_path = f"{V3_COCO_DIR}/{split}/_annotations.coco.json"
            if os.path.exists(json_path):
                v3_splits[split] = load_coco_dataset(json_path, f"v3_{split}", f"{V3_COCO_DIR}/{split}")

        # Combine v3 splits (will be re-split later)
        v3_combined_images = []
        v3_combined_annotations = []
        v3_image_paths = {}

        next_img_id = 1
        next_ann_id = 1
        old_to_new_img_id = {}

        for split_data in v3_splits.values():
            for img in split_data['images']:
                old_id = img['id']
                img_copy = img.copy()
                img_copy['id'] = next_img_id
                old_to_new_img_id[old_id] = next_img_id
                v3_combined_images.append(img_copy)

                if img['file_name'] in split_data['image_paths']:
                    v3_image_paths[img_copy['file_name']] = split_data['image_paths'][img['file_name']]

                next_img_id += 1

            for ann in split_data['annotations']:
                ann_copy = ann.copy()
                ann_copy['id'] = next_ann_id
                ann_copy['image_id'] = old_to_new_img_id[ann['image_id']]
                v3_combined_annotations.append(ann_copy)
                next_ann_id += 1

        v3_combined = {
            'images': v3_combined_images,
            'annotations': v3_combined_annotations,
            'categories': [{'id': i, 'name': name, 'supercategory': 'hemorrhage'}
                           for i, name in enumerate(['IPH', 'IVH', 'SAH', 'SDH'])],
            'image_paths': v3_image_paths
        }

        log(f"  V3 combined: {len(v3_combined['images'])} images, {len(v3_combined['annotations'])} annotations")

        # [2/9] Remap V3 classes
        log("\n[2/9] Remapping V3 classes (4-class → 6-class positions)...")
        v3_combined['annotations'], filtered = remap_annotations(
            v3_combined['annotations'],
            V3_TO_V4_MAPPING
        )
        log(f"  Remapped {len(v3_combined['annotations'])} annotations, filtered {filtered}")

        # [3/9] Load Roboflow dataset
        log("\n[3/9] Loading Roboflow COCO dataset...")
        roboflow_data = load_coco_dataset(
            f"{ROBOFLOW_COCO_DIR}/train/_annotations.coco.json",
            "roboflow",
            image_base_dir=f"{ROBOFLOW_COCO_DIR}/train"
        )

        # [4/9] Remap and filter Roboflow
        log("\n[4/9] Remapping Roboflow classes (8 categories → 6 classes)...")
        roboflow_data['annotations'], filtered = remap_annotations(
            roboflow_data['annotations'],
            ROBOFLOW_TO_V4_MAPPING
        )
        log(f"  Remapped {len(roboflow_data['annotations'])} annotations, filtered {filtered} noise categories")

        datasets_to_merge = [v3_combined, roboflow_data]
        prefixes = ['v3', 'roboflow']

    else:  # V5
        # ===== V5 PIPELINE =====
        # [1/9] Load V4 dataset (with augmentation reversal)
        log("\n[1/9] Loading V4 COCO dataset (reverting augmentation)...")
        v4_combined = load_and_reverse_v4_augmentation(config['sources']['v4'])

        # V4 data is already in correct 6-class format - no remapping needed
        log(f"  V4 loaded: {len(v4_combined['images'])} images, {len(v4_combined['annotations'])} annotations")

        # [2/9] Load V5 Roboflow dataset
        log("\n[2/9] Loading V5 Roboflow COCO dataset...")
        v5_roboflow_json = os.path.join(config['sources']['v5_roboflow'], '_annotations.coco.json')

        if not os.path.exists(v5_roboflow_json):
            log(f"  ❌ ERROR: V5 Roboflow annotations not found at {v5_roboflow_json}")
            log(f"  Expected location: {config['sources']['v5_roboflow']}")
            return

        v5_roboflow_data = load_coco_dataset(
            v5_roboflow_json,
            "v5_roboflow",
            image_base_dir=config['sources']['v5_roboflow']
        )

        # [3/9] Remap V5 Roboflow classes (8 categories → 6 classes)
        log("\n[3/9] Remapping V5 Roboflow classes (8 categories → 6 classes)...")
        v5_roboflow_data['annotations'], filtered = remap_annotations(
            v5_roboflow_data['annotations'],
            V5_ROBOFLOW_TO_V4_MAPPING
        )
        log(f"  Remapped {len(v5_roboflow_data['annotations'])} annotations, filtered {filtered} noise categories")

        datasets_to_merge = [v4_combined, v5_roboflow_data]
        prefixes = ['v4', 'v5']

    # [4.5/9] Load no-hemorrhage downloads if available (kept separate for split balancing)
    no_hem_pool = None
    no_hem_dir = config['sources']['no_hem']
    no_hem_json = os.path.join(no_hem_dir, '_annotations.coco.json')

    if os.path.exists(no_hem_json):
        log(f"\n[4.5/9] Loading downloaded no-hemorrhage images from {args.version}...")
        no_hem_pool = load_coco_dataset(
            no_hem_json,
            "no_hem_downloads",
            image_base_dir=no_hem_dir
        )
        log(f"  No-hemorrhage downloads: {len(no_hem_pool['images'])} images (kept separate for balanced distribution)")
        log(f"  These will be distributed across train/valid/test to achieve 50% balance in each split")
    else:
        log(f"\n[4.5/9] No downloaded no-hemorrhage images found at {no_hem_dir}")
        log(f"  (Optional) Run: python scripts/download_missing_no_hemorrhage_images.py --version {args.version}")
        log(f"  To achieve 50% balance in splits")

    # [5/9] Merge datasets (no_hem_pool kept separate)
    log(f"\n[5/9] Merging {args.version.upper()} datasets...")
    unified_categories = [
        {'id': i, 'name': name, 'supercategory': 'hemorrhage'}
        for i, name in enumerate(V4_6CLASS_NAMES)
    ]

    combined_coco = merge_coco_datasets(
        datasets_to_merge,
        unified_categories,
        dataset_prefixes=prefixes
    )

    # Filter out images without files
    log("\n[5.5/9] Filtering out images without available files...")
    combined_coco = filter_missing_images(combined_coco)

    # Fix out-of-bounds bounding boxes
    log("\n[5.6/9] Fixing out-of-bounds bounding boxes...")
    combined_coco = fix_out_of_bounds_bboxes(combined_coco)

    if args.dry_run:
        log("\n✅ DRY RUN COMPLETE - Configuration valid")
        return

    # [6/9] Patient-level split
    log("\n[6/9] Performing patient-level train/val/test split...")
    train_coco, val_coco, test_coco = patient_level_split(
        combined_coco,
        TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
        RANDOM_SEED
    )

    # [6.1/9] Load and analyze no-hemorrhage CSV for balancing
    csv_path = '/Users/yugambahl/Desktop/brain_ct/data/metadata/no_hemorrhage_positive_feedback.csv'
    csv_matches = None

    if os.path.exists(csv_path) and no_hem_pool is not None:
        log("\n[6.1/9] Loading no-hemorrhage feedback CSV...")
        csv_data = load_no_hemorrhage_csv(csv_path)

        log("\n[6.2/9] Matching CSV images to no-hem pool...")
        # Match CSV to the no_hem_pool instead of combined_coco
        csv_matches = match_csv_to_dataset_images(no_hem_pool, csv_data)

        # Apply balancing: distribute no_hem_pool across splits
        train_coco, val_coco, test_coco = balance_no_hemorrhage_in_splits(
            train_coco, val_coco, test_coco,
            target_ratio=0.5,  # 50% target
            no_hem_pool_coco=no_hem_pool,
            csv_matches=csv_matches
        )
    elif no_hem_pool is not None:
        log(f"\n⚠️ CSV file not found: {csv_path}")
        log("  Distributing no-hem pool without FP/TN categorization...")
        # Distribute without FP/TN info - treat all as equal priority
        dummy_csv_matches = {
            'fp_matched_ids': [],
            'tn_matched_ids': [img['id'] for img in no_hem_pool['images']]
        }
        train_coco, val_coco, test_coco = balance_no_hemorrhage_in_splits(
            train_coco, val_coco, test_coco,
            target_ratio=0.5,
            no_hem_pool_coco=no_hem_pool,
            csv_matches=dummy_csv_matches
        )
    else:
        log("\n[6.1/9] No downloaded no-hem images or CSV - skipping balancing")

    # Define output directories based on version
    temp_dir = f"{config['output_base']}/temp_combined"
    output_6class_original = f"{config['output_base']}/6class_coco_original"
    output_6class_augmented = f"{config['output_base']}/6class_coco"
    output_4class_original = f"{config['output_base']}/4class_coco_original"
    output_4class_augmented = f"{config['output_base']}/4class_coco"

    # [7/9] Save original (pre-augmentation) dataset ALWAYS
    log("\n[7/9] Saving original (pre-augmentation) dataset...")
    os.makedirs(output_6class_original, exist_ok=True)
    save_coco_split(train_coco, output_6class_original, "train", use_symlinks=False)
    save_coco_split(val_coco, output_6class_original, "valid", use_symlinks=False)
    save_coco_split(test_coco, output_6class_original, "test", use_symlinks=False)
    log(f"  ✓ Saved original dataset to: {output_6class_original}")

    # [8/9] Apply augmentation (optional)
    if not args.skip_augmentation:
        log("\n[8/9] Applying 3x augmentation to training set...")
        os.makedirs(temp_dir, exist_ok=True)
        save_coco_split(train_coco, temp_dir, "train", use_symlinks=False)
        save_coco_split(val_coco, temp_dir, "valid", use_symlinks=False)
        save_coco_split(test_coco, temp_dir, "test", use_symlinks=False)

        apply_augmentation(temp_dir, output_6class_augmented, AUGMENTATION_MULTIPLIER)
        log(f"  ✓ Saved augmented dataset to: {output_6class_augmented}")

        # Clean up temp directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    else:
        log("\n[8/9] Skipping augmentation (--skip-augmentation flag set)")

    # [9/9] Generate 4-class filtered versions
    log("\n[9/9] Generating 4-class filtered versions...")
    create_filtered_4class_dataset(output_6class_original, output_4class_original)
    log(f"  ✓ Created 4-class original: {output_4class_original}")

    if not args.skip_augmentation:
        create_filtered_4class_dataset(output_6class_augmented, output_4class_augmented)
        log(f"  ✓ Created 4-class augmented: {output_4class_augmented}")

    # Save metadata
    metadata = {
        'version': args.version,
        'type': '6-class full',
        'classes': V4_6CLASS_NAMES,
        'sources': config['sources'],
        'created': datetime.now().isoformat(),
        'splits': {
            'train': len(train_coco['images']),
            'valid': len(val_coco['images']),
            'test': len(test_coco['images'])
        },
        'augmented': not args.skip_augmentation
    }

    metadata_path = os.path.join(output_6class_original, 'metadata.yaml')
    with open(metadata_path, 'w') as f:
        yaml.dump(metadata, f)

    log(f"\n  Saved metadata: {metadata_path}")

    log("\n" + "=" * 80)
    log(f"✅ {args.version.upper()} DATASET PREPARATION COMPLETE!")
    log("=" * 80)
    log(f"\nGenerated datasets:")
    log(f"  • 6-class original: {output_6class_original}")
    log(f"  • 4-class original: {output_4class_original}")
    if not args.skip_augmentation:
        log(f"  • 6-class augmented: {output_6class_augmented}")
        log(f"  • 4-class augmented: {output_4class_augmented}")

    log(f"\nNext steps:")
    log(f"  1. Validate original: python scripts/validate_v4_dataset.py --version {args.version} --dataset-type original")
    if not args.skip_augmentation:
        log(f"  2. Validate augmented: python scripts/validate_v4_dataset.py --version {args.version} --dataset-type augmented")
    log(f"  3. Analyze datasets: python scripts/analyze_v4_dataset.py --version {args.version}")
    log(f"  4. Train models using Modal")


if __name__ == "__main__":
    main()
