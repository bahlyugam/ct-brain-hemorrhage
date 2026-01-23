#!/usr/bin/env python3
"""
Brain CT Hemorrhage Detection - V5 Combined Dataset Preparation

Combines 3 Roboflow datasets into unified 6-class COCO format:
1. CTBrain_Reverification-SmartPolygon_Changes.v2i.coco (13,857 images)
2. uat-ct-brain-hemorrhage-v4.v1i.coco (5,409 images)
3. v5_no_hemorrhage_downloads (5,979 images - negative examples)

Output: Single COCO dataset with train/valid splits (85/15)

Usage:
    python scripts/prepare_v5_combined_dataset.py
    python scripts/prepare_v5_combined_dataset.py --dry-run
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
from datetime import datetime
import argparse

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input dataset paths
DATASET1_DIR = "/Users/yugambahl/Desktop/brain_ct/data/training_datasets/roboflow_14k/CTBrain_Reverification-SmartPolygon_Changes.v2i.coco"
DATASET2_DIR = "/Users/yugambahl/Desktop/brain_ct/data/training_datasets/roboflow_14k/uat-ct-brain-hemorrhage-v4.v1i.coco"
DATASET3_DIR = "/Users/yugambahl/Desktop/brain_ct/data/training_datasets/roboflow_14k/v5_no_hemorrhage_downloads"

# Output path
OUTPUT_DIR = "/Users/yugambahl/Desktop/brain_ct/data/training_datasets/v5/roboflow_original_images"

# Split ratios (NO test split)
TRAIN_RATIO = 0.85
VAL_RATIO = 0.15

# Random seed for reproducibility
RANDOM_SEED = 42

# Category mappings
# Dataset 1: CTBrain_Reverification (9 categories)
# Categories: IPH-Mvyf(0), EDH(1), HC(2), IPH(3), IVH(4), SAH(5), SDH(6), abnormal(7), normal(8)
DATASET1_TO_V5_MAPPING = {
    0: None,  # IPH-Mvyf → FILTER (becomes negative example)
    1: 0,     # EDH → EDH
    2: 1,     # HC → HC
    3: 2,     # IPH → IPH
    4: 3,     # IVH → IVH
    5: 4,     # SAH → SAH
    6: 5,     # SDH → SDH
    7: None,  # abnormal → FILTER (becomes negative example)
    8: None,  # normal → FILTER (becomes negative example)
}

# Dataset 2: uat-ct-brain-hemorrhage-v4 (8 categories with duplicate IPH)
# Categories: IPH(0), "0"(1), EDH(2), HC(3), IPH(4), IVH(5), SAH(6), SDH(7)
DATASET2_TO_V5_MAPPING = {
    0: 2,     # IPH → IPH (first duplicate)
    1: None,  # "0" → FILTER (noise category)
    2: 0,     # EDH → EDH
    3: 1,     # HC → HC
    4: 2,     # IPH → IPH (second duplicate)
    5: 3,     # IVH → IVH
    6: 4,     # SAH → SAH
    7: 5,     # SDH → SDH
}

# Dataset 3: v5_no_hemorrhage - No mapping needed, already negative examples

# Target 6-class schema
V5_6CLASS_NAMES = ['EDH', 'HC', 'IPH', 'IVH', 'SAH', 'SDH']

# Dataset prefixes (for filename collision prevention)
DATASET_PREFIXES = ['d1', 'd2', 'd3']


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
        split_name: Name for logging (e.g., "dataset1", "dataset2")
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
            if os.path.exists(path):
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
            'description': 'V5 Brain CT Hemorrhage Dataset - Combined',
            'version': '5.0',
            'year': 2025,
        },
        'image_paths': {}
    }

    next_image_id = 1
    next_ann_id = 1

    for dataset_idx, dataset in enumerate(datasets):
        prefix = dataset_prefixes[dataset_idx] if dataset_prefixes else ''
        old_to_new_img_id = {}

        log(f"  Merging dataset {dataset_idx + 1}/{len(datasets)} (prefix: {prefix})...")

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
    - No-hem: 590349519_18.png → 590349519
    - With prefixes: d1_525190930_29_png.rf.*.jpg → 525190930
    """
    # Remove extension and Roboflow hash
    base = os.path.splitext(filename)[0]
    if '.rf.' in base:
        base = base.split('.rf.')[0]

    # Remove _png suffix (Roboflow format)
    if base.endswith('_png'):
        base = base[:-4]

    # Remove prefixes (d1_, d2_, d3_, etc.)
    prefixes = ['d1_', 'd2_', 'd3_', 'v3_', 'v4_', 'v5_', 'roboflow_', 'thin_', 'thick_', 'no_hem_']
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


def patient_level_split_two_way(
    coco_data: Dict,
    train_ratio: float,
    val_ratio: float,
    seed: int
) -> Tuple[Dict, Dict]:
    """
    Split dataset by patient ID into train/valid only (no test split).

    Ensures:
    - All images from same patient are in same split
    - All 6 classes are represented in both splits
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
    target_val = int(total_images * val_ratio)

    # Select validation patients (ensuring all classes)
    val_patients = []
    val_images = set()
    val_classes_covered = set()
    val_image_count = 0

    for patient in patient_info:
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
            if val_image_count + patient['num_images'] <= target_val * 1.3:
                val_patients.append(patient['id'])
                val_images.update(patient_to_images[patient['id']])
                val_image_count += patient['num_images']
                val_classes_covered.update(patient['classes'])

        # Stop if we have enough and all classes covered
        if val_image_count >= target_val and val_classes_covered == all_classes:
            break

    log(f"  Valid: {len(val_patients)} patients, {len(val_images)} images")
    log(f"    Classes: {sorted(val_classes_covered)}")

    # Training gets all remaining
    train_patients = [p['id'] for p in patient_info if p['id'] not in val_patients]
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

    # Verify no leakage
    assert len(set(train_patients) & set(val_patients)) == 0, "Patient leakage detected!"

    return train_coco, val_coco


def save_coco_split(
    coco_data: Dict,
    output_dir: str,
    split_name: str,
    use_copy: bool = True
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

    # Copy images
    copied = 0
    missing = 0
    for img_info in tqdm(coco_data['images'], desc=f"  Copying {split_name} images"):
        filename = img_info['file_name']
        source_path = coco_data['image_paths'].get(filename)

        if source_path is None or not os.path.exists(source_path):
            missing += 1
            continue

        dest_path = os.path.join(split_dir, filename)

        if use_copy:
            shutil.copy2(source_path, dest_path)
        else:
            if os.path.exists(dest_path):
                os.remove(dest_path)
            os.symlink(os.path.abspath(source_path), dest_path)

        copied += 1

    log(f"  Copied {copied}/{len(coco_data['images'])} images ({missing} missing)")


def get_class_distribution(coco_data: Dict, class_names: List[str]) -> Dict[str, int]:
    """Get annotation count per class."""
    counts = Counter()
    for ann in coco_data['annotations']:
        cat_id = ann['category_id']
        if 0 <= cat_id < len(class_names):
            counts[class_names[cat_id]] += 1
    return dict(counts)


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Prepare V5 Combined Dataset')
    parser.add_argument('--dry-run', action='store_true',
                        help='Validate configuration without executing')
    args = parser.parse_args()

    print("=" * 80)
    print("BRAIN CT HEMORRHAGE DETECTION - V5 COMBINED DATASET PREPARATION")
    print("=" * 80)
    print(f"Output: {OUTPUT_DIR}")
    print(f"Split ratios: Train {TRAIN_RATIO*100:.0f}%, Valid {VAL_RATIO*100:.0f}%")
    print("=" * 80)

    if args.dry_run:
        log("DRY RUN - No files will be written")

    # =========================================================================
    # Step 1: Load Dataset 1 (CTBrain_Reverification)
    # =========================================================================
    log("\n[1/7] Loading Dataset 1 (CTBrain_Reverification)...")
    dataset1_json = os.path.join(DATASET1_DIR, "train", "_annotations.coco.json")
    dataset1 = load_coco_dataset(dataset1_json, "dataset1", os.path.join(DATASET1_DIR, "train"))

    # Remap categories
    log("  Remapping categories...")
    dataset1['annotations'], filtered1 = remap_annotations(
        dataset1['annotations'],
        DATASET1_TO_V5_MAPPING
    )
    log(f"    Kept {len(dataset1['annotations'])} annotations, filtered {filtered1} (IPH-Mvyf, abnormal, normal)")

    # =========================================================================
    # Step 2: Load Dataset 2 (uat-ct-brain-hemorrhage-v4)
    # =========================================================================
    log("\n[2/7] Loading Dataset 2 (uat-ct-brain-hemorrhage-v4)...")
    dataset2_json = os.path.join(DATASET2_DIR, "train", "_annotations.coco.json")
    dataset2 = load_coco_dataset(dataset2_json, "dataset2", os.path.join(DATASET2_DIR, "train"))

    # Remap categories
    log("  Remapping categories...")
    dataset2['annotations'], filtered2 = remap_annotations(
        dataset2['annotations'],
        DATASET2_TO_V5_MAPPING
    )
    log(f"    Kept {len(dataset2['annotations'])} annotations, filtered {filtered2} (noise '0' category)")

    # =========================================================================
    # Step 3: Load Dataset 3 (v5_no_hemorrhage_downloads)
    # =========================================================================
    log("\n[3/7] Loading Dataset 3 (v5_no_hemorrhage_downloads)...")
    dataset3_json = os.path.join(DATASET3_DIR, "_annotations.coco.json")
    dataset3 = load_coco_dataset(dataset3_json, "dataset3", DATASET3_DIR)
    # No remapping needed - these are negative examples with 0 annotations

    # =========================================================================
    # Step 4: Merge datasets
    # =========================================================================
    log("\n[4/7] Merging datasets...")

    # Create unified 6-class category schema
    unified_categories = [
        {'id': i, 'name': name, 'supercategory': 'hemorrhage'}
        for i, name in enumerate(V5_6CLASS_NAMES)
    ]

    merged = merge_coco_datasets(
        datasets=[dataset1, dataset2, dataset3],
        unified_categories=unified_categories,
        dataset_prefixes=DATASET_PREFIXES
    )

    # Filter out images without files
    merged = filter_missing_images(merged)

    # Fix out-of-bounds bboxes
    merged = fix_out_of_bounds_bboxes(merged)

    log(f"\n  Final merged dataset:")
    log(f"    Total images: {len(merged['images'])}")
    log(f"    Total annotations: {len(merged['annotations'])}")

    # Class distribution
    class_dist = get_class_distribution(merged, V5_6CLASS_NAMES)
    log(f"    Class distribution: {class_dist}")

    # Count negative examples (images without annotations)
    images_with_ann = set(ann['image_id'] for ann in merged['annotations'])
    negative_count = len(merged['images']) - len(images_with_ann)
    log(f"    Negative examples (no annotations): {negative_count}")

    if args.dry_run:
        log("\nDRY RUN complete - skipping write operations")
        return

    # =========================================================================
    # Step 5: Patient-level split (85/15)
    # =========================================================================
    log("\n[5/7] Performing patient-level split (85/15)...")
    train_coco, val_coco = patient_level_split_two_way(
        merged,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        seed=RANDOM_SEED
    )

    # =========================================================================
    # Step 6: Save splits
    # =========================================================================
    log("\n[6/7] Saving train split...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_coco_split(train_coco, OUTPUT_DIR, "train", use_copy=True)

    log("\n[7/7] Saving valid split...")
    save_coco_split(val_coco, OUTPUT_DIR, "valid", use_copy=True)

    # =========================================================================
    # Generate metadata
    # =========================================================================
    log("\nGenerating metadata...")

    train_class_dist = get_class_distribution(train_coco, V5_6CLASS_NAMES)
    val_class_dist = get_class_distribution(val_coco, V5_6CLASS_NAMES)

    train_neg = len(train_coco['images']) - len(set(ann['image_id'] for ann in train_coco['annotations']))
    val_neg = len(val_coco['images']) - len(set(ann['image_id'] for ann in val_coco['annotations']))

    metadata = {
        'version': '5.0',
        'description': 'V5 Brain CT Hemorrhage Dataset - Combined from 3 sources',
        'created': datetime.now().isoformat(),
        'sources': {
            'dataset1': {
                'name': 'CTBrain_Reverification-SmartPolygon_Changes.v2i.coco',
                'path': DATASET1_DIR,
            },
            'dataset2': {
                'name': 'uat-ct-brain-hemorrhage-v4.v1i.coco',
                'path': DATASET2_DIR,
            },
            'dataset3': {
                'name': 'v5_no_hemorrhage_downloads',
                'path': DATASET3_DIR,
            },
        },
        'classes': V5_6CLASS_NAMES,
        'split_ratios': {
            'train': TRAIN_RATIO,
            'valid': VAL_RATIO,
        },
        'statistics': {
            'train': {
                'images': len(train_coco['images']),
                'annotations': len(train_coco['annotations']),
                'negative_examples': train_neg,
                'class_distribution': train_class_dist,
            },
            'valid': {
                'images': len(val_coco['images']),
                'annotations': len(val_coco['annotations']),
                'negative_examples': val_neg,
                'class_distribution': val_class_dist,
            },
        },
    }

    metadata_path = os.path.join(OUTPUT_DIR, 'metadata.yaml')
    with open(metadata_path, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False)

    log(f"  Saved metadata: {metadata_path}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 80)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"\nTrain split:")
    print(f"  - Images: {len(train_coco['images'])}")
    print(f"  - Annotations: {len(train_coco['annotations'])}")
    print(f"  - Negative examples: {train_neg}")
    print(f"  - Class distribution: {train_class_dist}")
    print(f"\nValid split:")
    print(f"  - Images: {len(val_coco['images'])}")
    print(f"  - Annotations: {len(val_coco['annotations'])}")
    print(f"  - Negative examples: {val_neg}")
    print(f"  - Class distribution: {val_class_dist}")
    print("=" * 80)


if __name__ == "__main__":
    main()
