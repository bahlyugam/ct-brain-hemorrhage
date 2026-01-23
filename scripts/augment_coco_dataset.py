"""
Offline COCO Dataset Augmentation Script

Multiplies the training dataset by applying aggressive augmentations to prevent overfitting.
Generates multiple augmented versions of each training image while keeping validation/test sets unchanged.

Usage:
    python scripts/augment_coco_dataset.py --input /data/filtered_4class/coco --output /data/augmented_4class/coco --multiplier 3
"""

import os
import json
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import cv2
import numpy as np
from tqdm import tqdm
import albumentations as A


def clip_bbox_to_image(bbox: List[float], img_width: int, img_height: int) -> Optional[List[float]]:
    """
    Clip bounding box to image boundaries and validate.

    Args:
        bbox: [x, y, width, height] in COCO format (pixels)
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        Clipped bbox [x, y, width, height] or None if zero-area after clipping
    """
    x, y, w, h = bbox

    # Clip coordinates to image boundaries
    x = max(0.0, x)
    y = max(0.0, y)
    x_max = min(float(img_width), x + w)
    y_max = min(float(img_height), y + h)

    # Recalculate width and height after clipping
    w = x_max - x
    h = y_max - y

    # Return None if clipping results in zero-area bbox
    if w <= 0 or h <= 0:
        return None

    return [x, y, w, h]


def load_coco_annotations(json_path: str) -> Dict[str, Any]:
    """Load COCO format annotations JSON."""
    with open(json_path, 'r') as f:
        return json.load(f)


def save_coco_annotations(annotations: Dict[str, Any], json_path: str):
    """Save COCO format annotations JSON."""
    with open(json_path, 'w') as f:
        json.dump(annotations, f, indent=2)


def get_augmentation_pipeline(image_size: int = 640, strength: str = 'aggressive') -> A.Compose:
    """
    Get augmentation pipeline for offline dataset multiplication.

    Args:
        image_size: Target image size
        strength: 'aggressive', 'medium', or 'light'

    Returns:
        Albumentations Compose object
    """
    if strength == 'aggressive':
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=20,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.8
            ),
            A.RandomSizedBBoxSafeCrop(
                height=image_size,
                width=image_size,
                erosion_rate=0.0,
                p=0.6
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.6
            ),
            A.CLAHE(
                clip_limit=4.0,
                tile_grid_size=(8, 8),
                p=0.3
            ),
            A.CoarseDropout(
                max_holes=12,
                max_height=24,
                max_width=24,
                min_holes=4,
                min_height=8,
                min_width=8,
                fill_value=0,
                p=0.4
            ),
            A.LongestMaxSize(max_size=image_size, interpolation=cv2.INTER_LINEAR),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=0
            ),
        ], bbox_params=A.BboxParams(
            format='coco',
            label_fields=['category_ids'],
            min_visibility=0.3,
            min_area=400
        ))
    elif strength == 'medium':
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.075,
                scale_limit=0.15,
                rotate_limit=15,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.6
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=0.5
            ),
            A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.2),
            A.LongestMaxSize(max_size=image_size, interpolation=cv2.INTER_LINEAR),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=0
            ),
        ], bbox_params=A.BboxParams(
            format='coco',
            label_fields=['category_ids'],
            min_visibility=0.3,
            min_area=400
        ))
    else:  # light
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=10,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.4
            ),
            A.LongestMaxSize(max_size=image_size, interpolation=cv2.INTER_LINEAR),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=0
            ),
        ], bbox_params=A.BboxParams(
            format='coco',
            label_fields=['category_ids'],
            min_visibility=0.3,
            min_area=400
        ))


def augment_coco_dataset(
    input_dir: str,
    output_dir: str,
    multiplier: int = 3,
    strength: str = 'aggressive',
    image_size: int = 640
):
    """
    Create augmented version of COCO dataset.

    Args:
        input_dir: Path to original COCO dataset (containing train/, valid/, test/)
        output_dir: Path to save augmented dataset
        multiplier: Dataset size multiplier (e.g., 3 = 2 augmented + 1 original per image)
        strength: Augmentation strength ('aggressive', 'medium', 'light')
        image_size: Target image size
    """
    print(f"\n{'='*80}")
    print(f"AUGMENTING COCO DATASET")
    print(f"{'='*80}")
    print(f"Input:      {input_dir}")
    print(f"Output:     {output_dir}")
    print(f"Multiplier: {multiplier}x (original + {multiplier-1} augmented versions)")
    print(f"Strength:   {strength}")
    print(f"{'='*80}\n")

    # Process each split
    for split in ['train', 'valid', 'test']:
        split_input = Path(input_dir) / split
        split_output = Path(output_dir) / split

        # Check if split exists
        if not split_input.exists():
            print(f"⚠ Skipping {split} (not found)")
            continue

        print(f"\n📂 Processing {split} split...")

        # Create output directory (flat structure, no images/ subdirectory)
        split_output.mkdir(parents=True, exist_ok=True)

        # Load annotations
        ann_file = split_input / '_annotations.coco.json'
        if not ann_file.exists():
            print(f"⚠ No annotations found for {split}, skipping")
            continue

        coco_data = load_coco_annotations(str(ann_file))

        # For validation and test, just copy (no augmentation)
        if split in ['valid', 'test']:
            print(f"  Copying {len(coco_data['images'])} images (no augmentation)...")

            # Copy images (flat structure - images are in split_input/ not split_input/images/)
            for img_info in tqdm(coco_data['images']):
                src = split_input / img_info['file_name']
                dst = split_output / img_info['file_name']
                if not src.exists():
                    print(f"  ⚠ File not found: {src}, skipping")
                    continue
                shutil.copy2(src, dst)

            # Copy annotations
            save_coco_annotations(coco_data, str(split_output / '_annotations.coco.json'))
            print(f"  ✓ Copied {len(coco_data['images'])} images")
            continue

        # For training, apply augmentation
        print(f"  Augmenting {len(coco_data['images'])} images × {multiplier}...")

        transform = get_augmentation_pipeline(image_size, strength)

        # New COCO data structure
        new_coco_data = {
            'info': coco_data.get('info', {}),
            'licenses': coco_data.get('licenses', []),
            'categories': coco_data['categories'],
            'images': [],
            'annotations': []
        }

        next_image_id = 1
        next_ann_id = 1

        # Create image_id -> annotations mapping
        image_to_anns = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in image_to_anns:
                image_to_anns[img_id] = []
            image_to_anns[img_id].append(ann)

        # Process each image
        for img_info in tqdm(coco_data['images']):
            # Flat structure - images are in split_input/ not split_input/images/
            img_path = split_input / img_info['file_name']

            if not img_path.exists():
                print(f"  ⚠ File not found: {img_path}, skipping")
                continue

            image = cv2.imread(str(img_path))

            if image is None:
                print(f"  ⚠ Failed to load {img_info['file_name']}, skipping")
                continue

            # Get annotations for this image
            img_anns = image_to_anns.get(img_info['id'], [])

            # Convert annotations to COCO bbox format
            bboxes = []
            category_ids = []
            for ann in img_anns:
                bboxes.append(ann['bbox'])  # Already in COCO format [x, y, w, h]
                category_ids.append(ann['category_id'])

            # Generate multiplier versions (original + augmented)
            for version in range(multiplier):
                if version == 0:
                    # First version: original (no augmentation, just resize)
                    aug_image = image.copy()
                    aug_bboxes = bboxes.copy()
                    aug_category_ids = category_ids.copy()
                else:
                    # Apply augmentation
                    try:
                        augmented = transform(
                            image=image,
                            bboxes=bboxes,
                            category_ids=category_ids
                        )
                        aug_image = augmented['image']
                        aug_bboxes = augmented['bboxes']
                        aug_category_ids = augmented['category_ids']
                    except Exception as e:
                        print(f"  ⚠ Augmentation failed for {img_info['file_name']} v{version}: {e}")
                        continue

                # Save augmented image (flat structure - directly in split_output)
                base_name = Path(img_info['file_name']).stem
                ext = Path(img_info['file_name']).suffix
                new_filename = f"{base_name}_aug{version}{ext}"
                new_img_path = split_output / new_filename

                cv2.imwrite(str(new_img_path), aug_image)

                # Add image info
                new_img_info = {
                    'id': next_image_id,
                    'file_name': new_filename,
                    'width': aug_image.shape[1],
                    'height': aug_image.shape[0],
                }
                new_coco_data['images'].append(new_img_info)

                # Add annotations with bbox clipping
                img_width = aug_image.shape[1]
                img_height = aug_image.shape[0]

                for bbox, cat_id in zip(aug_bboxes, aug_category_ids):
                    # Clip bbox to image boundaries
                    clipped_bbox = clip_bbox_to_image(list(bbox), img_width, img_height)

                    if clipped_bbox is None:
                        # Skip zero-area boxes after clipping
                        continue

                    new_ann = {
                        'id': next_ann_id,
                        'image_id': next_image_id,
                        'category_id': cat_id,
                        'bbox': clipped_bbox,  # Clipped [x, y, w, h]
                        'area': clipped_bbox[2] * clipped_bbox[3],  # Recalculated w * h
                        'iscrowd': 0,
                        'segmentation': []
                    }
                    new_coco_data['annotations'].append(new_ann)
                    next_ann_id += 1

                next_image_id += 1

        # Save augmented annotations
        save_coco_annotations(new_coco_data, str(split_output / '_annotations.coco.json'))

        print(f"  ✓ Created {len(new_coco_data['images'])} augmented images")
        print(f"  ✓ Created {len(new_coco_data['annotations'])} augmented annotations")

    print(f"\n{'='*80}")
    print(f"✅ Dataset augmentation complete!")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Augment COCO dataset to prevent overfitting')
    parser.add_argument('--input', type=str, required=True,
                        help='Input COCO dataset directory')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for augmented dataset')
    parser.add_argument('--multiplier', type=int, default=3,
                        help='Dataset size multiplier (default: 3)')
    parser.add_argument('--strength', type=str, default='aggressive',
                        choices=['light', 'medium', 'aggressive'],
                        help='Augmentation strength (default: aggressive)')
    parser.add_argument('--image-size', type=int, default=640,
                        help='Target image size (default: 640)')

    args = parser.parse_args()

    augment_coco_dataset(
        input_dir=args.input,
        output_dir=args.output,
        multiplier=args.multiplier,
        strength=args.strength,
        image_size=args.image_size
    )


if __name__ == '__main__':
    main()
