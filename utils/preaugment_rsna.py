#!/usr/bin/env python3
"""
Pre-augmentation script using RSNA ICH 2019 winners' strategy.

Creates 4x dataset with evidence-based augmentations:
1. Original (no transform)
2. Horizontal flip
3. Rotation +10°
4. Rotation -10° + Brightness/Contrast adjustment

Augmentation applied OFFLINE for 8-10x training speedup.
"""

import cv2
import numpy as np
import albumentations as A
from pathlib import Path
import shutil
from tqdm import tqdm
import yaml
import math


def rotate_yolo_bbox(bbox, angle_deg, img_h, img_w):
    """
    Rotate YOLO format bounding box.

    Args:
        bbox: [class_id, x_center, y_center, width, height] (normalized 0-1)
        angle_deg: Rotation angle in degrees (positive = clockwise)
        img_h, img_w: Image dimensions

    Returns:
        Rotated bbox in YOLO format
    """
    class_id, x_center, y_center, width, height = bbox

    # Convert normalized coordinates to pixel coordinates
    x_center_px = x_center * img_w
    y_center_px = y_center * img_h
    width_px = width * img_w
    height_px = height * img_h

    # Convert YOLO center format to corner format (x1, y1, x2, y2)
    x1 = x_center_px - width_px / 2
    y1 = y_center_px - height_px / 2
    x2 = x_center_px + width_px / 2
    y2 = y_center_px + height_px / 2

    # Get 4 corners
    corners = np.array([
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2]
    ])

    # Rotation matrix (around image center)
    angle_rad = -angle_deg * math.pi / 180  # Negative for clockwise
    center_x, center_y = img_w / 2, img_h / 2

    # Translate to origin, rotate, translate back
    corners_centered = corners - np.array([center_x, center_y])

    rotation_matrix = np.array([
        [math.cos(angle_rad), -math.sin(angle_rad)],
        [math.sin(angle_rad), math.cos(angle_rad)]
    ])

    rotated_corners = corners_centered @ rotation_matrix.T
    rotated_corners += np.array([center_x, center_y])

    # Get bounding box of rotated corners
    x_min = rotated_corners[:, 0].min()
    y_min = rotated_corners[:, 1].min()
    x_max = rotated_corners[:, 0].max()
    y_max = rotated_corners[:, 1].max()

    # Clip to image boundaries
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(img_w, x_max)
    y_max = min(img_h, y_max)

    # Convert back to YOLO format (normalized)
    new_x_center = ((x_min + x_max) / 2) / img_w
    new_y_center = ((y_min + y_max) / 2) / img_h
    new_width = (x_max - x_min) / img_w
    new_height = (y_max - y_min) / img_h

    # Ensure valid bbox
    new_x_center = np.clip(new_x_center, 0, 1)
    new_y_center = np.clip(new_y_center, 0, 1)
    new_width = np.clip(new_width, 0, 1)
    new_height = np.clip(new_height, 0, 1)

    return [class_id, new_x_center, new_y_center, new_width, new_height]


def flip_yolo_bbox(bbox):
    """
    Flip YOLO format bounding box horizontally.

    Args:
        bbox: [class_id, x_center, y_center, width, height]

    Returns:
        Flipped bbox
    """
    class_id, x_center, y_center, width, height = bbox
    new_x_center = 1.0 - x_center
    return [class_id, new_x_center, y_center, width, height]


def load_yolo_label(label_path):
    """Load YOLO format label file."""
    bboxes = []
    if not label_path.exists():
        return bboxes

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                coords = [float(x) for x in parts[1:5]]
                bboxes.append([class_id] + coords)
    return bboxes


def save_yolo_label(label_path, bboxes):
    """Save YOLO format label file."""
    with open(label_path, 'w') as f:
        for bbox in bboxes:
            class_id = int(bbox[0])
            coords = bbox[1:5]
            f.write(f"{class_id} {' '.join([f'{c:.6f}' for c in coords])}\n")


def augment_image_and_label(img_path, label_path, output_img_dir, output_label_dir):
    """
    Apply 4 augmentations to a single image and its label.

    Returns:
        List of (augmented_img_path, augmented_label_path) tuples
    """
    # Read image
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Warning: Could not read image {img_path}")
        return []

    img_h, img_w = img.shape[:2]

    # Load bounding boxes
    bboxes = load_yolo_label(label_path)

    # Base filename (without extension)
    base_name = img_path.stem
    img_ext = img_path.suffix

    results = []

    # Define augmentations
    augmentations = {
        'original': {
            'transform': None,
            'bbox_transform': None
        },
        'flip': {
            'transform': A.HorizontalFlip(p=1.0),
            'bbox_transform': flip_yolo_bbox
        },
        'rot10': {
            'transform': A.Rotate(
                limit=(10, 10),
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=1.0
            ),
            'bbox_transform': lambda bbox: rotate_yolo_bbox(bbox, 10, img_h, img_w)
        },
        'rot10_bc': {
            'transform': A.Compose([
                A.Rotate(
                    limit=(-10, -10),
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=1.0
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=(0.15, 0.15),  # +15%
                    contrast_limit=(0.20, 0.20),    # +20%
                    p=1.0
                )
            ]),
            'bbox_transform': lambda bbox: rotate_yolo_bbox(bbox, -10, img_h, img_w)
        }
    }

    # Apply each augmentation
    for aug_name, aug_config in augmentations.items():
        # Output paths
        output_img_path = output_img_dir / f"{base_name}_{aug_name}{img_ext}"
        output_label_path = output_label_dir / f"{base_name}_{aug_name}.txt"

        # Apply image transformation
        if aug_config['transform'] is None:
            # Original - just copy
            augmented_img = img.copy()
        else:
            # Apply albumentation
            augmented = aug_config['transform'](image=img)
            augmented_img = augmented['image']

        # Transform bounding boxes
        if aug_config['bbox_transform'] is None:
            # Original - just copy
            augmented_bboxes = bboxes.copy()
        else:
            # Apply bbox transformation
            augmented_bboxes = [aug_config['bbox_transform'](bbox) for bbox in bboxes]

        # Save augmented image
        cv2.imwrite(str(output_img_path), augmented_img)

        # Save augmented label
        save_yolo_label(output_label_path, augmented_bboxes)

        results.append((output_img_path, output_label_path))

    return results


def process_dataset(input_base_dir, output_base_dir, splits=['train']):
    """
    Process entire dataset.

    Args:
        input_base_dir: Path to original dataset
        output_base_dir: Path to output augmented dataset
        splits: List of splits to augment (default: only 'train')
    """
    input_base = Path(input_base_dir)
    output_base = Path(output_base_dir)

    print("\n" + "="*80)
    print("PRE-AUGMENTATION: RSNA ICH 2019 Winners Strategy")
    print("="*80)
    print(f"Input: {input_base}")
    print(f"Output: {output_base}")
    print(f"Augmentations: 4x (original, flip, rot+10°, rot-10°+brightness/contrast)")
    print("="*80 + "\n")

    # Create output directories
    output_base.mkdir(parents=True, exist_ok=True)

    # Process each split
    for split in ['train', 'valid', 'test']:
        print(f"\n{'='*80}")
        print(f"Processing {split.upper()} split")
        print(f"{'='*80}")

        input_img_dir = input_base / split / 'images'
        input_label_dir = input_base / split / 'labels'

        if not input_img_dir.exists():
            print(f"Warning: {input_img_dir} does not exist, skipping...")
            continue

        output_img_dir = output_base / split / 'images'
        output_label_dir = output_base / split / 'labels'
        output_img_dir.mkdir(parents=True, exist_ok=True)
        output_label_dir.mkdir(parents=True, exist_ok=True)

        # Get all images
        image_files = list(input_img_dir.glob('*.jpg')) + list(input_img_dir.glob('*.png'))

        print(f"Found {len(image_files)} images in {split} split")

        if split in splits:
            # Augment this split
            print(f"Applying 4x augmentation...")

            success_count = 0
            for img_path in tqdm(image_files, desc=f"Augmenting {split}"):
                label_path = input_label_dir / f"{img_path.stem}.txt"

                try:
                    results = augment_image_and_label(
                        img_path, label_path,
                        output_img_dir, output_label_dir
                    )
                    if results:
                        success_count += 1
                except Exception as e:
                    print(f"\nError processing {img_path}: {e}")

            output_count = len(list(output_img_dir.glob('*')))
            print(f"✓ Augmented {success_count} images")
            print(f"✓ Generated {output_count} total images ({len(image_files)} × 4)")
        else:
            # Just copy this split (no augmentation for valid/test)
            print(f"Copying {split} split (no augmentation)...")

            for img_path in tqdm(image_files, desc=f"Copying {split}"):
                label_path = input_label_dir / f"{img_path.stem}.txt"

                # Copy image
                shutil.copy2(img_path, output_img_dir / img_path.name)

                # Copy label if exists
                if label_path.exists():
                    shutil.copy2(label_path, output_label_dir / label_path.name)

            print(f"✓ Copied {len(image_files)} images")

    # Copy and update data.yaml
    print(f"\n{'='*80}")
    print("Creating data.yaml")
    print(f"{'='*80}")

    input_yaml = input_base / 'data.yaml'
    output_yaml = output_base / 'data.yaml'

    if input_yaml.exists():
        with open(input_yaml, 'r') as f:
            data_config = yaml.safe_load(f)

        # Update path
        data_config['path'] = str(output_base)

        # Save
        with open(output_yaml, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)

        print(f"✓ Created {output_yaml}")

    # Print summary
    print(f"\n{'='*80}")
    print("AUGMENTATION COMPLETE")
    print(f"{'='*80}")

    for split in ['train', 'valid', 'test']:
        output_img_dir = output_base / split / 'images'
        if output_img_dir.exists():
            count = len(list(output_img_dir.glob('*')))
            print(f"{split.upper():<10}: {count:>6} images")

    print(f"{'='*80}\n")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Pre-augment dataset with RSNA strategy')
    parser.add_argument(
        '--input',
        type=str,
        default='/Users/yugambahl/Desktop/brain_ct/data/UAT_CT_BRAIN_HEMORRHAGE.v2i.yolov8_combined',
        help='Input dataset directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='/Users/yugambahl/Desktop/brain_ct/data/UAT_CT_BRAIN_HEMORRHAGE.v2i.yolov8_combined_augmented',
        help='Output augmented dataset directory'
    )
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train'],
        help='Splits to augment (default: train only)'
    )

    args = parser.parse_args()

    process_dataset(args.input, args.output, args.splits)

    print("\n✓ Pre-augmentation complete!")
    print(f"✓ Now update train.py to use: {args.output}")
    print(f"✓ Set USE_PREAUGMENTED = True\n")


if __name__ == '__main__':
    main()