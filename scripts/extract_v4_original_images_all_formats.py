#!/usr/bin/env python3
"""
Extract all original (non-augmented) images from v4 dataset into a separate folder.
Includes both PNG and JPG files.

This script:
1. Copies all aug0 images from train folder (original training images) - both .png and .jpg
2. Copies all images from valid folder (all are originals) - both .png and .jpg
3. Copies all images from test folder (all are originals) - both .png and .jpg
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm


def extract_original_images(v4_dataset_path, output_folder):
    """
    Extract all original images from v4 dataset (PNG and JPG).

    Args:
        v4_dataset_path: Path to v4/6class_coco directory
        output_folder: Path to output folder for original images
    """
    v4_path = Path(v4_dataset_path)
    output_path = Path(output_folder)

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Extracting original images from: {v4_path}")
    print(f"Output directory: {output_path}")
    print("-" * 80)

    total_copied = 0
    seen_files = set()
    duplicates = []

    # Supported image formats
    image_extensions = ['.png', '.jpg', '.jpeg']

    # 1. Extract aug0 images from train folder (both PNG and JPG)
    train_folder = v4_path / "train"
    if train_folder.exists():
        print("\n[1/3] Processing train folder (aug0 images only - PNG & JPG)...")

        # Get all aug0 images (both PNG and JPG)
        train_images = []
        for ext in image_extensions:
            train_images.extend([f for f in train_folder.glob(f'*_aug0{ext}')])

        for img_path in tqdm(sorted(train_images), desc="Copying train/aug0"):
            # Remove _aug0 suffix from filename
            new_name = img_path.name.replace('_aug0.png', '.png').replace('_aug0.jpg', '.jpg').replace('_aug0.jpeg', '.jpeg')
            dest_path = output_path / new_name

            if new_name not in seen_files:
                shutil.copy2(img_path, dest_path)
                seen_files.add(new_name)
                total_copied += 1
            else:
                duplicates.append(('train', new_name))

        print(f"   Copied {total_copied} original images from train folder")

    # 2. Extract all images from valid folder (both PNG and JPG)
    valid_folder = v4_path / "valid"
    valid_copied = 0
    if valid_folder.exists():
        print("\n[2/3] Processing valid folder (all images are originals - PNG & JPG)...")

        # Get all images (both PNG and JPG)
        valid_images = []
        for ext in image_extensions:
            valid_images.extend([f for f in valid_folder.glob(f'*{ext}')])

        for img_path in tqdm(sorted(valid_images), desc="Copying valid"):
            dest_path = output_path / img_path.name

            if img_path.name not in seen_files:
                shutil.copy2(img_path, dest_path)
                seen_files.add(img_path.name)
                total_copied += 1
                valid_copied += 1
            else:
                duplicates.append(('valid', img_path.name))

        print(f"   Copied {valid_copied} new images from valid folder")
        if len(valid_images) - valid_copied > 0:
            print(f"   Skipped {len(valid_images) - valid_copied} duplicates already in train")

    # 3. Extract all images from test folder (both PNG and JPG)
    test_folder = v4_path / "test"
    test_copied = 0
    if test_folder.exists():
        print("\n[3/3] Processing test folder (all images are originals - PNG & JPG)...")

        # Get all images (both PNG and JPG)
        test_images = []
        for ext in image_extensions:
            test_images.extend([f for f in test_folder.glob(f'*{ext}')])

        for img_path in tqdm(sorted(test_images), desc="Copying test"):
            dest_path = output_path / img_path.name

            if img_path.name not in seen_files:
                shutil.copy2(img_path, dest_path)
                seen_files.add(img_path.name)
                total_copied += 1
                test_copied += 1
            else:
                duplicates.append(('test', img_path.name))

        print(f"   Copied {test_copied} new images from test folder")
        if len(test_images) - test_copied > 0:
            print(f"   Skipped {len(test_images) - test_copied} duplicates already in train/valid")

    # Summary
    print("\n" + "=" * 80)
    print(f"COMPLETE: Extracted {total_copied} unique original images (PNG & JPG)")
    print(f"Total duplicates across splits: {len(duplicates)}")

    # Count by format
    png_count = sum(1 for f in seen_files if f.endswith('.png'))
    jpg_count = sum(1 for f in seen_files if f.endswith(('.jpg', '.jpeg')))
    print(f"\nBreakdown by format:")
    print(f"  PNG: {png_count}")
    print(f"  JPG: {jpg_count}")
    print("=" * 80)

    return total_copied, len(duplicates)


if __name__ == "__main__":
    # Configuration
    V4_DATASET_PATH = "/Users/yugambahl/Desktop/brain_ct/data/training_datasets/v4/6class_coco"
    OUTPUT_FOLDER = "/Users/yugambahl/Desktop/brain_ct/data/training_datasets/v4/original_images_only"

    # Run extraction
    total, duplicates = extract_original_images(V4_DATASET_PATH, OUTPUT_FOLDER)

    print(f"\nAll {total} unique original images have been extracted successfully!")
    print(f"({duplicates} duplicates were found across splits)")
    print(f"Location: {OUTPUT_FOLDER}")
