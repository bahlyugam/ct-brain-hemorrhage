#!/usr/bin/env python3
"""
Create COCO annotations for original_images_only folder.

This script:
1. Loads annotations from train/valid/test splits
2. Filters to only include original images (aug0 from train, all from valid/test)
3. Merges annotations while handling duplicates (keeps first occurrence)
4. Creates a new COCO JSON file for the original_images_only folder
"""

import json
from pathlib import Path
from collections import defaultdict


def load_coco_annotations(json_path):
    """Load COCO format annotations."""
    with open(json_path, 'r') as f:
        return json.load(f)


def create_original_images_annotations(v4_dataset_path, original_images_folder):
    """
    Create merged COCO annotations for original images only.

    Args:
        v4_dataset_path: Path to v4/6class_coco directory
        original_images_folder: Path to original_images_only directory
    """
    v4_path = Path(v4_dataset_path)
    output_path = Path(original_images_folder)

    print("=" * 80)
    print("CREATING COCO ANNOTATIONS FOR ORIGINAL IMAGES")
    print("=" * 80)

    # Load all three annotation files
    print("\n[1/4] Loading annotations from splits...")
    train_coco = load_coco_annotations(v4_path / "train" / "_annotations.coco.json")
    valid_coco = load_coco_annotations(v4_path / "valid" / "_annotations.coco.json")
    test_coco = load_coco_annotations(v4_path / "test" / "_annotations.coco.json")

    print(f"   Train: {len(train_coco['images'])} images, {len(train_coco['annotations'])} annotations")
    print(f"   Valid: {len(valid_coco['images'])} images, {len(valid_coco['annotations'])} annotations")
    print(f"   Test:  {len(test_coco['images'])} images, {len(test_coco['annotations'])} annotations")

    # Initialize merged COCO structure
    merged_coco = {
        "info": train_coco.get("info", {}),
        "licenses": train_coco.get("licenses", []),
        "categories": train_coco["categories"],  # Same across all splits
        "images": [],
        "annotations": []
    }

    # Track which images we've already added (to handle duplicates)
    seen_images = set()
    image_id_mapping = {}  # Old image_id -> new image_id
    next_image_id = 0
    next_annotation_id = 0

    # Get list of actual image files in original_images_only folder (PNG and JPG)
    print("\n[2/4] Loading actual image files from original_images_only...")
    actual_image_files = set()
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        actual_image_files.update({f.name for f in output_path.glob(ext)})

    png_count = sum(1 for f in actual_image_files if f.endswith('.png'))
    jpg_count = sum(1 for f in actual_image_files if f.endswith(('.jpg', '.jpeg')))
    print(f"   Found {len(actual_image_files)} image files (PNG: {png_count}, JPG: {jpg_count})")

    print("\n[3/4] Processing images and annotations...")

    # Process each split in order: train -> valid -> test
    # For duplicates, we keep the first occurrence (from train)

    splits_data = [
        ("train", train_coco),
        ("valid", valid_coco),
        ("test", test_coco)
    ]

    for split_name, coco_data in splits_data:
        print(f"\n   Processing {split_name}...")

        images_added = 0
        annotations_added = 0
        duplicates_skipped = 0
        not_in_folder = 0

        # Create mapping of image_id to annotations
        old_annotations = defaultdict(list)
        for ann in coco_data['annotations']:
            old_annotations[ann['image_id']].append(ann)

        # Create a dict of filename -> image data for quick lookup
        filename_to_image = {}
        for img in coco_data['images']:
            old_filename = img['file_name']

            # For train, handle both PNG and JPG aug0 images
            if split_name == "train":
                if old_filename.endswith("_aug0.png"):
                    new_filename = old_filename.replace("_aug0.png", ".png")
                    filename_to_image[new_filename] = img
                elif old_filename.endswith("_aug0.jpg"):
                    new_filename = old_filename.replace("_aug0.jpg", ".jpg")
                    filename_to_image[new_filename] = img
            else:
                filename_to_image[old_filename] = img

        # Now process all actual image files in the folder
        # This ensures we include even images without annotations in the JSON
        for actual_filename in sorted(actual_image_files):
            # Skip if we've already seen this image (duplicate across splits)
            if actual_filename in seen_images:
                duplicates_skipped += 1
                continue

            # Check if this image has metadata in the JSON
            if actual_filename in filename_to_image:
                # Image exists in JSON - use its metadata
                old_img = filename_to_image[actual_filename]

                # Add image with new ID and filename
                new_image = old_img.copy()
                new_image['id'] = next_image_id
                new_image['file_name'] = actual_filename
                merged_coco['images'].append(new_image)

                # Add annotations for this image
                for old_ann in old_annotations[old_img['id']]:
                    new_ann = old_ann.copy()
                    new_ann['id'] = next_annotation_id
                    new_ann['image_id'] = next_image_id
                    merged_coco['annotations'].append(new_ann)
                    annotations_added += 1
                    next_annotation_id += 1
            else:
                # Image exists in folder but not in JSON - create minimal metadata
                # Use default image dimensions (will be updated by Roboflow)
                new_image = {
                    'id': next_image_id,
                    'file_name': actual_filename,
                    'width': 512,  # placeholder
                    'height': 512,  # placeholder
                }
                merged_coco['images'].append(new_image)
                not_in_folder += 1  # Repurposing this counter for "not in JSON"

            seen_images.add(actual_filename)
            images_added += 1
            next_image_id += 1

        print(f"      Added: {images_added} images, {annotations_added} annotations")
        if duplicates_skipped > 0:
            print(f"      Skipped: {duplicates_skipped} duplicates")
        if not_in_folder > 0:
            print(f"      Note: {not_in_folder} images exist in folder but not in JSON (added with placeholder metadata)")

    # Save merged annotations
    print("\n[4/5] Saving merged annotations...")
    output_json = output_path / "_annotations.coco.json"
    with open(output_json, 'w') as f:
        json.dump(merged_coco, f, indent=2)

    print(f"   Saved to: {output_json}")

    # Verification
    print("\n[5/5] Verification...")
    actual_images = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        actual_images.extend(list(output_path.glob(ext)))
    print(f"   Images in folder:     {len(actual_images)}")
    print(f"   Images in COCO JSON:  {len(merged_coco['images'])}")
    print(f"   Total annotations:    {len(merged_coco['annotations'])}")
    print(f"   Categories:           {len(merged_coco['categories'])}")

    # Check if all images have corresponding files
    json_filenames = {img['file_name'] for img in merged_coco['images']}
    actual_filenames = {img.name for img in actual_images}

    missing_files = json_filenames - actual_filenames
    extra_files = actual_filenames - json_filenames

    if missing_files:
        print(f"\n   WARNING: {len(missing_files)} images in JSON but not in folder")
    if extra_files:
        print(f"\n   WARNING: {len(extra_files)} images in folder but not in JSON")

    if not missing_files and not extra_files:
        print("\n   ✓ Perfect match! All images have annotations.")

    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print(f"\nYou can now upload the folder to Roboflow:")
    print(f"  Folder: {output_path}")
    print(f"  Annotations: {output_json}")
    print(f"  Total images: {len(merged_coco['images'])}")
    print(f"  Total annotations: {len(merged_coco['annotations'])}")

    return merged_coco


if __name__ == "__main__":
    V4_DATASET_PATH = "/Users/yugambahl/Desktop/brain_ct/data/training_datasets/v4/6class_coco"
    ORIGINAL_IMAGES_FOLDER = "/Users/yugambahl/Desktop/brain_ct/data/training_datasets/v4/original_images_only"

    merged_coco = create_original_images_annotations(V4_DATASET_PATH, ORIGINAL_IMAGES_FOLDER)
