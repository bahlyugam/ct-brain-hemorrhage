#!/usr/bin/env python3
"""
Create COCO Annotations for Downloaded No-Hemorrhage Images

Creates a COCO format annotation file for no-hemorrhage images downloaded
from the CSV. These images have NO annotations (empty annotations list).

Usage:
    python scripts/create_no_hemorrhage_coco_annotations.py
    python scripts/create_no_hemorrhage_coco_annotations.py --image-dir /path/to/images
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from PIL import Image
from tqdm import tqdm

# Version-specific configuration
VERSION_CONFIG = {
    'v4': {
        'image_dir': '/Users/yugambahl/Desktop/brain_ct/data/training_datasets/no_hemorrhage_downloads',
    },
    'v5': {
        'image_dir': '/Users/yugambahl/Desktop/brain_ct/data/training_datasets/v5_no_hemorrhage_downloads',
    }
}


def log(message: str):
    """Print a timestamped log message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def create_coco_annotations_for_no_hemorrhage_images(
    image_dir: str,
    output_json_path: str,
    categories: list = None
):
    """
    Create COCO format annotations for no-hemorrhage images.

    Args:
        image_dir: Directory containing PNG images
        output_json_path: Path to save _annotations.coco.json
        categories: List of category dicts (default: 6-class hemorrhage categories)
    """
    if categories is None:
        # Default to V4 6-class categories
        categories = [
            {'id': 0, 'name': 'EDH', 'supercategory': 'hemorrhage'},
            {'id': 1, 'name': 'HC', 'supercategory': 'hemorrhage'},
            {'id': 2, 'name': 'IPH', 'supercategory': 'hemorrhage'},
            {'id': 3, 'name': 'IVH', 'supercategory': 'hemorrhage'},
            {'id': 4, 'name': 'SAH', 'supercategory': 'hemorrhage'},
            {'id': 5, 'name': 'SDH', 'supercategory': 'hemorrhage'}
        ]

    log(f"Scanning image directory: {image_dir}")

    # Find all PNG images
    image_files = list(Path(image_dir).glob('*.png'))
    log(f"  Found {len(image_files)} PNG images")

    if len(image_files) == 0:
        log("⚠️ No images found - nothing to annotate")
        return

    # Create COCO structure
    coco_data = {
        'info': {
            'description': 'No-Hemorrhage Images from CSV Feedback',
            'version': '1.0',
            'year': 2025,
            'contributor': 'Brain CT Hemorrhage Detection Pipeline',
            'date_created': datetime.now().isoformat()
        },
        'licenses': [],
        'images': [],
        'annotations': [],  # Empty - no hemorrhages in these images
        'categories': categories
    }

    # Process each image
    log("Processing images...")
    for img_idx, img_path in enumerate(tqdm(image_files), start=1):
        try:
            # Open image to get dimensions
            with Image.open(img_path) as img:
                width, height = img.size

            # Create image entry
            image_entry = {
                'id': img_idx,
                'file_name': img_path.name,
                'width': width,
                'height': height,
                'date_captured': datetime.now().isoformat(),
                'license': 0,
                'coco_url': '',
                'flickr_url': ''
            }

            coco_data['images'].append(image_entry)

        except Exception as e:
            log(f"  ⚠️ Error processing {img_path.name}: {e}")
            continue

    # Save COCO JSON
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(coco_data, f, indent=2)

    log(f"\n✅ COCO annotations saved to: {output_json_path}")
    log(f"   Images: {len(coco_data['images'])}")
    log(f"   Annotations: {len(coco_data['annotations'])} (empty - no hemorrhages)")
    log(f"   Categories: {len(coco_data['categories'])}")


def main():
    parser = argparse.ArgumentParser(
        description='Create COCO annotations for downloaded no-hemorrhage images'
    )
    parser.add_argument(
        '--image-dir',
        type=str,
        default='/Users/yugambahl/Desktop/brain_ct/data/training_datasets/no_hemorrhage_downloads',
        help='Directory containing downloaded PNG images'
    )
    parser.add_argument(
        '--output-json',
        type=str,
        default='/Users/yugambahl/Desktop/brain_ct/data/training_datasets/no_hemorrhage_downloads/_annotations.coco.json',
        help='Output path for COCO annotations JSON'
    )
    parser.add_argument(
        '--version',
        type=str,
        default='v4',
        choices=['v4', 'v5'],
        help='Dataset version (default: v4)'
    )

    args = parser.parse_args()

    # Get version-specific configuration
    config = VERSION_CONFIG[args.version]

    # Use version-specific defaults if user didn't override
    if args.image_dir == '/Users/yugambahl/Desktop/brain_ct/data/training_datasets/no_hemorrhage_downloads':
        args.image_dir = config['image_dir']

    if args.output_json == '/Users/yugambahl/Desktop/brain_ct/data/training_datasets/no_hemorrhage_downloads/_annotations.coco.json':
        args.output_json = os.path.join(config['image_dir'], '_annotations.coco.json')

    print("=" * 80)
    print("CREATE COCO ANNOTATIONS FOR NO-HEMORRHAGE IMAGES")
    print("=" * 80)

    if not os.path.exists(args.image_dir):
        log(f"❌ Error: Image directory not found: {args.image_dir}")
        return 1

    create_coco_annotations_for_no_hemorrhage_images(
        image_dir=args.image_dir,
        output_json_path=args.output_json
    )

    print("\n" + "=" * 80)
    print("ANNOTATION CREATION COMPLETE!")
    print("=" * 80)
    print(f"\nNext step:")
    print(f"  Re-run prepare_v4_dataset.py to merge these images into v4 dataset")

    return 0


if __name__ == "__main__":
    exit(main())
