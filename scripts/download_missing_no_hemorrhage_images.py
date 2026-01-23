#!/usr/bin/env python3
"""
Download Missing No-Hemorrhage Images from CSV

Downloads images from no_hemorrhage_positive_feedback.csv that are NOT currently
in the v4 dataset. Prioritizes False Positives (FP) over True Negatives (TN).

Usage:
    python scripts/download_missing_no_hemorrhage_images.py
    python scripts/download_missing_no_hemorrhage_images.py --max-images 2000 --fp-only
"""

import os
import sys
import csv
import json
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Set, Tuple

# Add parent directory to path for imports
sys.path.append('/Users/yugambahl/Desktop/brain_ct')
from utils.download_negative_feedback_images import download_from_s3, process_dicom_to_png
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Version-specific configuration
VERSION_CONFIG = {
    'v4': {
        'dataset_dirs': [
            '/Users/yugambahl/Desktop/brain_ct/data/training_datasets/v4/6class_coco',
            '/Users/yugambahl/Desktop/brain_ct/data/training_datasets/v3/filtered_4class/coco',
            '/Users/yugambahl/Desktop/brain_ct/data/training_datasets/roboflow_downloads/UAT_CT_BRAIN_HEMORRHAGE_V3.v3i.coco'
        ],
        'output_dir': '/Users/yugambahl/Desktop/brain_ct/data/training_datasets/no_hemorrhage_downloads',
    },
    'v5': {
        'dataset_dirs': [
            # V4 6-class splits (to avoid re-downloading)
            '/Users/yugambahl/Desktop/brain_ct/data/training_datasets/v4/6class_coco',
            # V4's existing no-hem downloads
            '/Users/yugambahl/Desktop/brain_ct/data/training_datasets/no_hemorrhage_downloads',
            # V5 new images
            '/Users/yugambahl/Desktop/brain_ct/data/training_datasets/v5',
        ],
        'output_dir': '/Users/yugambahl/Desktop/brain_ct/data/training_datasets/v5_no_hemorrhage_downloads',
    }
}


def log(message: str):
    """Print a timestamped log message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def extract_patient_id_and_instance(filename: str) -> Tuple[str, str]:
    """
    Extract both patient_id and instance_no from filename.

    Formats handled:
    - v3_thick_520892552_14.png → (520892552, 14)
    - roboflow_525190930_29_png.rf.xxx.jpg → (525190930, 29)
    - v3_12345_6_aug0.jpg → (12345, 6)
    - v5_12345_6_png.jpg → (12345, 6)

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

    # Remove _png suffix (V5 Roboflow format)
    if base.endswith('_png'):
        base = base[:-4]

    # Remove prefixes
    prefixes = ['v3_', 'v4_', 'v5_', 'roboflow_', 'thin_', 'thick_', 'no_hem_']
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if base.startswith(prefix):
                base = base[len(prefix):]
                changed = True
                break

    # Extract patient_id and instance_no
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


def load_existing_dataset_images(dataset_dirs: List[str]) -> Set[Tuple[str, str]]:
    """
    Load all existing images from dataset directories.

    Args:
        dataset_dirs: List of directories to scan for COCO annotations

    Returns:
        Set of (patient_id, instance_no) tuples present in dataset
    """
    existing_images = set()

    for dataset_dir in dataset_dirs:
        # Check if directory exists
        if not os.path.exists(dataset_dir):
            logger.warning(f"Directory not found: {dataset_dir}")
            continue

        # Try loading from COCO JSON with split structure (train/valid/test)
        has_splits = False
        for split in ['train', 'valid', 'test']:
            json_path = os.path.join(dataset_dir, split, '_annotations.coco.json')

            if os.path.exists(json_path):
                has_splits = True
                with open(json_path, 'r') as f:
                    data = json.load(f)

                for img in data['images']:
                    filename = img['file_name']
                    patient_id, instance_no = extract_patient_id_and_instance(filename)
                    existing_images.add((patient_id, instance_no))

        # If no splits found, try flat structure (V5 roboflow format)
        if not has_splits:
            json_path = os.path.join(dataset_dir, '_annotations.coco.json')
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)

                for img in data['images']:
                    filename = img['file_name']
                    patient_id, instance_no = extract_patient_id_and_instance(filename)
                    existing_images.add((patient_id, instance_no))
            else:
                # Fallback: scan directory for image files
                for fname in os.listdir(dataset_dir):
                    if fname.endswith(('.png', '.jpg', '.jpeg')):
                        patient_id, instance_no = extract_patient_id_and_instance(fname)
                        existing_images.add((patient_id, instance_no))

    return existing_images


def load_csv_with_priority(csv_path: str) -> Tuple[List[Dict], List[Dict]]:
    """
    Load CSV and separate into FP (high priority) and TN (low priority) lists.

    Args:
        csv_path: Path to no_hemorrhage_positive_feedback.csv

    Returns:
        (fp_images, tn_images) - Lists of dicts with patient_id, instance_no, image_url
    """
    fp_images = []
    tn_images = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_data = {
                'patient_id': row['patient_id'],
                'instance_no': row['instance_no'],
                'image_url': row.get('image_url', ''),
                'classification': row['classification_result']
            }

            if row['classification_result'] == 'False Positive (FP)':
                fp_images.append(image_data)
            elif row['classification_result'] == 'True Negative (TN)':
                tn_images.append(image_data)

    return fp_images, tn_images


def filter_missing_images(
    csv_images: List[Dict],
    existing_images: Set[Tuple[str, str]]
) -> List[Dict]:
    """
    Filter out images that already exist in the dataset.

    Args:
        csv_images: List of image dicts from CSV
        existing_images: Set of (patient_id, instance_no) already in dataset

    Returns:
        List of missing image dicts
    """
    missing = []

    for img_data in csv_images:
        key = (str(img_data['patient_id']), str(img_data['instance_no']))
        if key not in existing_images:
            missing.append(img_data)

    return missing


def download_images(
    images_to_download: List[Dict],
    output_dir: str,
    temp_dir: str,
    use_credentials: bool = True,
    aws_access_key: str = None,
    aws_secret_key: str = None,
    filter_slice_thickness: bool = True
) -> Tuple[int, int, int]:
    """
    Download and convert DICOM images to PNG.

    Args:
        images_to_download: List of image dicts with patient_id, instance_no, image_url
        output_dir: Directory to save PNG files
        temp_dir: Temporary directory for DICOM files
        use_credentials: Whether to use AWS credentials
        aws_access_key: AWS access key
        aws_secret_key: AWS secret key
        filter_slice_thickness: Whether to filter by slice thickness

    Returns:
        (successful, skipped, failed) counts
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    successful = 0
    failed = 0
    skipped = 0

    for img_data in tqdm(images_to_download, desc="Downloading images"):
        patient_id = str(img_data['patient_id'])
        instance_no = str(img_data['instance_no'])
        url = img_data['image_url']

        # Skip if no URL
        if not url or url == '':
            logger.warning(f"No image_url for {patient_id}_{instance_no}, skipping")
            failed += 1
            continue

        # Generate filenames
        file_id = f"{patient_id}_{instance_no}"
        dicom_path = os.path.join(temp_dir, f"{file_id}.dcm")
        png_path = os.path.join(output_dir, f"{file_id}.png")

        # Skip if PNG already exists
        if os.path.exists(png_path):
            logger.info(f"✅ PNG already exists: {file_id}.png - SKIPPING")
            successful += 1
            continue

        # Check if DICOM already exists
        if os.path.exists(dicom_path):
            logger.info(f"🔄 DICOM already exists, converting: {file_id}.dcm")
            result = process_dicom_to_png(dicom_path, png_path, filter_slice_thickness)
            if result is True:
                successful += 1
                os.remove(dicom_path)
            elif result is False and filter_slice_thickness:
                skipped += 1
                os.remove(dicom_path)
            else:
                failed += 1
        else:
            # Download DICOM
            logger.info(f"📥 Downloading {file_id}: {url}")
            if download_from_s3(url, dicom_path, use_credentials, aws_access_key, aws_secret_key):
                # Convert to PNG
                result = process_dicom_to_png(dicom_path, png_path, filter_slice_thickness)
                if result is True:
                    successful += 1
                elif result is False and filter_slice_thickness:
                    skipped += 1
                else:
                    failed += 1

                # Remove temporary DICOM
                if os.path.exists(dicom_path):
                    os.remove(dicom_path)
            else:
                failed += 1

    # Clean up temp folder if empty
    if os.path.exists(temp_dir) and not os.listdir(temp_dir):
        os.rmdir(temp_dir)

    return successful, skipped, failed


def main():
    parser = argparse.ArgumentParser(
        description='Download missing no-hemorrhage images from CSV to reach 50% balance'
    )
    parser.add_argument(
        '--csv-path',
        type=str,
        default='/Users/yugambahl/Desktop/brain_ct/data/metadata/no_hemorrhage_positive_feedback.csv',
        help='Path to no_hemorrhage_positive_feedback.csv'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/Users/yugambahl/Desktop/brain_ct/data/training_datasets/no_hemorrhage_downloads',
        help='Directory to save downloaded PNG images'
    )
    parser.add_argument(
        '--temp-dir',
        type=str,
        default='/Users/yugambahl/Desktop/brain_ct/data/training_datasets/no_hemorrhage_downloads/temp_dicom',
        help='Temporary directory for DICOM files'
    )
    parser.add_argument(
        '--max-images',
        type=int,
        default=None,
        help='Maximum number of images to download (default: no limit)'
    )
    parser.add_argument(
        '--fp-only',
        action='store_true',
        help='Download only False Positive images (skip True Negatives)'
    )
    parser.add_argument(
        '--use-credentials',
        action='store_true',
        default=True,
        help='Use AWS credentials for S3 access'
    )
    parser.add_argument(
        '--no-filter-thickness',
        action='store_true',
        help='Disable slice thickness filtering (download all slices)'
    )
    parser.add_argument(
        '--version',
        type=str,
        default='v4',
        choices=['v4', 'v5'],
        help='Dataset version (default: v4)'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("DOWNLOAD MISSING NO-HEMORRHAGE IMAGES")
    print("=" * 80)

    # Load AWS credentials from environment if needed
    aws_access_key = None
    aws_secret_key = None
    if args.use_credentials:
        from dotenv import load_dotenv
        load_dotenv()
        aws_access_key = os.getenv("AWS_ACCESS_KEY")
        aws_secret_key = os.getenv("AWS_SECRET_KEY")

        if not aws_access_key or not aws_secret_key:
            log("⚠️ AWS credentials not found in environment. Set AWS_ACCESS_KEY and AWS_SECRET_KEY in .env file")
            log("   Continuing without credentials (will fail for private S3 buckets)")
            args.use_credentials = False

    # Get version-specific configuration
    config = VERSION_CONFIG[args.version]

    # Override output_dir if specified in args
    if args.output_dir != '/Users/yugambahl/Desktop/brain_ct/data/training_datasets/no_hemorrhage_downloads':
        # User specified custom output directory
        output_dir = args.output_dir
    else:
        # Use version-specific default
        output_dir = config['output_dir']

    # [1/6] Load existing dataset images
    log(f"\n[1/6] Scanning existing {args.version} dataset for images...")
    log(f"  Checking directories: {config['dataset_dirs']}")
    existing_images = load_existing_dataset_images(config['dataset_dirs'])
    log(f"  Found {len(existing_images)} existing images in dataset")

    # [2/6] Load CSV with priority separation
    log("\n[2/6] Loading no-hemorrhage CSV...")
    fp_images, tn_images = load_csv_with_priority(args.csv_path)
    log(f"  Total CSV images: {len(fp_images) + len(tn_images)}")
    log(f"    False Positives (FP): {len(fp_images)} (high priority)")
    log(f"    True Negatives (TN): {len(tn_images)} (low priority)")

    # [3/6] Filter out existing images
    log("\n[3/6] Filtering out images already in dataset...")
    missing_fp = filter_missing_images(fp_images, existing_images)
    missing_tn = filter_missing_images(tn_images, existing_images)
    log(f"  Missing from dataset:")
    log(f"    FP: {len(missing_fp)} (need to download)")
    log(f"    TN: {len(missing_tn)} (need to download)")

    # [4/6] Determine download list with priority
    log("\n[4/6] Creating prioritized download list...")
    images_to_download = []

    # Add ALL missing FP images (highest priority)
    images_to_download.extend(missing_fp)
    log(f"  Added {len(missing_fp)} FP images (priority 1)")

    # Add TN images if not fp-only and within max-images limit
    if not args.fp_only:
        if args.max_images:
            remaining_slots = args.max_images - len(images_to_download)
            tn_to_add = missing_tn[:remaining_slots]
        else:
            tn_to_add = missing_tn

        images_to_download.extend(tn_to_add)
        log(f"  Added {len(tn_to_add)} TN images (priority 2)")
    else:
        log(f"  Skipping TN images (--fp-only flag set)")

    # Apply max-images limit
    if args.max_images and len(images_to_download) > args.max_images:
        images_to_download = images_to_download[:args.max_images]
        log(f"  Limited to {args.max_images} images (--max-images)")

    log(f"\n  Total images to download: {len(images_to_download)}")

    if len(images_to_download) == 0:
        log("\n✅ No images to download - all CSV images are already in the dataset!")
        return

    # [5/6] Download images
    log(f"\n[5/6] Downloading images...")
    log(f"  Output directory: {output_dir}")
    log(f"  Slice thickness filtering: {'enabled' if not args.no_filter_thickness else 'disabled'}")

    successful, skipped, failed = download_images(
        images_to_download,
        output_dir,
        args.temp_dir,
        use_credentials=args.use_credentials,
        aws_access_key=aws_access_key,
        aws_secret_key=aws_secret_key,
        filter_slice_thickness=not args.no_filter_thickness
    )

    # [6/6] Summary
    log("\n[6/6] Download Summary")
    log(f"  ✅ Successful: {successful}")
    log(f"  ⏭️  Skipped (slice thickness): {skipped}")
    log(f"  ❌ Failed: {failed}")
    log(f"  📊 Total attempted: {len(images_to_download)}")

    log("\n" + "=" * 80)
    log("DOWNLOAD COMPLETE!")
    log("=" * 80)
    log(f"\nNext steps:")
    log(f"  1. Create COCO annotations for downloaded images")
    log(f"  2. Re-run prepare_v4_dataset.py to include new images")
    log(f"  3. Verify 50% no-hemorrhage balance is achieved")


if __name__ == "__main__":
    main()
