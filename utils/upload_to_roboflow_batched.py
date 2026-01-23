#!/usr/bin/env python3
"""
Upload v4 original images dataset to Roboflow in batches.
This is faster for large datasets as it uploads smaller chunks.
"""
import os
import json
import shutil
from pathlib import Path
from roboflow import Roboflow
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Configuration
WORKSPACE_NAME = "tnsqai-maigj"
PROJECT_NAME = "ctbrain_reverification-smartpolygon_changes"
DATASET_PATH = Path("/Users/yugambahl/Desktop/brain_ct/data/training_datasets/v4/original_images_only")
ANNOTATION_FILE = DATASET_PATH / "_annotations.coco.json"
BATCH_SIZE = 500  # Upload 500 images at a time
TEMP_DIR = DATASET_PATH.parent / "temp_upload_batches"

print("=" * 80)
print("                ROBOFLOW BATCH UPLOAD (FASTER)")
print("=" * 80)
print()
print(f"Workspace: {WORKSPACE_NAME}")
print(f"Project: {PROJECT_NAME}")
print(f"Batch size: {BATCH_SIZE} images per batch")
print()

# Check if API key is set
api_key = os.getenv("ROBOFLOW_API_KEY")
if not api_key:
    print("❌ ROBOFLOW_API_KEY not found in environment variables")
    exit(1)

# Load COCO annotations
print("Loading annotations...")
with open(ANNOTATION_FILE) as f:
    coco_data = json.load(f)

total_images = len(coco_data['images'])
print(f"✓ Found {total_images:,} images with {len(coco_data['annotations']):,} annotations")
print()

# Calculate number of batches
num_batches = (total_images + BATCH_SIZE - 1) // BATCH_SIZE
print(f"Will upload in {num_batches} batches of ~{BATCH_SIZE} images each")
print()

# Initialize Roboflow
print("Connecting to Roboflow...")
rf = Roboflow(api_key=api_key)
workspace = rf.workspace(WORKSPACE_NAME)
project = workspace.project(PROJECT_NAME)
print("✓ Connected")
print()

# Create temp directory for batches
TEMP_DIR.mkdir(exist_ok=True)
print(f"Created temp directory: {TEMP_DIR}")
print()

# Create image_id to annotations mapping
print("Preparing batches...")
annotations_by_image = {}
for ann in coco_data['annotations']:
    img_id = ann['image_id']
    if img_id not in annotations_by_image:
        annotations_by_image[img_id] = []
    annotations_by_image[img_id].append(ann)

# Process batches
print("=" * 80)
print("Starting batch upload...")
print("=" * 80)
print()

for batch_num in range(num_batches):
    start_idx = batch_num * BATCH_SIZE
    end_idx = min((batch_num + 1) * BATCH_SIZE, total_images)
    batch_images = coco_data['images'][start_idx:end_idx]

    print(f"[Batch {batch_num + 1}/{num_batches}] Processing images {start_idx + 1}-{end_idx}...")

    # Create batch directory
    batch_dir = TEMP_DIR / f"batch_{batch_num}"
    batch_dir.mkdir(exist_ok=True)

    # Get batch image IDs
    batch_img_ids = {img['id'] for img in batch_images}

    # Get batch annotations
    batch_annotations = []
    for img_id in batch_img_ids:
        if img_id in annotations_by_image:
            batch_annotations.extend(annotations_by_image[img_id])

    # Create batch COCO JSON
    batch_coco = {
        'info': coco_data.get('info', {}),
        'licenses': coco_data.get('licenses', []),
        'categories': coco_data['categories'],
        'images': batch_images,
        'annotations': batch_annotations
    }

    # Save batch JSON
    batch_json = batch_dir / "_annotations.coco.json"
    with open(batch_json, 'w') as f:
        json.dump(batch_coco, f)

    # Copy images to batch directory
    print(f"   Copying {len(batch_images)} images...")
    for img in tqdm(batch_images, desc="   Copying", leave=False):
        src = DATASET_PATH / img['file_name']
        dst = batch_dir / img['file_name']
        if src.exists():
            shutil.copy2(src, dst)

    # Upload batch
    print(f"   Uploading batch {batch_num + 1}/{num_batches}...")
    try:
        project.upload(
            model_format="coco",
            image_path=str(batch_dir),
            annotation_path=str(batch_json),
            num_workers=10,
            num_retries=3
        )
        print(f"   ✓ Batch {batch_num + 1} uploaded successfully")
    except Exception as e:
        print(f"   ❌ Batch {batch_num + 1} failed: {e}")
        print(f"   Continuing with next batch...")

    # Clean up batch directory
    shutil.rmtree(batch_dir)
    print()

# Clean up temp directory
shutil.rmtree(TEMP_DIR)

print("=" * 80)
print("✓ UPLOAD COMPLETE!")
print("=" * 80)
print()
print(f"Uploaded {total_images:,} images in {num_batches} batches")
print()
print(f"View your project at:")
print(f"  https://app.roboflow.com/{WORKSPACE_NAME}/{PROJECT_NAME}")
print()
