#!/usr/bin/env python3
"""
Upload bbox-only dataset to Roboflow using workspace.upload_dataset()
"""
import os
from roboflow import Roboflow
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

# Configuration
WORKSPACE_NAME = "tnsqai-maigj"
PROJECT_NAME = "ct-brain-bbox-polygons"
# Point to the PARENT folder containing train/ subfolder
DATASET_PATH = "/Users/yugambahl/Desktop/brain_ct/data/training_datasets/v5/roboflow_bbox_only"

print("=" * 80)
print("                    ROBOFLOW UPLOAD")
print("=" * 80)
print()
print(f"Workspace: {WORKSPACE_NAME}")
print(f"Project: {PROJECT_NAME}")
print(f"Dataset: {DATASET_PATH}")
print()

# Check if API key is set
api_key = os.getenv("ROBOFLOW_API_KEY")
if not api_key:
    print("❌ ROBOFLOW_API_KEY not found in environment variables")
    print()
    print("Please set it by either:")
    print("  1. Creating a .env file with: ROBOFLOW_API_KEY=your_key_here")
    print("  2. Or export it: export ROBOFLOW_API_KEY=your_key_here")
    print()
    print("Get your API key from: https://app.roboflow.com/settings/api")
    exit(1)

# Check directory structure
train_dir = Path(DATASET_PATH) / "train"
annotation_file = train_dir / "_annotations.coco.json"

if not train_dir.exists():
    print(f"❌ Train directory not found: {train_dir}")
    exit(1)

if not annotation_file.exists():
    print(f"❌ Annotation file not found: {annotation_file}")
    exit(1)

# Count images
image_count = len(list(train_dir.glob("*.jpg"))) + len(list(train_dir.glob("*.png")))
print(f"✓ Found {image_count:,} images in train/")
print(f"✓ Found _annotations.coco.json")
print()

print("=" * 80)
print("Connecting to Roboflow...")
print("=" * 80)
print()

# Initialize Roboflow
rf = Roboflow(api_key=api_key)

# Get workspace
print(f"Loading workspace: {WORKSPACE_NAME}")
workspace = rf.workspace(WORKSPACE_NAME)
print("✓ Workspace loaded")
print()

print("=" * 80)
print("Starting upload...")
print("=" * 80)
print()
print(f"⏳ Uploading {image_count:,} images with COCO annotations...")
print("   This may take a while...")
print()

try:
    # Use workspace.upload_dataset() for COCO format
    workspace.upload_dataset(
        dataset_path=DATASET_PATH,
        project_name=PROJECT_NAME,
        num_workers=10,
        project_license="MIT",
        project_type="object-detection",
        batch_name=None,
        num_retries=3,
        dataset_format="coco",
    )

    print()
    print("=" * 80)
    print("✓ UPLOAD COMPLETE!")
    print("=" * 80)
    print()
    print(f"View your project at:")
    print(f"  https://app.roboflow.com/{WORKSPACE_NAME}/{PROJECT_NAME}")
    print()

except Exception as e:
    print()
    print("=" * 80)
    print("❌ UPLOAD FAILED")
    print("=" * 80)
    print()
    print(f"Error: {e}")
    print()
    import traceback
    traceback.print_exc()
    exit(1)
