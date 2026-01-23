#!/usr/bin/env python3
"""
Upload v4 original images dataset to Roboflow.
"""
import os
from roboflow import Roboflow
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

# Configuration
WORKSPACE_NAME = "tnsqai-maigj"  # Just the workspace name, not the URL
PROJECT_NAME = "ctbrain_reverification-smartpolygon_changes"
DATASET_PATH = "/Users/yugambahl/Desktop/brain_ct/data/training_datasets/v4/original_images_only"
ANNOTATION_FILE = f"{DATASET_PATH}/_annotations.coco.json"

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
    print("Get your API key from: https://app.roboflow.com/auth-cli")
    exit(1)

# Check if annotation file exists
if not Path(ANNOTATION_FILE).exists():
    print(f"❌ Annotation file not found: {ANNOTATION_FILE}")
    exit(1)

# Count images
image_count = len(list(Path(DATASET_PATH).glob("*.png"))) + len(list(Path(DATASET_PATH).glob("*.jpg")))
print(f"✓ Found {image_count:,} images")
print(f"✓ Found annotation file")
print()

print("=" * 80)
print("Connecting to Roboflow...")
print("=" * 80)
print()

# Initialize Roboflow
rf = Roboflow(api_key=api_key)

# Get workspace (just the name, not URL)
print(f"Loading workspace: {WORKSPACE_NAME}")
workspace = rf.workspace(WORKSPACE_NAME)
print("✓ Workspace loaded")

# Get or create project
print(f"Loading project: {PROJECT_NAME}")
project = workspace.project(PROJECT_NAME)
print("✓ Project loaded")
print()

print("=" * 80)
print("Starting upload...")
print("=" * 80)
print()
print("⏳ Uploading 14,635 images with parallel processing...")
print("   This will be much faster than sequential upload!")
print()

# Upload dataset with COCO format using batch upload
# The num_workers parameter enables parallel uploads
try:
    project.upload(
        model_format="coco",
        image_path=DATASET_PATH,
        annotation_path=ANNOTATION_FILE,
        num_workers=10,  # Upload 10 images in parallel
        num_retries=3    # Retry failed uploads 3 times
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
    print("If the upload is too slow, try using the roboflow CLI instead:")
    print()
    print("  roboflow import \\")
    print(f"    -w {WORKSPACE_NAME} \\")
    print(f"    -p {PROJECT_NAME} \\")
    print("    --formats coco \\")
    print(f"    {DATASET_PATH}")
    print()
    exit(1)