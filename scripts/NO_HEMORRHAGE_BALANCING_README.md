# No-Hemorrhage Image Balancing Workflow

This document describes the workflow for achieving 50% no-hemorrhage balance in the v4 dataset by downloading missing images from the CSV feedback data.

## Overview

The v4 dataset currently has only 32% no-hemorrhage images (3,453 out of 10,781). To reach 50% balance, we need ~1,937 additional no-hemorrhage images.

The CSV file `no_hemorrhage_positive_feedback.csv` contains 325,454 images:
- **213 False Positives (FP)**: Model incorrectly predicted hemorrhage (HIGH PRIORITY)
- **325,241 True Negatives (TN)**: Model correctly identified no hemorrhage (LOW PRIORITY)

However, only 2,599 (0.8%) of these CSV images currently exist in the v4 dataset. The rest need to be downloaded from S3.

## Key Feature: Split-Level Balancing

The updated pipeline now distributes downloaded no-hemorrhage images **proportionally across train/valid/test splits** to achieve 50% balance **in each split independently**.

**How it works:**
1. V3 + Roboflow datasets are merged and split by patient (80/15/5)
2. Downloaded no-hem images are kept in a separate pool
3. Each split's shortfall is calculated independently
4. No-hem pool is distributed proportionally to fill each split's gap
5. FP images are prioritized over TN images during distribution

**Example:**
- Train needs 3,000 more no-hem → gets 77% of pool
- Valid needs 600 more no-hem → gets 15% of pool
- Test needs 300 more no-hem → gets 8% of pool

This ensures **each split reaches ~50% no-hemorrhage balance**, not just the overall dataset.

## Workflow Steps

### Step 1: Download Missing No-Hemorrhage Images

```bash
python scripts/download_missing_no_hemorrhage_images.py
```

**Options:**
- `--max-images N`: Limit download to N images (default: no limit)
- `--fp-only`: Download only False Positive images (skip True Negatives)
- `--csv-path PATH`: Path to CSV file (default: data/metadata/no_hemorrhage_positive_feedback.csv)
- `--output-dir PATH`: Directory to save PNG images (default: data/training_datasets/no_hemorrhage_downloads)
- `--use-credentials`: Use AWS credentials for S3 access (requires .env file)
- `--no-filter-thickness`: Disable slice thickness filtering

**What it does:**
1. Scans existing v4 dataset to identify which images are already included
2. Loads CSV and separates into FP (high priority) and TN (low priority)
3. Filters out images already in dataset
4. Downloads ALL missing FP images first
5. Downloads TN images to fill remaining quota (up to --max-images)
6. Converts DICOM files to PNG with proper windowing

**Example:**
```bash
# Download up to 2000 images, prioritizing FP
python scripts/download_missing_no_hemorrhage_images.py --max-images 2000

# Download ONLY False Positives
python scripts/download_missing_no_hemorrhage_images.py --fp-only
```

### Step 2: Create COCO Annotations

```bash
python scripts/create_no_hemorrhage_coco_annotations.py
```

**What it does:**
1. Scans the no_hemorrhage_downloads directory for PNG images
2. Creates COCO format annotations with empty annotations list (no hemorrhages)
3. Saves `_annotations.coco.json` in the downloads directory

**Output:**
- `/Users/yugambahl/Desktop/brain_ct/data/training_datasets/no_hemorrhage_downloads/_annotations.coco.json`

### Step 3: Run V4 Dataset Preparation

```bash
python scripts/prepare_v4_dataset.py
```

**What it does:**
1. Loads V3 dataset (4-class filtered)
2. Loads Roboflow dataset (8 categories → 6 classes)
3. **NEW**: Loads downloaded no-hemorrhage images if available
4. Merges all datasets with prefixes (v3_, roboflow_, no_hem_)
5. Performs patient-level splitting (80/15/5 train/val/test)
6. **NEW**: Analyzes and reports no-hemorrhage balance
7. Applies 3x augmentation to training set
8. Creates filtered 4-class version

## Automated Workflow (Recommended)

Use the integration script to run all steps automatically:

```bash
# Download and prepare with all steps
bash scripts/prepare_v4_with_no_hem_downloads.sh

# Download only FP images and prepare
bash scripts/prepare_v4_with_no_hem_downloads.sh --fp-only

# Download up to 2000 images and prepare
bash scripts/prepare_v4_with_no_hem_downloads.sh --max-images 2000

# Skip download, just create annotations and prepare
bash scripts/prepare_v4_with_no_hem_downloads.sh --skip-download
```

## Expected Results

### Before Downloading Additional Images

- Total original images: 10,781
- No-hemorrhage: 3,453 (32.0%)
- CSV images already in dataset: 2,599 (48 FP + 2,551 TN)

### After Downloading (Example: 2000 additional images)

**Assuming successful download of 2000 images:**
- Total original images: ~12,781
- No-hemorrhage: ~5,453 (42.7%)
- Improvement: +2,000 no-hem images (+10.7% towards 50% target)

**To reach 50% exactly:**
- Need: ~7,328 no-hemorrhage images (to match 7,328 with-hemorrhage)
- Current: 3,453 no-hem
- Gap: 3,875 more needed
- CSV pool: 323,000+ images available
- **Recommendation**: Download ~3,875 images to reach 50% balance

```bash
# Download exactly 3875 images to reach 50% balance
python scripts/download_missing_no_hemorrhage_images.py --max-images 3875
```

## CSV Image Prioritization

The download script uses this priority order:

1. **ALL False Positives (213 missing)**: Highest priority
   - These are cases where the model incorrectly predicted hemorrhage
   - Learning from these errors improves model specificity

2. **True Negatives (323,000+ missing)**: Fill remaining quota
   - Model correctly identified no hemorrhage
   - Good quality negative examples

## File Structure

After running the workflow:

```
data/training_datasets/
├── no_hemorrhage_downloads/
│   ├── _annotations.coco.json      # Created in Step 2
│   ├── 520892552_14.png           # Downloaded images
│   ├── 525190930_29.png
│   └── ...
├── v4/
│   ├── 6class_coco/                # Final dataset with all images
│   │   ├── train/
│   │   │   ├── _annotations.coco.json
│   │   │   ├── v3_*.jpg           # From v3 dataset
│   │   │   ├── roboflow_*.jpg     # From Roboflow
│   │   │   ├── no_hem_*.png       # From downloaded images
│   │   │   └── *_aug*.jpg         # Augmented versions
│   │   ├── valid/
│   │   └── test/
│   └── 4class_coco/                # Filtered version
```

## Prerequisites

### Required Files
- `data/metadata/no_hemorrhage_positive_feedback.csv` - CSV with image_url column
- `.env` file with AWS credentials (for S3 access)

### Environment Variables (.env file)
```
AWS_ACCESS_KEY=your_access_key_here
AWS_SECRET_KEY=your_secret_key_here
```

### Python Dependencies
- pandas
- pydicom
- boto3
- Pillow
- tqdm
- python-dotenv

Install:
```bash
pip install pandas pydicom boto3 Pillow tqdm python-dotenv
```

## Troubleshooting

### Issue: "AWS credentials not found"
**Solution**: Create a `.env` file in the project root with AWS credentials:
```
AWS_ACCESS_KEY=your_key
AWS_SECRET_KEY=your_secret
```

### Issue: "Failed to download from S3"
**Solution**:
- Check if S3 bucket is public or requires credentials
- Verify AWS credentials are correct
- Check internet connection

### Issue: "Skipped (slice thickness)"
**Explanation**: Images with slice thickness ≠ 5.0mm or 5.5mm are skipped by default
**Solution**: Use `--no-filter-thickness` to disable filtering (not recommended)

### Issue: "No images found in downloads directory"
**Solution**:
- Verify Step 1 (download) completed successfully
- Check `data/training_datasets/no_hemorrhage_downloads/` for PNG files
- Run download script again if needed

## Validation

After running the workflow, validate the dataset:

```bash
# Validate dataset integrity
python scripts/validate_v4_dataset.py

# Analyze class distribution and no-hem balance
python scripts/analyze_v4_dataset.py
```

Expected output should show improved no-hemorrhage percentage.

## Notes

- **Download time**: Depends on number of images and network speed (~1-2 hours for 2000 images)
- **Storage**: Each PNG image is ~50-200KB. 2000 images ≈ 200-400MB
- **Slice thickness filtering**: Recommended to keep enabled for consistency with v3 dataset
- **Patient-level splitting**: Downloaded images are included before splitting, so they'll be distributed across train/val/test splits
- **CSV matching**: Only images with valid `image_url` in CSV can be downloaded

## Success Criteria

- [ ] Download script completes without errors
- [ ] COCO annotations created successfully
- [ ] V4 preparation includes downloaded images
- [ ] No-hemorrhage percentage improves towards 50%
- [ ] Patient-level splitting has no leakage
- [ ] Validation passes with 100% valid bounding boxes
