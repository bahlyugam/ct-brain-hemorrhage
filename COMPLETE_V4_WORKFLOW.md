# Complete V4 Dataset Preparation Workflow

This guide provides the complete step-by-step process to download missing no-hemorrhage images, merge them with existing datasets, and prepare the balanced v4 dataset for training.

## Prerequisites

### 1. Environment Setup
```bash
# Ensure Python dependencies are installed
pip install pandas pydicom boto3 Pillow tqdm python-dotenv

# Verify you're in the project root
cd /Users/yugambahl/Desktop/brain_ct
```

### 2. AWS Credentials
Create a `.env` file in the project root:
```bash
# .env
AWS_ACCESS_KEY=your_aws_access_key_here
AWS_SECRET_KEY=your_aws_secret_key_here
```

### 3. Verify Data Sources
```bash
# Check V3 dataset exists
ls data/training_datasets/v3/filtered_4class/coco/train/_annotations.coco.json

# Check Roboflow dataset exists
ls data/training_datasets/roboflow_downloads/UAT_CT_BRAIN_HEMORRHAGE_V3.v3i.coco/train/_annotations.coco.json

# Check CSV exists
ls data/metadata/no_hemorrhage_positive_feedback.csv
```

---

## Part 1: Download Missing No-Hemorrhage Images

### Step 1.1: Calculate Required Images

Current v4 dataset balance:
- **Total**: 10,781 images
- **No-hemorrhage**: 3,453 (32.0%)
- **With-hemorrhage**: 7,328 (68.0%)

To achieve 50% balance:
- **Target no-hem**: 7,328 (to match with-hem count)
- **Current no-hem**: 3,453
- **Gap**: **3,875 images needed**

### Step 1.2: Download Missing Images

**Recommended: Download exactly 3,875 images**

```bash
python scripts/download_missing_no_hemorrhage_images.py --max-images 3875
```

**What this does:**
1. Scans existing v4 dataset to identify which CSV images are already included
2. Separates CSV into:
   - **165 False Positives (FP)** - missing from dataset (HIGH PRIORITY)
   - **322,690 True Negatives (TN)** - missing from dataset (LOW PRIORITY)
3. Downloads ALL 165 FP images first
4. Downloads 3,710 TN images to reach 3,875 total
5. Converts DICOM files to PNG with proper windowing
6. Filters by slice thickness (5.0mm or 5.5mm)

**Expected Output:**
```
[6/6] Download Summary
  ✅ Successful: 3500-3800
  ⏭️  Skipped (slice thickness): 50-200
  ❌ Failed: 50-150
  📊 Total attempted: 3875
```

**Time estimate:** 2-4 hours (depends on network speed and S3 performance)

**Storage requirement:** ~400-800 MB

**Alternative Options:**

```bash
# Download ONLY False Positives (165 images)
python scripts/download_missing_no_hemorrhage_images.py --fp-only

# Download 2000 images (partial improvement)
python scripts/download_missing_no_hemorrhage_images.py --max-images 2000

# Download ALL available missing images (not recommended - 323,000+ images)
python scripts/download_missing_no_hemorrhage_images.py
```

### Step 1.3: Verify Downloads

```bash
# Check downloaded images
ls -lh data/training_datasets/no_hemorrhage_downloads/ | head -20

# Count downloaded PNGs
ls data/training_datasets/no_hemorrhage_downloads/*.png | wc -l
```

Expected: ~3,500-3,800 PNG files

---

## Part 2: Create COCO Annotations

### Step 2.1: Generate Annotations

```bash
python scripts/create_no_hemorrhage_coco_annotations.py
```

**What this does:**
1. Scans all PNG images in `no_hemorrhage_downloads/`
2. Creates COCO format annotations
3. Sets empty annotations list (no hemorrhages in these images)
4. Saves `_annotations.coco.json`

**Expected Output:**
```
✅ COCO annotations saved to: .../no_hemorrhage_downloads/_annotations.coco.json
   Images: 3542
   Annotations: 0 (empty - no hemorrhages)
   Categories: 6
```

### Step 2.2: Verify Annotations

```bash
# Check annotation file exists
ls -lh data/training_datasets/no_hemorrhage_downloads/_annotations.coco.json

# View first few lines
head -30 data/training_datasets/no_hemorrhage_downloads/_annotations.coco.json
```

---

## Part 3: Prepare V4 Dataset with Balanced Splits

### Step 3.1: Run Dataset Preparation

```bash
python scripts/prepare_v4_dataset.py
```

**What this does:**
1. **[1/9] Load V3 dataset** - 6,999 images from filtered 4-class
2. **[2/9] Remap V3 classes** - 4-class → 6-class positions
3. **[3/9] Load Roboflow** - 3,782 images (8 categories)
4. **[4/9] Remap Roboflow** - 8 categories → 6 classes, filter noise
5. **[4.5/9] Load no-hem downloads** - ~3,500 images (kept separate)
6. **[5/9] Merge V3 + Roboflow** - 10,781 images (no_hem pool separate)
7. **[5.5/9] Filter missing images**
8. **[5.6/9] Fix out-of-bounds bboxes** - Pre-augmentation clipping
9. **[6/9] Patient-level split** - 80/15/5 train/val/test
   - Ensures NO patient leakage between splits
   - Verifies with assertions
10. **[6.1/9] Load CSV** - Categorize FP vs TN
11. **[6.2/9] Match CSV to pool** - Identify FP/TN in downloads
12. **[6.5/9] Balance splits** - **KEY STEP: PROPORTIONAL DISTRIBUTION**
    - Calculate shortfall per split
    - Distribute pool proportionally
    - FP images get priority
    - Achieve ~50% in each split
13. **[7/9] Save pre-augmentation** - Temporary combined dataset
14. **[8/9] Apply 3x augmentation** - Training set only
15. **[9/9] Create 4-class filtered** - EDH/HC removed, symlinks

**Expected Output:**
```
[6.5/9] Balancing no-hemorrhage images in splits...

  Current balance per split:
    Train: 2747/8620 no-hem (31.9%) | Need 3126 more
    Valid: 537/1617 no-hem (33.2%) | Need 543 more
    Test: 169/544 no-hem (31.1%) | Need 206 more
  Total shortfall: 3875 no-hem images

  Available in no-hem pool: 3542 images
    FP (high priority): 165
    TN (low priority): 3377

  Distributing 3542 images:
    Train: 2860 images
    Valid: 497 images
    Test: 185 images

  Final balance after distribution:
    Train: 5607/11480 no-hem (48.8%)
    Valid: 1034/2114 no-hem (48.9%)
    Test: 354/729 no-hem (48.6%)

  Overall: 6995/14323 no-hem (48.8%)
```

**Time estimate:** 30-60 minutes (augmentation is slow)

**Storage requirement:** ~25-30 GB total
- 6-class dataset: ~15 GB
- 4-class dataset: ~10 GB (symlinks)

### Step 3.2: Verify No Patient Leakage

The pipeline automatically verifies with assertions (lines 513-515):
```python
assert len(set(train_patients) & set(val_patients)) == 0
assert len(set(train_patients) & set(test_patients)) == 0
assert len(set(val_patients) & set(test_patients)) == 0
```

If there's any leakage, the script will **crash with AssertionError**.

**Additional verification:**
```bash
# The validation script also checks for patient leakage
python scripts/validate_v4_dataset.py
```

---

## Part 4: Analyze Dataset

### Step 4.1: Comprehensive Analysis

```bash
python scripts/analyze_v4_dataset.py
```

**Expected Output:**
```
================================================================================
V4 DATASET - CLASS DISTRIBUTION (Including No-Hemorrhage Images)
================================================================================

TRAIN SPLIT:
  Total images: 34,440 (after 3x augmentation)
  Original (before augmentation): 11,480
    Augmented (_aug suffix): 22,960
  Total annotations: 35,340

  Class Distribution (Annotations):
    EDH: 800 annotations
    HC: 700 annotations
    IPH: 4,000 annotations
    IVH: 1,600 annotations
    SAH: 3,800 annotations
    SDH: 3,700 annotations

  Class Distribution (Images with each class):
    EDH: 650 images
    HC: 600 images
    IPH: 3,200 images
    IVH: 1,300 images
    SAH: 3,100 images
    SDH: 3,000 images

  Hemorrhage Status:
    Images with NO hemorrhages: 16,821 (48.8% of total train)
    Images WITH hemorrhages: 17,619 (51.2% of total train)

VALID SPLIT:
  Total images: 2,114
  Class Distribution...
  Hemorrhage Status:
    Images with NO hemorrhages: 1,034 (48.9%)
    Images WITH hemorrhages: 1,080 (51.1%)

TEST SPLIT:
  Total images: 729
  Hemorrhage Status:
    Images with NO hemorrhages: 354 (48.6%)
    Images WITH hemorrhages: 375 (51.4%)

================================================================================
ORIGINAL IMAGES SUMMARY (Before Augmentation)
================================================================================
Total original images: 14,323
  With hemorrhages: 7,328 (51.2%)
  Without hemorrhages: 6,995 (48.8%)

Original images by class:
  IPH: 3,200 images (22.3%)
  SAH: 3,100 images (21.6%)
  SDH: 3,000 images (20.9%)
  IVH: 1,300 images (9.1%)
  EDH: 650 images (4.5%)
  HC: 600 images (4.2%)

================================================================================
PATIENT STATISTICS
================================================================================
Train: 2,218 unique patients
Valid: 495 unique patients
Test: 151 unique patients

Total unique patients: 2,864

✓ No Train/Valid patient overlap
✓ No Train/Test patient overlap
✓ No Valid/Test patient overlap
```

### Step 4.2: Export Dataset CSV

```bash
# Export 6-class dataset metadata
python scripts/export_v4_dataset_csv.py

# Export 4-class dataset metadata
python scripts/export_v4_dataset_csv.py \
  --dataset-dir data/training_datasets/v4/4class_coco \
  --output data/metadata/v4_dataset_4class.csv
```

**Output files:**
- `data/metadata/v4_dataset_6class.csv`
- `data/metadata/v4_dataset_4class.csv`

---

## Part 5: Validate Dataset

### Step 5.1: Run Validation

```bash
python scripts/validate_v4_dataset.py
```

**What this checks:**
1. ✅ COCO format validity (using pycocotools)
2. ✅ All 6 classes present in train/val/test
3. ✅ Patient-level leakage detection (ZERO overlap required)
4. ✅ Image file integrity (all images readable)
5. ✅ Bounding box validity (100% within image bounds)
6. ✅ Augmentation verification (3.0x multiplier)
7. ✅ Class distribution analysis
8. ✅ Symlink integrity for 4-class dataset

**Expected Output:**
```
================================================================================
V4 DATASET VALIDATION REPORT
================================================================================

[1/8] Validating COCO format...
  ✅ Train annotations valid
  ✅ Valid annotations valid
  ✅ Test annotations valid

[2/8] Checking class distribution...
  ✅ All 6 classes present in train
  ✅ All 6 classes present in valid
  ✅ All 6 classes present in test

[3/8] Checking patient-level leakage...
  ✅ No Train/Valid overlap (0 patients)
  ✅ No Train/Test overlap (0 patients)
  ✅ No Valid/Test overlap (0 patients)

[4/8] Validating image files...
  ✅ Train: 34,440/34,440 images readable
  ✅ Valid: 2,114/2,114 images readable
  ✅ Test: 729/729 images readable

[5/8] Validating bounding boxes...
  ✅ Train: 35,340/35,340 boxes valid (100.0%)
  ✅ Valid: 2,330/2,330 boxes valid (100.0%)
  ✅ Test: 723/723 boxes valid (100.0%)

[6/8] Verifying augmentation...
  ✅ Augmentation multiplier: 3.0x
  ✅ Original images: 11,480
  ✅ Total after augmentation: 34,440

[7/8] Analyzing class balance...
  No-hemorrhage images:
    Train: 16,821/34,440 (48.8%)
    Valid: 1,034/2,114 (48.9%)
    Test: 354/729 (48.6%)
  ✅ All splits near 50% target

[8/8] Checking 4-class dataset...
  ✅ Symlinks valid
  ✅ EDH/HC filtered correctly

================================================================================
✅ VALIDATION PASSED - Dataset ready for training!
================================================================================
```

---

## Part 6: Training

### Step 6.1: Train 6-Class Model (Full Dataset)

```bash
modal run train.py
```

**Configuration:**
- Classes: EDH, HC, IPH, IVH, SAH, SDH
- Training images: 34,440 (11,480 original × 3)
- Validation images: 2,114
- Test images: 729
- Model: RT-DETR Medium
- Epochs: 200
- Batch size: 16

### Step 6.2: Train 4-Class Model (Filtered Dataset)

```bash
modal run train.py --use_filtered_dataset
```

**Configuration:**
- Classes: IPH, IVH, SAH, SDH (EDH/HC removed)
- Same image counts, filtered annotations
- Model: RT-DETR Medium
- Epochs: 200
- Batch size: 16

---

## Automated Workflow (Alternative)

If you want to run all steps automatically:

```bash
# Complete workflow: download → annotate → prepare
bash scripts/prepare_v4_with_no_hem_downloads.sh --max-images 3875

# Then validate
python scripts/validate_v4_dataset.py

# Then analyze
python scripts/analyze_v4_dataset.py

# Then train
modal run train.py
```

---

## Summary Checklist

### Pre-Download
- [ ] AWS credentials configured in `.env`
- [ ] V3 dataset exists
- [ ] Roboflow dataset exists
- [ ] CSV file exists
- [ ] Python dependencies installed

### Download Phase
- [ ] Downloaded ~3,500-3,800 no-hemorrhage images
- [ ] Created COCO annotations
- [ ] Verified annotation file exists

### Preparation Phase
- [ ] Ran `prepare_v4_dataset.py` successfully
- [ ] No errors during patient-level splitting
- [ ] Balancing achieved ~48-50% in each split
- [ ] Augmentation completed (3x multiplier)
- [ ] Both 6-class and 4-class datasets created

### Validation Phase
- [ ] All COCO formats valid
- [ ] ZERO patient leakage between splits
- [ ] 100% valid bounding boxes
- [ ] All images readable
- [ ] Balance verified ~48-50% per split

### Training Phase
- [ ] Validation passed
- [ ] Ready to train

---

## Expected Final Dataset Statistics

**6-Class Dataset:**
```
Total Original Images: ~14,300
  Train: ~11,480 (80%) → 34,440 after 3x augmentation
  Valid: ~2,114 (15%)
  Test: ~729 (5%)

Balance per split:
  Train: 48.8% no-hem, 51.2% with-hem
  Valid: 48.9% no-hem, 51.1% with-hem
  Test: 48.6% no-hem, 51.4% with-hem

Patient distribution:
  Train: ~2,218 patients (NO overlap with val/test)
  Valid: ~495 patients (NO overlap with train/test)
  Test: ~151 patients (NO overlap with train/val)

Classes: EDH, HC, IPH, IVH, SAH, SDH
Storage: ~15 GB
```

**4-Class Dataset:**
```
Same structure as 6-class
Classes: IPH, IVH, SAH, SDH (EDH/HC filtered)
Storage: ~10 GB (symlinks to 6-class images)
```

---

## Troubleshooting

### Issue: Download fails with S3 access error
**Solution:** Check AWS credentials in `.env` file

### Issue: Patient leakage assertion error
**Solution:** This should never happen - contact maintainer if it does

### Issue: Balance not exactly 50%
**Explanation:** Expected - pool size may be slightly less than shortfall due to:
- Slice thickness filtering
- Download failures
- File corruption
Achieving 48-50% is acceptable.

### Issue: Augmentation takes too long
**Solution:** Normal - 30-60 minutes expected for ~11,000 images

### Issue: Out of disk space
**Solution:** Need ~30 GB free space. Clean up temp files:
```bash
rm -rf data/training_datasets/v4/temp_combined
rm -rf data/training_datasets/no_hemorrhage_downloads/temp_dicom
```

---

## Files Created During Workflow

```
data/
├── training_datasets/
│   ├── no_hemorrhage_downloads/
│   │   ├── _annotations.coco.json
│   │   ├── 520892552_14.png
│   │   ├── 525190930_29.png
│   │   └── ... (~3,500 PNG files)
│   └── v4/
│       ├── 6class_coco/
│       │   ├── train/
│       │   │   ├── _annotations.coco.json
│       │   │   ├── v3_*.jpg
│       │   │   ├── roboflow_*.jpg
│       │   │   ├── no_hem_*.png
│       │   │   └── *_aug*.jpg
│       │   ├── valid/
│       │   ├── test/
│       │   └── metadata.yaml
│       └── 4class_coco/
│           ├── train/
│           ├── valid/
│           ├── test/
│           └── metadata.yaml
└── metadata/
    ├── v4_dataset_6class.csv
    └── v4_dataset_4class.csv
```

---

## Next Steps After Training

1. Monitor training metrics on Modal dashboard
2. Evaluate model on test set
3. Compare 6-class vs 4-class performance
4. Analyze FP/TN performance specifically
5. Generate confusion matrices
6. Deploy best model

---

**Good luck with your training! 🚀**
