# V4 Dataset - Quick Start Guide

## TL;DR - Complete Workflow in 5 Commands

```bash
# 1. Download missing no-hemorrhage images (2-4 hours)
python scripts/download_missing_no_hemorrhage_images.py --max-images 3875

# 2. Create COCO annotations (< 1 minute)
python scripts/create_no_hemorrhage_coco_annotations.py

# 3. Prepare v4 dataset with balanced splits (30-60 minutes)
python scripts/prepare_v4_dataset.py

# 4. Validate dataset (5 minutes)
python scripts/validate_v4_dataset.py

# 5. Train model
modal run train.py
```

## Or Use Automated Script

```bash
# All-in-one command (downloads + prepares + validates)
bash scripts/prepare_v4_with_no_hem_downloads.sh --max-images 3875
python scripts/validate_v4_dataset.py
modal run train.py
```

## Key Points

### Data Leakage Protection ✅
- **Patient-level splitting** ensures ZERO patient overlap between train/valid/test
- **Assertions verify** no leakage (crashes if detected)
- **No-hem images** distributed proportionally while maintaining patient isolation

### Balance Achievement 🎯
- **Target:** 50% no-hemorrhage in each split
- **Achieved:** ~48-50% in train/valid/test independently
- **Strategy:** FP images prioritized, then TN images fill remaining quota

### Expected Results 📊
```
Final Dataset:
  Total: ~14,300 original images (~43,000 after 3x augmentation)

  Train: ~11,480 original (48.8% no-hem) → 34,440 after augmentation
  Valid: ~2,114 original (48.9% no-hem)
  Test: ~729 original (48.6% no-hem)

  Patients: 2,864 unique (NO overlap between splits)
  Classes: EDH, HC, IPH, IVH, SAH, SDH
```

### What Gets Downloaded 📥
- **165 False Positives** (model incorrectly predicted hemorrhage) - HIGH PRIORITY
- **3,710 True Negatives** (model correctly identified no hemorrhage) - FILL QUOTA
- **Total: 3,875 images** to achieve 50% balance

### Storage Requirements 💾
- Downloads: ~400-800 MB
- Final v4 6-class: ~15 GB
- Final v4 4-class: ~10 GB (symlinks)
- **Total needed:** ~30 GB free space

### Time Requirements ⏱️
1. Download: 2-4 hours
2. Annotate: < 1 minute
3. Prepare: 30-60 minutes
4. Validate: 5 minutes
**Total: ~3-5 hours**

## Prerequisites

1. **AWS Credentials** - Create `.env` file:
   ```
   AWS_ACCESS_KEY=your_key
   AWS_SECRET_KEY=your_secret
   ```

2. **Python Dependencies:**
   ```bash
   pip install pandas pydicom boto3 Pillow tqdm python-dotenv
   ```

3. **Existing Datasets:**
   - V3 filtered 4-class COCO dataset
   - Roboflow V3 COCO dataset
   - no_hemorrhage_positive_feedback.csv

## Verification

```bash
# Check balance
python scripts/analyze_v4_dataset.py

# Verify no patient leakage (should pass all checks)
python scripts/validate_v4_dataset.py

# Export metadata
python scripts/export_v4_dataset_csv.py
```

## Training

```bash
# Train 6-class model (all hemorrhage types)
modal run train.py

# Train 4-class model (IPH, IVH, SAH, SDH only)
modal run train.py --use_filtered_dataset
```

## Troubleshooting

**Download fails?** → Check AWS credentials in `.env`

**Out of space?** → Need 30 GB free, clean temp files:
```bash
rm -rf data/training_datasets/v4/temp_combined
```

**Validation fails?** → Check error message:
- Patient leakage: Should never happen (pipeline bug)
- Invalid bboxes: Re-run preparation (should be 100% valid)
- Missing files: Check downloads completed

**Balance not exactly 50%?** → Normal:
- Download failures, slice thickness filtering reduce pool slightly
- 48-50% is acceptable

## Full Documentation

See [COMPLETE_V4_WORKFLOW.md](COMPLETE_V4_WORKFLOW.md) for detailed step-by-step guide.

## Support

For issues or questions:
1. Check [NO_HEMORRHAGE_BALANCING_README.md](scripts/NO_HEMORRHAGE_BALANCING_README.md)
2. Review validation output for specific errors
3. Check logs in preparation output
