# Brain CT Hemorrhage Detection - Project Overview

## 📁 Project Structure

```
brain_ct/
├── train.py                          # Main training script (Modal)
├── analyze_and_balance_dataset.py    # Dataset balancing tool
├── analyze_class_distribution.py     # Class analysis & model recommendation
├── ct_augmentations.py               # Evidence-based CT augmentations
├── yolo_augmented_dataset.py         # YOLOv8 augmentation wrapper
├── custom_loss.py                    # Focal loss & class weights
│
├── data/
│   ├── documentation/                # 📚 ALL DOCUMENTATION HERE
│   │   ├── README.md                        # Documentation index
│   │   ├── HYPERPARAMETER_EXPLANATION.md    # 30+ page detailed guide
│   │   ├── HYPERPARAMETERS_QUICK_REFERENCE.txt  # Quick lookup
│   │   ├── AUGMENTATION_SUMMARY.md          # Evidence-based augmentations
│   │   ├── MODEL_SELECTION_AND_CLASS_WEIGHTS.md # Analysis & rationale
│   │   ├── training_configuration_summary.txt   # One-page overview
│   │   └── dataset_analysis.json            # Statistics & weights
│   │
│   ├── UAT_CT_BRAIN_HEMORRHAGE.v2i.yolov8_combined/  # Training dataset
│   │   ├── train/
│   │   │   ├── images/    # 5,989 images (thin + thick slices)
│   │   │   └── labels/    # YOLO format annotations
│   │   ├── valid/
│   │   │   ├── images/    # 1,220 images
│   │   │   └── labels/
│   │   ├── test/
│   │   │   ├── images/    # 335 images
│   │   │   └── labels/
│   │   └── data.yaml      # Dataset configuration
│   │
│   └── no_hemorrhage_positive_feedback/  # Source for balancing
│       └── png/           # 2,700 no-hemorrhage images
│
└── [patient folders...]   # Raw DICOM data
```

## 🎯 Quick Start

### View Documentation
```bash
# Start here - one-page overview
cat data/documentation/training_configuration_summary.txt

# Quick reference during training
cat data/documentation/HYPERPARAMETERS_QUICK_REFERENCE.txt

# Deep dive - understand every parameter
open data/documentation/HYPERPARAMETER_EXPLANATION.md

# Complete documentation index
open data/documentation/README.md
```

### Training Commands
```bash
# Start new training
modal run train.py::main

# Resume from checkpoint
modal run train.py::resume_training

# Validation only
modal run train.py::run_validation
```

### Dataset Analysis
```bash
# Analyze class distribution & get model recommendation
python3 analyze_class_distribution.py

# Balance dataset (already done - adds no-hemorrhage images)
python3 analyze_and_balance_dataset.py --auto-confirm
```

## 📊 Current Configuration

### Dataset
- **Total**: 7,544 images (balanced 1:1 hemorrhage:no-hemorrhage)
- **Classes**: 6 hemorrhage types (EDH, HC, IPH, IVH, SAH, SDH)
- **Critical imbalance**: EDH 22x rarer than SAH (125 vs 2,757 instances)

### Model
- **Architecture**: YOLOv8m (25.9M parameters)
- **Why**: Optimal for 7.5K images, sufficient capacity for class imbalance
- **Upgrade**: From YOLOv8s (2.3x more parameters)

### Training
- **Batch size**: 16
- **Image size**: 640×640
- **Epochs**: 200 (patience=100)
- **Learning rate**: 0.001 (fine-tuning from pretrained)

### Loss Weights
- **Box**: 7.5 (localization critical)
- **Cls**: 0.5 (with 3.98x internal weight for EDH)
- **DFL**: 1.5 (box quality)

### Augmentations
✅ Evidence-based (RSNA 2019 winners):
- Horizontal flip (50%)
- Rotation (±10°/±16°)
- Translation (6.25%/10%)
- Scale, Mosaic, Multi-scale

❌ Disabled (harmful for medical):
- Vertical flip, Shear, Perspective, Mixup, Copy-paste

## 🎯 Expected Performance

| Metric | Target |
|--------|--------|
| Overall mAP50 | 0.67-0.70 |
| Sensitivity | 0.85-0.90 |
| Specificity | 0.90-0.95 |
| **EDH Recall** | **0.70-0.80** ⭐ (critical improvement) |

**Impact**: 30-80% reduction in missed epidural hemorrhages vs previous model

## 📚 Documentation Location

**⚠️ IMPORTANT: All documentation is now in `data/documentation/`**

This keeps the project root clean while organizing all guides, references, and analysis reports in one place.

### Documentation Files
1. **README.md** - Start here, documentation index
2. **HYPERPARAMETER_EXPLANATION.md** - Comprehensive guide (30+ pages)
3. **HYPERPARAMETERS_QUICK_REFERENCE.txt** - Quick lookup (1 page)
4. **AUGMENTATION_SUMMARY.md** - Evidence-based augmentations
5. **MODEL_SELECTION_AND_CLASS_WEIGHTS.md** - Analysis & rationale
6. **training_configuration_summary.txt** - Complete overview
7. **dataset_analysis.json** - Statistics & weights

## 🔍 Finding Information

**Question** → **Where to Look**

- Why this hyperparameter value? → `HYPERPARAMETER_EXPLANATION.md`
- Quick parameter lookup? → `HYPERPARAMETERS_QUICK_REFERENCE.txt`
- What augmentations? → `AUGMENTATION_SUMMARY.md`
- Why YOLOv8m? → `MODEL_SELECTION_AND_CLASS_WEIGHTS.md`
- Overall configuration? → `training_configuration_summary.txt`
- Class statistics? → `dataset_analysis.json`
- Everything! → `README.md` (index)

## 📈 Monitoring During Training

### Key Metrics (WandB)
1. **EDH recall (class 0)** - Most critical ⚠️
2. Classification loss - Should stabilize < 1.5
3. Box loss - Should decrease < 1.0
4. Per-class mAP - All should be > 0.55

### After 50 Epochs
- ✅ EDH recall > 0.60
- ⚠️ EDH recall < 0.60 → Increase cls weight to 1.0

### After 100 Epochs
- ✅ EDH recall > 0.70 → Clinical target met
- ✅ Overall mAP50 > 0.65 → Ready for deployment

## 🛠️ Utilities

### Analysis Scripts
- `analyze_class_distribution.py` - Dataset analysis, model recommendation
- `analyze_and_balance_dataset.py` - Balance hemorrhage:no-hemorrhage ratio

### Training Scripts
- `train.py` - Main Modal training script
- `ct_augmentations.py` - Albumentations pipeline
- `yolo_augmented_dataset.py` - YOLOv8 integration
- `custom_loss.py` - Focal loss implementation

## 🚀 Next Steps

1. **Review documentation** (start with `data/documentation/README.md`)
2. **Verify dataset** (already balanced to 1:1 ratio)
3. **Start training** (`modal run train.py::main`)
4. **Monitor EDH recall** (most critical metric)
5. **Adjust if needed** (see adjustment guidelines in docs)

## 📞 Quick Help

- **Documentation**: `data/documentation/README.md`
- **Quick reference**: `data/documentation/HYPERPARAMETERS_QUICK_REFERENCE.txt`
- **Training issues**: Check monitoring guidelines in any doc
- **Understanding config**: `data/documentation/HYPERPARAMETER_EXPLANATION.md`

---

**Version**: v6_evidence_based_aug
**Model**: YOLOv8m (25.9M parameters)
**Status**: ✅ Ready for Training

**All documentation is in `data/documentation/` - START THERE!** 📚