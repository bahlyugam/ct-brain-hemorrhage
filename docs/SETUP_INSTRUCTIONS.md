# RF-DETR Training Setup Instructions

## Problem: 125k File Limit Exceeded

Your dataset has **216,808 files** which exceeds Modal's `.add_local_dir()` limit of **125,000 files**.

The solution is to **upload your dataset to a Modal Volume** instead of mounting it.

---

## Step-by-Step Setup

### Step 1: Upload Dataset to Modal Volume (ONE-TIME ONLY)

Run this command from your local machine to upload the original dataset:

```bash
modal volume put brain-ct-datasets \
  /Users/yugambahl/Desktop/brain_ct/data/training_datasets/archived/current_20251105/UAT_CT_BRAIN_HEMORRHAGE.v2i.yolov8_combined \
  /original_6class
```

**This will take 30-60 minutes depending on your internet speed.**

**What this does:**
- Creates a Modal Volume named `brain-ct-datasets`
- Uploads all 15,090 files from your original 6-class dataset
- Stores them at `/original_6class` in the volume

### Step 2: Verify Dataset Upload

Check if the upload was successful:

```bash
modal run train.py::check_data
```

**Expected output:**
```
✓ Dataset found at /data/original_6class
✓ Total files: 15090
```

### Step 3: Prepare Filtered Datasets

Create the filtered 4-class datasets (removes EDH/HC):

```bash
modal run train.py::prepare_datasets
```

**This will:**
- Filter out images containing EDH (class 0) or HC (class 1)
- Remap classes: IPH(2→0), IVH(3→1), SAH(4→2), SDH(5→3)
- Create YOLO format dataset at `/data/filtered_4class/yolo`
- Convert to COCO format at `/data/filtered_4class/coco`
- Takes ~30-60 minutes

### Step 4: Train Models

#### Train YOLOv8m (Filtered 4-Class)
```bash
modal run train.py::train_yolo_filtered
```

#### Train RF-DETR Medium (Filtered 4-Class)
```bash
modal run train.py::train_rfdetr_filtered
```

---

## Command Reference

| Command | Purpose | Duration | Cost |
|---------|---------|----------|------|
| `modal volume put ...` | Upload dataset (ONE-TIME) | 30-60 min | Free (storage) |
| `modal run train.py::check_data` | Verify upload | 10 sec | ~$0.01 |
| `modal run train.py::prepare_datasets` | Create filtered datasets | 30-60 min | ~$0.50 |
| `modal run train.py::train_yolo_filtered` | Train YOLO | 20-30 hrs | ~$20-30 |
| `modal run train.py::train_rfdetr_filtered` | Train RF-DETR | 30-40 hrs | ~$30-40 |

---

## Troubleshooting

### Error: "Unique hash count exceeds API limit"
**Solution:** You're trying to mount too many files. Use `modal volume put` instead (Step 1).

### Error: "Dataset NOT found"
**Solution:** Run Step 1 to upload the dataset first.

### Check Volume Contents
```bash
modal volume ls brain-ct-datasets /original_6class
```

### Delete and Re-upload
```bash
# Delete volume
modal volume delete brain-ct-datasets

# Re-upload
modal volume put brain-ct-datasets /Users/yugambahl/Desktop/brain_ct/data/training_datasets/archived/current_20251105/UAT_CT_BRAIN_HEMORRHAGE.v2i.yolov8_combined /original_6class
```

---

## What Changed

### Before (WRONG - File Limit Exceeded)
```python
# Tried to mount 216k files - FAILED
.add_local_dir(local_data_dir, "/datasets")
```

### After (CORRECT - Using Modal Volume)
```python
# Upload dataset to Modal Volume (one-time)
modal volume put brain-ct-datasets <local_path> /original_6class

# Volume is mounted in functions
@app.function(volumes={"/data": dataset_volume})
def prepare_filtered_datasets():
    # Access at /data/original_6class
    filter_yolo_dataset(
        input_path="/data/original_6class",
        output_path="/data/filtered_4class/yolo"
    )
```

---

## Dataset Structure in Modal Volume

After completing all steps, your Modal Volume will have:

```
/data/
├── original_6class/          # Original 6-class dataset (15,090 files)
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── valid/
│   └── test/
│
└── filtered_4class/          # Filtered 4-class dataset (~14,000 files)
    ├── yolo/                 # YOLO format
    │   ├── data.yaml
    │   ├── train/
    │   ├── valid/
    │   └── test/
    │
    └── coco/                 # COCO format (for RF-DETR)
        ├── train/
        │   ├── images/
        │   └── _annotations.coco.json
        ├── valid/
        └── test/
```

---

## Next Steps

1. **Run Step 1** to upload your dataset
2. **Run Step 2** to verify
3. **Run Step 3** to prepare filtered datasets
4. **Run Step 4** to train models

**Questions?** Check the troubleshooting section or run `modal --help`

