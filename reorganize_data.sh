#!/bin/bash
# Script to reorganize the data folder into a clean structure

set -e  # Exit on error

BASE_DIR="/Users/yugambahl/Desktop/brain_ct/data"
cd "$BASE_DIR"

echo "=================================================================================="
echo "REORGANIZING DATA FOLDER"
echo "=================================================================================="
echo ""

# Create new directory structure
echo "Creating new directory structure..."
mkdir -p raw_data/dicom_patients
mkdir -p raw_data/ct_images
mkdir -p raw_data/ct_images_png
mkdir -p processed/roboflow_versions
mkdir -p processed/no_hemorrhage_images
mkdir -p training_datasets/current
mkdir -p training_datasets/archived
mkdir -p metadata
mkdir -p documentation

echo "✓ Created directory structure"
echo ""

# Move documentation (if not already moved)
echo "Organizing documentation..."
if [ -d "documentation" ]; then
    echo "✓ Documentation already in place"
else
    echo "⚠️  Documentation folder not found (already moved?)"
fi
echo ""

# Move DICOM patient folders
echo "Moving DICOM patient folders..."
count=0
for dir in *" AnonymousPatient"; do
    if [ -d "$dir" ]; then
        mv "$dir" raw_data/dicom_patients/
        count=$((count+1))
    fi
done
echo "✓ Moved $count DICOM patient folders"
echo ""

# Move CT Images folders
echo "Moving CT Images folders..."
count=0
for dir in *" CT Images"; do
    if [ -d "$dir" ]; then
        mv "$dir" raw_data/ct_images/
        count=$((count+1))
    fi
done
echo "✓ Moved $count CT Images folders"
echo ""

# Move CT Images PNG
echo "Moving CT Images PNG..."
if [ -d "CT Images PNG" ]; then
    mv "CT Images PNG" raw_data/ct_images_png/
    echo "✓ Moved CT Images PNG (1074 files)"
else
    echo "⚠️  CT Images PNG not found"
fi
echo ""

# Move no_hemorrhage_positive_feedback
echo "Moving no_hemorrhage images..."
if [ -d "no_hemorrhage_positive_feedback" ]; then
    mv no_hemorrhage_positive_feedback processed/no_hemorrhage_images/
    echo "✓ Moved no_hemorrhage_positive_feedback"
else
    echo "⚠️  no_hemorrhage_positive_feedback not found"
fi
echo ""

# Move old Roboflow versions
echo "Archiving old Roboflow versions..."
count=0
for dir in UAT_CT_BRAIN_HEMORRHAGE.v2i.yolov8 ct_brain_hemorrhage.v4i.yolov8 ct_brain_hemorrhage.v5i.yolov8 ct_brain_hemorrhage.v6i.yolov8; do
    if [ -d "$dir" ]; then
        mv "$dir" processed/roboflow_versions/
        count=$((count+1))
    fi
done
echo "✓ Moved $count old Roboflow versions"
echo ""

# Keep current training dataset in place, create symlink
echo "Setting up current training dataset..."
if [ -d "UAT_CT_BRAIN_HEMORRHAGE.v2i.yolov8_combined" ]; then
    mv UAT_CT_BRAIN_HEMORRHAGE.v2i.yolov8_combined training_datasets/current/
    # Create symlink in original location for backward compatibility
    ln -s training_datasets/current/UAT_CT_BRAIN_HEMORRHAGE.v2i.yolov8_combined UAT_CT_BRAIN_HEMORRHAGE.v2i.yolov8_combined
    echo "✓ Moved current training dataset (with backward-compatible symlink)"
else
    echo "⚠️  Current training dataset not found"
fi
echo ""

# Move metadata files
echo "Moving metadata files..."
count=0
for file in *.csv *.json *.xlsx *.zip; do
    if [ -f "$file" ]; then
        mv "$file" metadata/
        count=$((count+1))
    fi
done
echo "✓ Moved $count metadata files"
echo ""

echo "=================================================================================="
echo "REORGANIZATION COMPLETE!"
echo "=================================================================================="
echo ""
echo "New structure:"
echo "data/"
echo "├── documentation/              # All docs (already organized)"
echo "├── training_datasets/"
echo "│   ├── current/               # Current balanced dataset (7.5K images)"
echo "│   └── archived/              # Old versions for reference"
echo "├── raw_data/"
echo "│   ├── dicom_patients/        # Original DICOM files by patient"
echo "│   ├── ct_images/             # Extracted CT images by patient"
echo "│   └── ct_images_png/         # PNG exports (1074 files)"
echo "├── processed/"
echo "│   ├── roboflow_versions/     # Old Roboflow exports (v2, v4, v5, v6)"
echo "│   └── no_hemorrhage_images/  # No-hemorrhage dataset (2700 images)"
echo "└── metadata/                  # CSVs, JSONs, spreadsheets"
echo ""