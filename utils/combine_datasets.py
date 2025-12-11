"""
Script to combine old dataset and new dataset with CORRECT class mapping
"""

import os
import shutil
from pathlib import Path
import yaml
from collections import defaultdict
import random

# Configuration
OLD_DATASET_ROOT = Path("/Users/yugambahl/Desktop/brain_ct/data/ct_brain_hemorrhage.v5i.yolov8")
NEW_DATASET_ROOT = Path("/Users/yugambahl/Desktop/brain_ct/data/UAT_CT_BRAIN_HEMORRHAGE.v2i.yolov8")
COMBINED_DATASET_ROOT = Path("/Users/yugambahl/Desktop/brain_ct/data/UAT_CT_BRAIN_HEMORRHAGE.v2i.yolov8_combined")

# CORRECT class mapping: OLD index → NEW index
OLD_TO_NEW_CLASS_INDEX = {
    0: 1,  # hemorrhage contusion → HC
    1: 2,  # intraparenchymal hemorrhage → IPH
    2: 3,  # intraventricular hemorrhage → IVH
    3: 4,  # subarachnoid hemorrhage → SAH
    4: 5,  # subdural hemorrhage → SDH
}

# New dataset split ratios
NEW_TRAIN_RATIO = 0.80
NEW_VAL_RATIO = 0.15
NEW_TEST_RATIO = 0.05

def extract_patient_id(filename):
    """Extract patient ID from filename (format: patientID_slice_...)"""
    return filename.split('_')[0]

def remap_old_label_file(label_path):
    """
    Remap OLD dataset class indices to NEW dataset indices
    OLD: 0,1,2,3,4 → NEW: 1,2,3,4,5
    """
    if not os.path.exists(label_path):
        return
    
    lines = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                old_class_idx = int(parts[0])
                
                # Remap to new index
                if old_class_idx in OLD_TO_NEW_CLASS_INDEX:
                    new_class_idx = OLD_TO_NEW_CLASS_INDEX[old_class_idx]
                    parts[0] = str(new_class_idx)
                    lines.append(' '.join(parts))
                else:
                    print(f"WARNING: Unexpected class index {old_class_idx} in {label_path}")
    
    # Write remapped labels
    with open(label_path, 'w') as f:
        f.write('\n'.join(lines))
        if lines:
            f.write('\n')

def verify_new_label_file(label_path):
    """
    Verify NEW dataset labels are in correct range [0-5]
    Class 0 (EDH) should exist in new dataset
    """
    if not os.path.exists(label_path):
        return True
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_idx = int(parts[0])
                if class_idx < 0 or class_idx > 5:
                    print(f"ERROR: Invalid class index {class_idx} in {label_path}")
                    return False
    return True

def copy_old_dataset_split(src_root, dst_root, split_name, domain_label):
    """Copy OLD dataset and remap class indices"""
    src_img_dir = src_root / split_name / "images"
    src_lbl_dir = src_root / split_name / "labels"
    
    dst_img_dir = dst_root / split_name / "images"
    dst_lbl_dir = dst_root / split_name / "labels"
    
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)
    
    if not src_img_dir.exists():
        print(f"Warning: {src_img_dir} does not exist, skipping")
        return []
    
    copied_files = []
    class_remapping_count = defaultdict(int)
    
    for img_file in src_img_dir.glob("*"):
        if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            # Copy image with domain prefix
            dst_img_name = f"{domain_label}_{img_file.name}"
            dst_img_path = dst_img_dir / dst_img_name
            shutil.copy2(img_file, dst_img_path)
            
            # Copy and remap label
            lbl_file = src_lbl_dir / f"{img_file.stem}.txt"
            dst_lbl_path = dst_lbl_dir / f"{domain_label}_{img_file.stem}.txt"
            
            if lbl_file.exists():
                shutil.copy2(lbl_file, dst_lbl_path)
                
                # Track remapping before doing it
                with open(lbl_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            old_idx = int(parts[0])
                            if old_idx in OLD_TO_NEW_CLASS_INDEX:
                                new_idx = OLD_TO_NEW_CLASS_INDEX[old_idx]
                                class_remapping_count[f"{old_idx}→{new_idx}"] += 1
                
                # Remap class indices
                remap_old_label_file(dst_lbl_path)
            
            copied_files.append(dst_img_name)
    
    print(f"  Class remapping summary for {split_name}:")
    for mapping, count in sorted(class_remapping_count.items()):
        print(f"    {mapping}: {count} instances")
    
    return copied_files

def split_new_dataset_by_patient(new_dataset_root):
    """Split new dataset into train/val/test by patient ID"""
    train_img_dir = new_dataset_root / "train" / "images"
    
    if not train_img_dir.exists():
        raise FileNotFoundError(f"{train_img_dir} not found")
    
    # Group images by patient
    patient_images = defaultdict(list)
    for img_file in train_img_dir.glob("*"):
        if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            patient_id = extract_patient_id(img_file.name)
            patient_images[patient_id].append(img_file.name)
    
    # Split patients
    patient_ids = list(patient_images.keys())
    random.seed(42)
    random.shuffle(patient_ids)
    
    n_patients = len(patient_ids)
    n_train = int(n_patients * NEW_TRAIN_RATIO)
    n_val = int(n_patients * NEW_VAL_RATIO)
    
    train_patients = patient_ids[:n_train]
    val_patients = patient_ids[n_train:n_train + n_val]
    test_patients = patient_ids[n_train + n_val:]
    
    # Create image lists
    splits = {
        'train': [img for pid in train_patients for img in patient_images[pid]],
        'val': [img for pid in val_patients for img in patient_images[pid]],
        'test': [img for pid in test_patients for img in patient_images[pid]]
    }
    
    print(f"\nNew dataset split (5mm thick slices):")
    print(f"  Total patients: {n_patients}")
    print(f"  Train: {len(train_patients)} patients, {len(splits['train'])} images")
    print(f"  Val: {len(val_patients)} patients, {len(splits['val'])} images")
    print(f"  Test: {len(test_patients)} patients, {len(splits['test'])} images")
    
    return splits, {'train': train_patients, 'val': val_patients, 'test': test_patients}

def copy_new_dataset_split(new_root, combined_root, split_images, split_name, domain_label):
    """Copy NEW dataset (already has correct class indices)"""
    src_img_dir = new_root / "train" / "images"
    src_lbl_dir = new_root / "train" / "labels"
    
    dst_img_dir = combined_root / split_name / "images"
    dst_lbl_dir = combined_root / split_name / "labels"
    
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)
    
    class_count = defaultdict(int)
    
    for img_name in split_images:
        # Copy image with domain prefix
        src_img = src_img_dir / img_name
        dst_img_name = f"{domain_label}_{img_name}"
        dst_img = dst_img_dir / dst_img_name
        shutil.copy2(src_img, dst_img)
        
        # Copy label (NO remapping needed - already correct)
        src_lbl = src_lbl_dir / f"{Path(img_name).stem}.txt"
        dst_lbl = dst_lbl_dir / f"{domain_label}_{Path(img_name).stem}.txt"
        
        if src_lbl.exists():
            shutil.copy2(src_lbl, dst_lbl)
            
            # Verify and count classes
            verify_new_label_file(dst_lbl)
            
            with open(src_lbl, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_idx = int(parts[0])
                        class_count[class_idx] += 1
    
    print(f"  Class distribution in {split_name}:")
    class_names = ['EDH', 'HC', 'IPH', 'IVH', 'SAH', 'SDH']
    for idx in sorted(class_count.keys()):
        print(f"    Class {idx} ({class_names[idx]}): {class_count[idx]} instances")

def verify_combined_dataset(combined_root):
    """Verify the combined dataset has correct class indices"""
    print("\n" + "="*80)
    print("VERIFICATION: Checking combined dataset class indices")
    print("="*80)
    
    class_names = ['EDH', 'HC', 'IPH', 'IVH', 'SAH', 'SDH']
    
    for split in ['train', 'valid', 'test']:
        label_dir = combined_root / split / "labels"
        if not label_dir.exists():
            continue
        
        split_class_count = defaultdict(int)
        thin_class_count = defaultdict(int)
        thick_class_count = defaultdict(int)
        invalid_count = 0
        
        for label_file in label_dir.glob("*.txt"):
            is_thin = label_file.name.startswith('thin_')
            is_thick = label_file.name.startswith('thick_')
            
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_idx = int(parts[0])
                        
                        if 0 <= class_idx <= 5:
                            split_class_count[class_idx] += 1
                            if is_thin:
                                thin_class_count[class_idx] += 1
                            elif is_thick:
                                thick_class_count[class_idx] += 1
                        else:
                            invalid_count += 1
                            print(f"  ⚠️  Invalid class {class_idx} in {label_file.name}")
        
        print(f"\n{split.upper()} - Combined Class Distribution:")
        print(f"{'Class':<6} {'Name':<6} {'Total':<8} {'Thin':<8} {'Thick':<8}")
        print("-" * 40)
        for idx in range(6):
            print(f"{idx:<6} {class_names[idx]:<6} {split_class_count[idx]:<8} "
                  f"{thin_class_count[idx]:<8} {thick_class_count[idx]:<8}")
        
        if invalid_count > 0:
            print(f"\n⚠️  Found {invalid_count} invalid class indices in {split}!")
        else:
            print(f"✓ All class indices valid (0-5)")

def create_combined_yaml(combined_root, old_yaml_path, new_yaml_path):
    """Create combined data.yaml"""
    with open(old_yaml_path) as f:
        old_yaml = yaml.safe_load(f)
    
    with open(new_yaml_path) as f:
        new_yaml = yaml.safe_load(f)
    
    # Count images
    train_count = len(list((combined_root / "train" / "images").glob("*")))
    val_count = len(list((combined_root / "valid" / "images").glob("*")))
    test_count = len(list((combined_root / "test" / "images").glob("*")))
    
    # Create new yaml with 6 classes
    combined_yaml = {
        'path': str(combined_root.absolute()),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': 6,  # 6 classes including EDH
        'names': ['EDH', 'HC', 'IPH', 'IVH', 'SAH', 'SDH'],
        'domains': {
            'thin_slices': ['0.625mm', '1.25mm'],
            'thick_slices': ['5mm']
        },
        'class_mapping_note': 'Old dataset classes 0-4 remapped to new indices 1-5. New dataset class 0 (EDH) is new.',
        'combined_info': {
            'total_images': train_count + val_count + test_count,
            'train_images': train_count,
            'val_images': val_count,
            'test_images': test_count,
            'old_dataset': {
                'source': 'ct_brain_hemorrhage.v6i',
                'original_classes': old_yaml['names'],
                'remapped_to_indices': '1-5',
                'prefix': 'thin_'
            },
            'new_dataset': {
                'source': 'uat_ct_brain_hemorrhage-bvzz7.v2',
                'classes': new_yaml['names'],
                'class_indices': '0-5',
                'prefix': 'thick_'
            }
        }
    }
    
    yaml_path = combined_root / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(combined_yaml, f, default_flow_style=False, sort_keys=False)
    
    print(f"\nCreated combined data.yaml at {yaml_path}")
    print(f"Classes: {combined_yaml['names']}")
    print(f"Total images: {combined_yaml['combined_info']['total_images']}")

def main():
    print("=" * 80)
    print("COMBINING CT BRAIN HEMORRHAGE DATASETS WITH CORRECT CLASS MAPPING")
    print("=" * 80)
    
    print("\nCLASS MAPPING VERIFICATION:")
    print("OLD → NEW:")
    old_classes = ['hemorrhage contusion', 'intraparenchymal hemorrhage', 
                   'intraventricular hemorrhage', 'subarachnoid hemorrhage', 
                   'subdural hemorrhage']
    new_classes = ['EDH', 'HC', 'IPH', 'IVH', 'SAH', 'SDH']
    
    for old_idx, new_idx in OLD_TO_NEW_CLASS_INDEX.items():
        print(f"  {old_idx} ({old_classes[old_idx]}) → {new_idx} ({new_classes[new_idx]})")
    
    print(f"\nNEW dataset class 0 (EDH) has no equivalent in old dataset - will be preserved")
    
    input("\nPress Enter to continue with dataset combination...")
    
    # Clean and create combined dataset directory
    if COMBINED_DATASET_ROOT.exists():
        print(f"\nRemoving existing {COMBINED_DATASET_ROOT}")
        shutil.rmtree(COMBINED_DATASET_ROOT)
    COMBINED_DATASET_ROOT.mkdir(parents=True)
    
    # Step 1: Copy old dataset with remapping
    print("\n" + "=" * 80)
    print("Step 1: Copying OLD dataset (thin slices) with class remapping")
    print("=" * 80)
    
    for split in ['train', 'valid', 'test']:
        print(f"\nProcessing old {split} split...")
        files = copy_old_dataset_split(
            OLD_DATASET_ROOT, 
            COMBINED_DATASET_ROOT,
            split,
            domain_label="thin"
        )
        print(f"  Copied {len(files)} images")
    
    # Step 2: Split new dataset by patient
    print("\n" + "=" * 80)
    print("Step 2: Splitting NEW dataset by patient (thick slices)")
    print("=" * 80)
    
    new_splits, new_patients = split_new_dataset_by_patient(NEW_DATASET_ROOT)
    
    # Step 3: Copy new dataset splits
    print("\n" + "=" * 80)
    print("Step 3: Copying NEW dataset splits (NO class remapping needed)")
    print("=" * 80)
    
    for split_name, images in new_splits.items():
        print(f"\nProcessing new {split_name} split...")
        dst_split = 'valid' if split_name == 'val' else split_name
        copy_new_dataset_split(
            NEW_DATASET_ROOT,
            COMBINED_DATASET_ROOT,
            images,
            dst_split,
            domain_label="thick"
        )
        print(f"  Copied {len(images)} images")
    
    # Step 4: Create combined data.yaml
    print("\n" + "=" * 80)
    print("Step 4: Creating combined data.yaml")
    print("=" * 80)
    
    create_combined_yaml(
        COMBINED_DATASET_ROOT,
        OLD_DATASET_ROOT / "data.yaml",
        NEW_DATASET_ROOT / "data.yaml"
    )
    
    # Step 5: Verify combined dataset
    verify_combined_dataset(COMBINED_DATASET_ROOT)
    
    # Step 6: Save patient split info
    print("\n" + "=" * 80)
    print("Step 5: Saving metadata")
    print("=" * 80)
    
    split_info = {
        'class_mapping': {
            'old_to_new_indices': OLD_TO_NEW_CLASS_INDEX,
            'new_class_names': ['EDH', 'HC', 'IPH', 'IVH', 'SAH', 'SDH']
        },
        'thick_slice_patients': {
            'train': new_patients['train'],
            'val': new_patients['val'],
            'test': new_patients['test']
        }
    }
    
    with open(COMBINED_DATASET_ROOT / "combination_metadata.yaml", 'w') as f:
        yaml.dump(split_info, f, default_flow_style=False)
    
    print(f"Saved metadata to {COMBINED_DATASET_ROOT / 'combination_metadata.yaml'}")
    
    print("\n" + "=" * 80)
    print("DATASET COMBINATION COMPLETE!")
    print("=" * 80)
    print(f"\nCombined dataset location: {COMBINED_DATASET_ROOT.absolute()}")
    print("\n⚠️  IMPORTANT: Verify the class distribution using the analysis script!")
    print("="*80)

if __name__ == "__main__":
    main()