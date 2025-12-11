#!/usr/bin/env python3
"""
Script to analyze and balance the brain CT dataset.
Adds no-hemorrhage images to achieve 1:1 ratio for thick slices.
"""

import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

# Configuration
BASE_DIR = "/Users/yugambahl/Desktop/brain_ct/data/UAT_CT_BRAIN_HEMORRHAGE.v2i.yolov8_combined"
NO_HEMORRHAGE_DIR = "/Users/yugambahl/Desktop/brain_ct/data/no_hemorrhage_positive_feedback/png"
SPLITS = ['train', 'valid', 'test']

def extract_patient_id(filename):
    """Extract patient ID from filename (format: patientId_instanceNo)"""
    # Remove prefix (thin_ or thick_) if present
    name = filename
    for prefix in ['thin_', 'thick_']:
        if name.startswith(prefix):
            name = name[len(prefix):]

    # Extract patient ID (first part before underscore)
    parts = name.split('_')
    if len(parts) >= 2:
        return parts[0]
    return None

def has_hemorrhage(label_path):
    """Check if a label file indicates hemorrhage (non-empty file)"""
    if not os.path.exists(label_path):
        return False

    with open(label_path, 'r') as f:
        content = f.read().strip()

    return len(content) > 0

def analyze_dataset():
    """Analyze current dataset distribution"""
    print("\n" + "="*80)
    print("ANALYZING CURRENT DATASET")
    print("="*80)

    stats = {}
    all_patients = {split: set() for split in SPLITS}

    for split in SPLITS:
        img_dir = os.path.join(BASE_DIR, split, "images")
        label_dir = os.path.join(BASE_DIR, split, "labels")

        thick_hemorrhage = 0
        thick_no_hemorrhage = 0
        thin_hemorrhage = 0
        thin_no_hemorrhage = 0

        if not os.path.exists(img_dir):
            print(f"Warning: {img_dir} does not exist")
            continue

        for img_file in os.listdir(img_dir):
            if not img_file.endswith(('.jpg', '.jpeg', '.png')):
                continue

            # Extract patient ID
            patient_id = extract_patient_id(img_file)
            if patient_id:
                all_patients[split].add(patient_id)

            # Get corresponding label file
            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(label_dir, label_file)

            is_hemorrhage = has_hemorrhage(label_path)
            is_thick = img_file.startswith('thick_')

            if is_thick:
                if is_hemorrhage:
                    thick_hemorrhage += 1
                else:
                    thick_no_hemorrhage += 1
            else:  # thin slice
                if is_hemorrhage:
                    thin_hemorrhage += 1
                else:
                    thin_no_hemorrhage += 1

        stats[split] = {
            'thick_hemorrhage': thick_hemorrhage,
            'thick_no_hemorrhage': thick_no_hemorrhage,
            'thin_hemorrhage': thin_hemorrhage,
            'thin_no_hemorrhage': thin_no_hemorrhage,
            'patients': len(all_patients[split])
        }

        print(f"\n{split.upper()}:")
        print(f"  Thick slices:")
        print(f"    - Hemorrhage: {thick_hemorrhage}")
        print(f"    - No hemorrhage: {thick_no_hemorrhage}")
        print(f"    - Total: {thick_hemorrhage + thick_no_hemorrhage}")
        print(f"    - Ratio (hemorrhage:no_hemorrhage): {thick_hemorrhage}:{thick_no_hemorrhage}")
        print(f"  Thin slices:")
        print(f"    - Hemorrhage: {thin_hemorrhage}")
        print(f"    - No hemorrhage: {thin_no_hemorrhage}")
        print(f"    - Total: {thin_hemorrhage + thin_no_hemorrhage}")
        print(f"  Unique patients: {len(all_patients[split])}")

    return stats, all_patients

def get_available_no_hemorrhage_images():
    """Get all available no-hemorrhage images and group by patient"""
    print("\n" + "="*80)
    print("ANALYZING NO-HEMORRHAGE IMAGES")
    print("="*80)

    patient_images = defaultdict(list)

    for img_file in os.listdir(NO_HEMORRHAGE_DIR):
        if not img_file.endswith('.png'):
            continue

        patient_id = extract_patient_id(img_file)
        if patient_id:
            patient_images[patient_id].append(img_file)

    print(f"\nTotal no-hemorrhage images: {sum(len(imgs) for imgs in patient_images.values())}")
    print(f"Unique patients: {len(patient_images)}")
    print(f"Average images per patient: {sum(len(imgs) for imgs in patient_images.values()) / len(patient_images):.1f}")

    return patient_images

def calculate_needed_images(stats):
    """Calculate how many no-hemorrhage images needed for each split"""
    print("\n" + "="*80)
    print("CALCULATING REQUIRED NO-HEMORRHAGE IMAGES")
    print("="*80)

    needed = {}

    for split in SPLITS:
        thick_hemorrhage = stats[split]['thick_hemorrhage']
        thick_no_hemorrhage = stats[split]['thick_no_hemorrhage']

        # Target: 1:1 ratio
        needed_no_hemorrhage = thick_hemorrhage - thick_no_hemorrhage
        needed[split] = max(0, needed_no_hemorrhage)

        print(f"\n{split.upper()}:")
        print(f"  Current thick hemorrhage: {thick_hemorrhage}")
        print(f"  Current thick no-hemorrhage: {thick_no_hemorrhage}")
        print(f"  Target thick no-hemorrhage: {thick_hemorrhage}")
        print(f"  Need to add: {needed[split]} images")

    return needed

def distribute_no_hemorrhage_images(needed, patient_images, existing_patients):
    """
    Distribute no-hemorrhage images to train/valid/test ensuring no patient overlap
    """
    print("\n" + "="*80)
    print("DISTRIBUTING NO-HEMORRHAGE IMAGES")
    print("="*80)

    # Get list of available patients (not in any existing split)
    all_existing_patients = set()
    for split in SPLITS:
        all_existing_patients.update(existing_patients[split])

    available_patients = [p for p in patient_images.keys() if p not in all_existing_patients]

    print(f"\nAvailable patients (not in existing dataset): {len(available_patients)}")
    print(f"Total available images from these patients: {sum(len(patient_images[p]) for p in available_patients)}")

    # Shuffle patients for random distribution
    random.seed(42)
    random.shuffle(available_patients)

    # Calculate total needed
    total_needed = sum(needed.values())
    print(f"\nTotal images needed: {total_needed}")

    # Distribution ratios based on needed amounts
    if total_needed == 0:
        print("\nNo images needed - dataset is already balanced!")
        return {split: [] for split in SPLITS}

    distribution = {}
    images_added = {split: 0 for split in SPLITS}
    patient_idx = 0

    # Distribute patients round-robin to each split until needs are met
    for split in SPLITS:
        distribution[split] = []

    # Priority order based on needs
    splits_order = sorted(SPLITS, key=lambda s: needed[s], reverse=True)

    for split in splits_order:
        while images_added[split] < needed[split] and patient_idx < len(available_patients):
            patient_id = available_patients[patient_idx]
            patient_imgs = patient_images[patient_id]

            # Add all images from this patient to the current split
            distribution[split].extend(patient_imgs)
            images_added[split] += len(patient_imgs)

            print(f"Assigned patient {patient_id} ({len(patient_imgs)} images) to {split}")

            patient_idx += 1

    # Print distribution summary
    print("\n" + "="*80)
    print("DISTRIBUTION SUMMARY")
    print("="*80)

    for split in SPLITS:
        print(f"\n{split.upper()}:")
        print(f"  Needed: {needed[split]}")
        print(f"  Will add: {len(distribution[split])} images")
        print(f"  Status: {'✓ SUFFICIENT' if len(distribution[split]) >= needed[split] else '⚠ INSUFFICIENT'}")

    return distribution

def copy_images_to_dataset(distribution):
    """Copy distributed images to appropriate splits"""
    print("\n" + "="*80)
    print("COPYING IMAGES TO DATASET")
    print("="*80)

    for split, img_files in distribution.items():
        if not img_files:
            continue

        img_dir = os.path.join(BASE_DIR, split, "images")
        label_dir = os.path.join(BASE_DIR, split, "labels")

        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)

        print(f"\nCopying {len(img_files)} images to {split}...")

        for img_file in img_files:
            src_path = os.path.join(NO_HEMORRHAGE_DIR, img_file)

            # Create destination filename with thick_ prefix
            dst_img_file = f"thick_{img_file}"
            dst_img_path = os.path.join(img_dir, dst_img_file)

            # Copy image
            shutil.copy2(src_path, dst_img_path)

            # Create empty label file (no hemorrhage = empty file)
            label_file = os.path.splitext(dst_img_file)[0] + '.txt'
            label_path = os.path.join(label_dir, label_file)

            with open(label_path, 'w') as f:
                pass  # Empty file for no hemorrhage

        print(f"✓ Copied {len(img_files)} images to {split}")

def verify_final_distribution():
    """Verify the final dataset distribution"""
    print("\n" + "="*80)
    print("VERIFYING FINAL DATASET DISTRIBUTION")
    print("="*80)

    analyze_dataset()

def main(auto_confirm=False):
    print("="*80)
    print("BRAIN CT DATASET BALANCING TOOL")
    print("="*80)

    # Step 1: Analyze current dataset
    stats, existing_patients = analyze_dataset()

    # Step 2: Get available no-hemorrhage images
    patient_images = get_available_no_hemorrhage_images()

    # Step 3: Calculate needed images
    needed = calculate_needed_images(stats)

    # Check if we need to add any images
    total_needed = sum(needed.values())
    if total_needed == 0:
        print("\n✓ Dataset is already balanced!")
        return

    # Step 4: Distribute images
    distribution = distribute_no_hemorrhage_images(needed, patient_images, existing_patients)

    # Step 5: Confirm before copying
    print("\n" + "="*80)
    print("READY TO COPY IMAGES")
    print("="*80)

    total_to_add = sum(len(imgs) for imgs in distribution.values())
    print(f"\nTotal images to be added: {total_to_add}")

    # Check for insufficient images
    total_available = sum(len(patient_images[p]) for p in patient_images.keys()
                         if p not in set().union(*existing_patients.values()))

    if total_available < total_needed:
        print(f"\n⚠ WARNING: Not enough no-hemorrhage images available!")
        print(f"   Needed: {total_needed}")
        print(f"   Available: {total_available}")
        print(f"   Shortfall: {total_needed - total_available}")
        print(f"\n   The dataset will be partially balanced with available images.")

    if auto_confirm:
        response = 'yes'
        print("\nAuto-confirming (running in non-interactive mode)...")
    else:
        response = input("\nProceed with copying images? (yes/no): ").strip().lower()

    if response == 'yes':
        # Step 6: Copy images
        copy_images_to_dataset(distribution)

        # Step 7: Verify
        verify_final_distribution()

        print("\n" + "="*80)
        print("✓ DATASET BALANCING COMPLETE")
        print("="*80)
    else:
        print("\nOperation cancelled.")

if __name__ == "__main__":
    import sys
    auto_confirm = '--auto-confirm' in sys.argv
    main(auto_confirm=auto_confirm)
