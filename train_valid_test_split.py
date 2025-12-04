import os
import shutil
import random
import re
from collections import defaultdict
import argparse
import yaml
import sys
from datetime import datetime

def log(message):
    """Print a timestamped log message"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def extract_patient_instance(image_name):
    """Extract patient ID and instance number from image filename"""
    match = re.match(r'^(\d+_\d+)_', image_name)
    if match:
        return match.group(1)  # Return patientid_instance
    return None

def extract_patient_id(image_name):
    """Extract just the patient ID from image filename"""
    match = re.match(r'^(\d+)_\d+_', image_name)
    if match:
        return match.group(1)  # Return just patientid
    return None

def group_by_patient_instance(image_names):
    """Group image filenames by patient ID and instance number"""
    groups = defaultdict(list)
    for image_name in image_names:
        patient_instance = extract_patient_instance(image_name)
        if patient_instance:
            groups[patient_instance].append(image_name)
    return groups

def read_labels(label_dir, image_name):
    """Read class labels from a YOLO format label file"""
    label_path = os.path.join(label_dir, os.path.splitext(image_name)[0] + '.txt')
    classes = set()
    
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    classes.add(class_id)
    
    return classes

def get_instance_classes(train_label_dir, patient_instance_to_images):
    """Get the classes present in each patient_instance"""
    instance_classes = {}
    
    for patient_instance, images in patient_instance_to_images.items():
        # Just check the first image for each instance since all augmentations have the same labels
        if images:
            classes = read_labels(train_label_dir, images[0])
            instance_classes[patient_instance] = classes
    
    return instance_classes

def get_patient_classes(patient_to_instances, instance_classes):
    """Get the classes present in each patient based on its instances"""
    patient_classes = {}
    
    for patient_id, instances in patient_to_instances.items():
        all_classes = set()
        for patient_instance, _ in instances:
            if patient_instance in instance_classes:
                all_classes.update(instance_classes[patient_instance])
        
        patient_classes[patient_id] = all_classes
    
    return patient_classes

def analyze_class_distribution(instance_classes):
    """Analyze the distribution of classes in the dataset"""
    class_counts = defaultdict(int)
    instance_per_class = defaultdict(int)
    
    for patient_instance, classes in instance_classes.items():
        for class_id in classes:
            class_counts[class_id] += 1
            instance_per_class[class_id] += 1
    
    return class_counts, instance_per_class

def split_dataset(dataset_path, val_ratio=0.15, test_ratio=0.05, move_files=True, seed=42, max_test_ratio=0.07):
    """
    Split a YOLOv8 dataset into train, validation, and test sets,
    keeping all augmentations of the same patient_id_instance together and
    ensuring all classes are represented in each split.
    
    Args:
        dataset_path: Root path of the dataset
        val_ratio: Ratio of data to use for validation
        test_ratio: Target ratio of data to use for testing
        move_files: If True, move files to new directories; if False, only calculate splits
        seed: Random seed for reproducibility
        max_test_ratio: Maximum allowable test set ratio
    """
    log(f"Starting dataset split with val_ratio={val_ratio}, test_ratio={test_ratio}, max_test_ratio={max_test_ratio}")
    
    # Define paths
    train_img_dir = os.path.join(dataset_path, 'train', 'images')
    train_label_dir = os.path.join(dataset_path, 'train', 'labels')
    
    if not os.path.exists(train_img_dir):
        log(f"Error: Images directory not found at {train_img_dir}")
        return None
    
    if not os.path.exists(train_label_dir):
        log(f"Warning: Labels directory not found at {train_label_dir}")
    
    # Create new directories if we're moving files
    if move_files:
        val_img_dir = os.path.join(dataset_path, 'val', 'images')
        val_label_dir = os.path.join(dataset_path, 'val', 'labels')
        test_img_dir = os.path.join(dataset_path, 'test', 'images')
        test_label_dir = os.path.join(dataset_path, 'test', 'labels')
        
        for dir_path in [val_img_dir, val_label_dir, test_img_dir, test_label_dir]:
            os.makedirs(dir_path, exist_ok=True)
            log(f"Created directory: {dir_path}")
    
    # Get all image files
    all_images = os.listdir(train_img_dir)
    log(f"Found {len(all_images)} images in training directory")
    
    # Group images by patient_instance (patientid_instance)
    patient_instance_to_images = group_by_patient_instance(all_images)
    log(f"Found {len(patient_instance_to_images)} unique patient-instance combinations")
    
    # Calculate average number of augmentations per patient-instance
    avg_augs = len(all_images) / len(patient_instance_to_images)
    log(f"Average images per patient-instance combination: {avg_augs:.2f}")
    
    # Check augmentation distribution
    aug_counts = [len(imgs) for imgs in patient_instance_to_images.values()]
    min_augs = min(aug_counts)
    max_augs = max(aug_counts)
    log(f"Augmentation distribution: min={min_augs}, max={max_augs}")
    
    # Group by patient ID
    patient_to_instances = defaultdict(list)
    for patient_instance, images in patient_instance_to_images.items():
        patient_id = patient_instance.split('_')[0]
        patient_to_instances[patient_id].append((patient_instance, images))
    
    # Get all unique patient IDs
    all_patients = list(patient_to_instances.keys())
    total_patients = len(all_patients)
    log(f"Found {total_patients} unique patients")
    
    # Get the classes present in each patient_instance
    instance_classes = get_instance_classes(train_label_dir, patient_instance_to_images)
    
    # Get the classes present in each patient
    patient_classes = get_patient_classes(patient_to_instances, instance_classes)
    
    # Analyze class distribution
    class_counts, instances_per_class = analyze_class_distribution(instance_classes)
    
    log(f"Found {len(class_counts)} unique classes in the dataset")
    for class_id, count in sorted(class_counts.items()):
        log(f"Class {class_id}: {count} instances ({instances_per_class[class_id]} unique patient-instances)")
    
    # Create a list of (patient_id, total_images, classes) tuples
    patient_info = []
    for patient_id, instances in patient_to_instances.items():
        total_images = sum(len(images) for _, images in instances)
        patient_info.append((
            patient_id, 
            total_images, 
            patient_classes.get(patient_id, set())
        ))
    
    # Sort by image count (smallest first) and shuffle with seed
    patient_info.sort(key=lambda x: x[1])  
    
    log("Patient distribution:")
    top5 = sorted(patient_info, key=lambda x: x[1], reverse=True)[:5]
    bottom5 = sorted(patient_info, key=lambda x: x[1])[:5]
    log(f"Top 5 patients by image count: {[(p[0], p[1]) for p in top5]}")
    log(f"Bottom 5 patients by image count: {[(p[0], p[1]) for p in bottom5]}")
    
    total_images = len(all_images)
    avg_images_per_patient = total_images / total_patients if total_patients > 0 else 0
    log(f"Average images per patient: {avg_images_per_patient:.2f}")
    
    # Set the random seed for reproducibility
    random.seed(seed)
    
    # Identify all unique classes in the dataset
    all_classes = set().union(*[classes for _, _, classes in patient_info])
    log(f"Dataset contains classes: {sorted(all_classes)}")
    
    # We need to ensure all classes are represented in the test set
    # Shuffle the patient_info to randomize with fixed seed
    random.shuffle(patient_info)
    
    # First, select patients for the test set, ensuring all classes are represented
    test_patients = []
    test_images_count = 0
    test_classes_covered = set()
    target_test_images = total_images * test_ratio
    max_test_images = total_images * max_test_ratio
    
    # First pass: try to get patients that contribute missing classes until we cover all classes
    for patient_id, image_count, classes in patient_info:
        # Skip if adding this patient would exceed the maximum allowed test ratio
        if test_images_count + image_count > max_test_images:
            continue
            
        # Check if this patient contributes any new classes
        new_classes = classes - test_classes_covered
        if new_classes:
            test_patients.append(patient_id)
            test_images_count += image_count
            test_classes_covered.update(classes)
            
            log(f"Selected patient {patient_id} for test set, added classes: {new_classes}")
            
            # If we've covered all classes and reached our target ratio, we can stop
            if test_classes_covered == all_classes and test_images_count >= target_test_images:
                break
    
    # Second pass: If we still haven't covered all classes, try adding more patients
    # even if they don't contribute new classes, to get closer to our target ratio
    if test_classes_covered != all_classes or test_images_count < target_test_images:
        missing_classes = all_classes - test_classes_covered
        log(f"Warning: After first pass, test set is missing classes: {missing_classes}")
        
        # Try adding more patients, prioritizing those that have the missing classes
        remaining_patients = [(pid, ic, cl) for pid, ic, cl in patient_info 
                              if pid not in test_patients]
        
        # Sort remaining patients by how many missing classes they cover
        remaining_patients.sort(key=lambda x: len(missing_classes.intersection(x[2])), reverse=True)
        
        for patient_id, image_count, classes in remaining_patients:
            # Skip if adding this patient would exceed the maximum allowed test ratio
            if test_images_count + image_count > max_test_images:
                continue
                
            test_patients.append(patient_id)
            test_images_count += image_count
            test_classes_covered.update(classes)
            
            log(f"Added patient {patient_id} to test set in second pass")
            
            # If we've covered all classes and reached our target ratio, we can stop
            if test_classes_covered == all_classes and test_images_count >= target_test_images:
                break
    
    # Check if we've covered all classes in the test set
    if test_classes_covered != all_classes:
        missing_classes = all_classes - test_classes_covered
        log(f"Warning: Could not cover all classes in test set. Missing: {missing_classes}")
    
    # If we couldn't get close to the target, log a warning
    if test_images_count < target_test_images * 0.8:  # If we got less than 80% of target
        log(f"Warning: Could only allocate {test_images_count} images ({test_images_count/total_images*100:.1f}%) " +
            f"to test set, which is below the target of {target_test_images} ({test_ratio*100:.1f}%)")
    
    # Collect all test images
    test_images = []
    test_patient_instances = set()
    for patient_id in test_patients:
        for patient_instance, images in patient_to_instances[patient_id]:
            test_images.extend(images)
            test_patient_instances.add(patient_instance)
    
    # Create a set for faster lookup
    test_images_set = set(test_images)
    
    # Collect remaining patient-instances (those not in test set)
    remaining_patient_instances = []
    remaining_instance_classes = {}
    
    for patient_id, instances in patient_to_instances.items():
        if patient_id not in test_patients:
            for patient_instance, images in instances:
                remaining_patient_instances.append((patient_instance, images))
                remaining_instance_classes[patient_instance] = instance_classes.get(patient_instance, set())
    
    # Now select patient-instances for validation, ensuring all classes are represented
    all_classes_remaining = set().union(*[classes for _, classes in remaining_instance_classes.items()])
    log(f"Remaining dataset contains classes: {sorted(all_classes_remaining)}")
    
    # Shuffle remaining patient-instances 
    random.shuffle(remaining_patient_instances)
    
    # Calculate number of images for validation set
    remaining_images_count = total_images - test_images_count
    target_val_images = total_images * val_ratio
    
    # Sort remaining patient-instances by class coverage
    val_patient_instances = []
    val_images = []
    val_images_count = 0
    val_classes_covered = set()
    
    # First pass: try to get patient-instances that contribute missing classes
    for patient_instance, images in remaining_patient_instances:
        # Skip if adding this instance would exceed our validation ratio
        if val_images_count + len(images) > target_val_images:
            continue
            
        # Check if this instance contributes any new classes
        classes = remaining_instance_classes.get(patient_instance, set())
        new_classes = classes - val_classes_covered
        
        if new_classes:
            val_patient_instances.append(patient_instance)
            val_images.extend(images)
            val_images_count += len(images)
            val_classes_covered.update(classes)
            
            # If we've covered all classes and reached our target ratio, we can stop
            if val_classes_covered == all_classes_remaining and val_images_count >= target_val_images:
                break
    
    # Second pass: If we still haven't covered all classes or reached our target ratio,
    # add more instances even if they don't contribute new classes
    if val_classes_covered != all_classes_remaining or val_images_count < target_val_images:
        missing_classes = all_classes_remaining - val_classes_covered
        if missing_classes:
            log(f"Warning: After first pass, validation set is missing classes: {missing_classes}")
        
        # Get remaining patient-instances not yet selected for validation
        remaining = [(pi, imgs) for pi, imgs in remaining_patient_instances 
                    if pi not in val_patient_instances]
        
        # Sort by how many missing classes they cover
        remaining.sort(key=lambda x: len(missing_classes.intersection(
            remaining_instance_classes.get(x[0], set()))), reverse=True)
        
        for patient_instance, images in remaining:
            # Skip if adding this instance would exceed our validation ratio
            if val_images_count + len(images) > target_val_images:
                continue
                
            val_patient_instances.append(patient_instance)
            val_images.extend(images)
            val_images_count += len(images)
            val_classes_covered.update(remaining_instance_classes.get(patient_instance, set()))
            
            # If we've covered all classes and reached our target ratio, we can stop
            if val_classes_covered == all_classes_remaining and val_images_count >= target_val_images:
                break
    
    # Check if we've covered all classes in the validation set
    if val_classes_covered != all_classes_remaining:
        missing_classes = all_classes_remaining - val_classes_covered
        log(f"Warning: Could not cover all classes in validation set. Missing: {missing_classes}")
    
    # Create a set for faster lookup
    val_images_set = set(val_images)
    
    # The rest stay in training
    train_images = [img for img in all_images if img not in test_images_set and img not in val_images_set]
    
    # Analyze class distribution in each split
    train_classes = set()
    for img in train_images:
        patient_instance = extract_patient_instance(img)
        if patient_instance in instance_classes:
            train_classes.update(instance_classes[patient_instance])
    
    val_classes = set()
    for img in val_images:
        patient_instance = extract_patient_instance(img)
        if patient_instance in instance_classes:
            val_classes.update(instance_classes[patient_instance])
    
    test_classes = set()
    for img in test_images:
        patient_instance = extract_patient_instance(img)
        if patient_instance in instance_classes:
            test_classes.update(instance_classes[patient_instance])
    
    # Print statistics
    log("Split statistics:")
    log(f"Total patients: {total_patients}")
    log(f"Total patient-instance combinations: {len(patient_instance_to_images)}")
    log(f"Total images: {total_images}")
    log(f"Test patients: {len(test_patients)} ({len(test_images)} images, {len(test_images)/total_images*100:.1f}%)")
    log(f"Test classes: {sorted(test_classes)}")
    log(f"Validation instances: {len(val_patient_instances)} ({len(val_images)} images, {len(val_images)/total_images*100:.1f}%)")
    log(f"Validation classes: {sorted(val_classes)}")
    log(f"Training images: {len(train_images)} ({len(train_images)/total_images*100:.1f}%)")
    log(f"Training classes: {sorted(train_classes)}")
    
    # Move files if requested
    if move_files:
        # Create a backup directory for the original training data
        backup_dir = os.path.join(dataset_path, 'backup')
        original_img_dir = os.path.join(backup_dir, 'images')
        original_label_dir = os.path.join(backup_dir, 'labels')
        
        if not os.path.exists(backup_dir):
            os.makedirs(original_img_dir, exist_ok=True)
            os.makedirs(original_label_dir, exist_ok=True)
            log(f"Created backup directories: {backup_dir}")
            
            # Make a backup of all original images and labels
            log("Creating backup of original dataset...")
            for image_name in all_images:
                src_img_path = os.path.join(train_img_dir, image_name)
                dst_img_path = os.path.join(original_img_dir, image_name)
                shutil.copy2(src_img_path, dst_img_path)
                
                # Copy corresponding label file
                label_name = os.path.splitext(image_name)[0] + '.txt'
                src_label_path = os.path.join(train_label_dir, label_name)
                dst_label_path = os.path.join(original_label_dir, label_name)
                if os.path.exists(src_label_path):
                    shutil.copy2(src_label_path, dst_label_path)
        
        def move_files_to_dir(image_names, target_img_dir, target_label_dir):
            moved = 0
            for image_name in image_names:
                # Move image
                src_img_path = os.path.join(train_img_dir, image_name)
                dst_img_path = os.path.join(target_img_dir, image_name)
                
                # Move corresponding label file
                label_name = os.path.splitext(image_name)[0] + '.txt'
                src_label_path = os.path.join(train_label_dir, label_name)
                dst_label_path = os.path.join(target_label_dir, label_name)
                
                if os.path.exists(src_img_path):
                    shutil.copy2(src_img_path, dst_img_path)
                    os.remove(src_img_path)  # Remove from train dir
                    
                    if os.path.exists(src_label_path):
                        shutil.copy2(src_label_path, dst_label_path)
                        os.remove(src_label_path)  # Remove from train dir
                        moved += 1
            
            return moved
        
        # Move test and validation files
        log("Moving test images and labels...")
        test_moved = move_files_to_dir(test_images, test_img_dir, test_label_dir)
        
        log("Moving validation images and labels...")
        val_moved = move_files_to_dir(val_images, val_img_dir, val_label_dir)
        
        log(f"Moved {test_moved}/{len(test_images)} test images and {val_moved}/{len(val_images)} validation images")
    
    # Return statistics for verification
    return {
        "total_patients": total_patients,
        "total_patient_instances": len(patient_instance_to_images),
        "total_images": total_images,
        "test_patients": len(test_patients),
        "test_images": len(test_images),
        "test_ratio": len(test_images)/total_images,
        "test_classes": sorted(list(test_classes)),
        "val_patient_instances": len(val_patient_instances),
        "val_images": len(val_images),
        "val_ratio": len(val_images)/total_images,
        "val_classes": sorted(list(val_classes)),
        "train_images": len(train_images),
        "train_ratio": len(train_images)/total_images,
        "train_classes": sorted(list(train_classes)),
        "test_patient_ids": test_patients
    }

def update_yaml_config(dataset_path, split_stats=None):
    """
    Update the data.yaml file with the new directory structure
    
    Args:
        dataset_path: Root path of the dataset
        split_stats: Statistics from the split operation
    """
    yaml_path = os.path.join(dataset_path, 'data.yaml')
    
    if not os.path.exists(yaml_path):
        log(f"YAML file not found at {yaml_path}")
        # Create a new one with default values
        config = {
            'path': dataset_path,
            'train': os.path.join(dataset_path, 'train'),
            'val': os.path.join(dataset_path, 'val'),
            'test': os.path.join(dataset_path, 'test'),
            'names': {'0': 'hemorrhage'},  # Assuming class 0 is hemorrhage, adjust if needed
            'nc': 1  # Number of classes
        }
    else:
        # Read existing YAML
        with open(yaml_path, 'r') as f:
            try:
                config = yaml.safe_load(f)
                if config is None:
                    config = {}
            except Exception as e:
                log(f"Error reading YAML file: {e}")
                config = {}
        
        # Update paths
        config['path'] = dataset_path
        config['train'] = os.path.join(dataset_path, 'train')
        config['val'] = os.path.join(dataset_path, 'val')
        config['test'] = os.path.join(dataset_path, 'test')
    
    # Add split statistics if provided
    if split_stats:
        config['split_stats'] = split_stats
    
    # Write updated YAML
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    log(f"Updated YAML configuration at {yaml_path}")

def validate_dataset_split(dataset_path):
    """
    Validate that the dataset was properly split and that test set has unique patients
    and all splits contain all classes.
    
    Args:
        dataset_path: Root path of the dataset
    """
    log("Validating dataset split...")
    
    # Define paths
    train_img_dir = os.path.join(dataset_path, 'train', 'images')
    train_label_dir = os.path.join(dataset_path, 'train', 'labels')
    val_img_dir = os.path.join(dataset_path, 'val', 'images')
    val_label_dir = os.path.join(dataset_path, 'val', 'labels')
    test_img_dir = os.path.join(dataset_path, 'test', 'images')
    test_label_dir = os.path.join(dataset_path, 'test', 'labels')
    
    # Get all image files
    train_images = os.listdir(train_img_dir) if os.path.exists(train_img_dir) else []
    val_images = os.listdir(val_img_dir) if os.path.exists(val_img_dir) else []
    test_images = os.listdir(test_img_dir) if os.path.exists(test_img_dir) else []
    
    # Count total images
    total_images = len(train_images) + len(val_images) + len(test_images)
    
    # Extract patient IDs
    train_patients = set(extract_patient_id(img) for img in train_images if extract_patient_id(img))
    val_patients = set(extract_patient_id(img) for img in val_images if extract_patient_id(img))
    test_patients = set(extract_patient_id(img) for img in test_images if extract_patient_id(img))
    
    # Extract patient-instance combinations
    train_instances = set(extract_patient_instance(img) for img in train_images if extract_patient_instance(img))
    val_instances = set(extract_patient_instance(img) for img in val_images if extract_patient_instance(img))
    test_instances = set(extract_patient_instance(img) for img in test_images if extract_patient_instance(img))
    
    # Extract class information
    train_classes = set()
    for img in train_images:
        classes = read_labels(train_label_dir, img)
        train_classes.update(classes)
    
    val_classes = set()
    for img in val_images:
        classes = read_labels(val_label_dir, img)
        val_classes.update(classes)
    
    test_classes = set()
    for img in test_images:
        classes = read_labels(test_label_dir, img)
        test_classes.update(classes)
    
    # Get all unique classes
    all_classes = train_classes.union(val_classes).union(test_classes)
    
    # Check for patient overlap
    train_val_overlap = train_patients.intersection(val_patients)
    train_test_overlap = train_patients.intersection(test_patients)
    val_test_overlap = val_patients.intersection(test_patients)
    
    # Check for patient-instance distribution
    train_instance_groups = group_by_patient_instance(train_images)
    val_instance_groups = group_by_patient_instance(val_images)
    test_instance_groups = group_by_patient_instance(test_images)
    
    # Calculate ratios
    val_ratio = len(val_images) / total_images if total_images > 0 else 0
    test_ratio = len(test_images) / total_images if total_images > 0 else 0
    
    # Print validation results
    log("=== Dataset Split Validation ===")
    log(f"Total images: {total_images}")
    log(f"Training images: {len(train_images)} ({len(train_images)/total_images*100:.1f}%)")
    log(f"Validation images: {len(val_images)} ({val_ratio*100:.1f}%)")
    log(f"Test images: {len(test_images)} ({test_ratio*100:.1f}%)")
    log("\n=== Class Distribution ===")
    log(f"All classes in dataset: {sorted(all_classes)}")
    log(f"Training classes: {sorted(train_classes)}")
    log(f"Validation classes: {sorted(val_classes)}")
    log(f"Test classes: {sorted(test_classes)}")
    log("\n=== Patient Distribution ===")
    log(f"Training patients: {len(train_patients)}")
    log(f"Validation patients: {len(val_patients)}")
    log(f"Test patients: {len(test_patients)}")
    log("\n=== Patient-Instance Distribution ===")
    log(f"Training patient-instances: {len(train_instances)}")
    log(f"Validation patient-instances: {len(val_instances)}")
    log(f"Test patient-instances: {len(test_instances)}")
    log("\n=== Patient Overlap ===")
    log(f"Train-Val patient overlap: {len(train_val_overlap)} patients")
    log(f"Train-Test patient overlap: {len(train_test_overlap)} patients")
    log(f"Val-Test patient overlap: {len(val_test_overlap)} patients")
    
    # Validation checks
    validation_passed = True
    
    # Check if test set has unique patients
    if len(train_test_overlap) == 0:
        log("✅ PASS: Test set contains completely unique patients not seen in training")
    else:
        log("❌ FAIL: Test set contains patients that appear in the training set")
        log(f"Overlapping patients: {train_test_overlap}")
        validation_passed = False
    
    # Check validation split percentage (less strict range)
    if 0.12 <= val_ratio <= 0.18:
        log(f"✅ PASS: Validation split is close to 15% (actual: {val_ratio*100:.1f}%)")
    else:
        log(f"❌ FAIL: Validation split is not close to 15% (actual: {val_ratio*100:.1f}%)")
        validation_passed = False
    
    # Check test split percentage (less strict range)
    if 0.03 <= test_ratio <= 0.07:
        log(f"✅ PASS: Test split is close to 5% (actual: {test_ratio*100:.1f}%)")
    else:
        log(f"❌ FAIL: Test split is not close to 5% (actual: {test_ratio*100:.1f}%)")
        validation_passed = False
    
    # Check if all splits contain all classes
    log("\n=== Class Representation ===")
    
    # Check training set classes
    missing_train_classes = all_classes - train_classes
    if not missing_train_classes:
        log("✅ PASS: Training set contains all classes")
    else:
        log(f"❌ FAIL: Training set is missing classes: {missing_train_classes}")
        validation_passed = False
    
    # Check validation set classes
    missing_val_classes = all_classes - val_classes
    if not missing_val_classes:
        log("✅ PASS: Validation set contains all classes")
    else:
        log(f"❌ FAIL: Validation set is missing classes: {missing_val_classes}")
        validation_passed = False
    
    # Check test set classes
    missing_test_classes = all_classes - test_classes
    if not missing_test_classes:
        log("✅ PASS: Test set contains all classes")
    else:
        log(f"❌ FAIL: Test set is missing classes: {missing_test_classes}")
        validation_passed = False
    
    # Check that all patient-instances have the same number of augmentations in each set
    def check_augmentation_consistency(instance_groups):
        counts = [len(images) for images in instance_groups.values()]
        if not counts:
            return True
        return min(counts) == max(counts)
    
    train_aug_consistent = check_augmentation_consistency(train_instance_groups)
    val_aug_consistent = check_augmentation_consistency(val_instance_groups)
    test_aug_consistent = check_augmentation_consistency(test_instance_groups)
    
    log("\n=== Augmentation Consistency ===")
    if train_aug_consistent:
        log("✅ PASS: All training patient-instances have consistent number of augmentations")
    else:
        log("❌ FAIL: Training patient-instances have inconsistent numbers of augmentations")
        validation_passed = False
    
    if val_aug_consistent:
        log("✅ PASS: All validation patient-instances have consistent number of augmentations")
    else:
        log("❌ FAIL: Validation patient-instances have inconsistent numbers of augmentations")
        validation_passed = False
    
    if test_aug_consistent:
        log("✅ PASS: All test patient-instances have consistent number of augmentations")
    else:
        log("❌ FAIL: Test patient-instances have inconsistent numbers of augmentations")
        validation_passed = False
    
    # Verify that all patient-instances are kept together in one split
    all_instances = train_instances.union(val_instances).union(test_instances)
    split_integrity_passed = True
    
    for instance in all_instances:
        in_train = instance in train_instances
        in_val = instance in val_instances
        in_test = instance in test_instances
        
        # Each instance should only be in ONE of the splits
        if (in_train and in_val) or (in_train and in_test) or (in_val and in_test):
            if split_integrity_passed:  # Only print header once
                log("\n=== Split Integrity Issues ===")
                split_integrity_passed = False
            log(f"❌ FAIL: Patient-instance {instance} appears in multiple splits")
            validation_passed = False
    
    if split_integrity_passed:
        log("\n=== Split Integrity ===")
        log("✅ PASS: All patient-instances are kept together in the same split")
    
    # Check for incomplete label files
    def check_missing_labels(img_dir, label_dir):
        missing = 0
        if not os.path.exists(img_dir) or not os.path.exists(label_dir):
            return 0
        
        for img_file in os.listdir(img_dir):
            base_name = os.path.splitext(img_file)[0]
            label_file = base_name + '.txt'
            if not os.path.exists(os.path.join(label_dir, label_file)):
                missing += 1
        return missing
    
    missing_train_labels = check_missing_labels(train_img_dir, train_label_dir)
    missing_val_labels = check_missing_labels(val_img_dir, val_label_dir)
    missing_test_labels = check_missing_labels(test_img_dir, test_label_dir)
    
    log("\n=== Label Integrity ===")
    if missing_train_labels == 0:
        log("✅ PASS: All training images have corresponding label files")
    else:
        log(f"❌ FAIL: {missing_train_labels} training images are missing label files")
        validation_passed = False
    
    if missing_val_labels == 0:
        log("✅ PASS: All validation images have corresponding label files")
    else:
        log(f"❌ FAIL: {missing_val_labels} validation images are missing label files")
        validation_passed = False
    
    if missing_test_labels == 0:
        log("✅ PASS: All test images have corresponding label files")
    else:
        log(f"❌ FAIL: {missing_test_labels} test images are missing label files")
        validation_passed = False
    
    return validation_passed

def main():
    parser = argparse.ArgumentParser(description='Split YOLOv8 dataset with augmentations into train, validation, and test sets, ensuring class distribution across splits.')
    parser.add_argument('--val-ratio', type=float, default=0.15, help='Ratio of data to use for validation (default: 0.15)')
    parser.add_argument('--test-ratio', type=float, default=0.05, help='Target ratio of data to use for testing (default: 0.05)')
    parser.add_argument('--max-test-ratio', type=float, default=0.07, help='Maximum allowed test ratio (default: 0.07)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--dry-run', action='store_true', help='Calculate splits without moving files')
    parser.add_argument('--restore-backup', action='store_true', help='Restore from backup before running')
    
    args = parser.parse_args()

    dataset_path = "data/ct_brain_hemorrhage.v6i.yolov8"
    
    # Check if we should restore from backup first
    if args.restore_backup:
        backup_dir = os.path.join(dataset_path, 'backup')
        if os.path.exists(backup_dir):
            log("Restoring from backup...")
            
            # Define paths
            backup_img_dir = os.path.join(backup_dir, 'images')
            backup_label_dir = os.path.join(backup_dir, 'labels')
            train_img_dir = os.path.join(dataset_path, 'train', 'images')
            train_label_dir = os.path.join(dataset_path, 'train', 'labels')
            
            # Ensure training directories exist
            os.makedirs(train_img_dir, exist_ok=True)
            os.makedirs(train_label_dir, exist_ok=True)
            
            # Remove existing test and val directories
            val_dir = os.path.join(dataset_path, 'val')
            test_dir = os.path.join(dataset_path, 'test')
            
            if os.path.exists(val_dir):
                shutil.rmtree(val_dir)
                log(f"Removed validation directory: {val_dir}")
            
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)
                log(f"Removed test directory: {test_dir}")
            
            # Copy all files from backup to train
            backup_images = os.listdir(backup_img_dir)
            for image_name in backup_images:
                src_img_path = os.path.join(backup_img_dir, image_name)
                dst_img_path = os.path.join(train_img_dir, image_name)
                shutil.copy2(src_img_path, dst_img_path)
                
                # Copy corresponding label file
                label_name = os.path.splitext(image_name)[0] + '.txt'
                src_label_path = os.path.join(backup_label_dir, label_name)
                dst_label_path = os.path.join(train_label_dir, label_name)
                if os.path.exists(src_label_path):
                    shutil.copy2(src_label_path, dst_label_path)
            
            log(f"Restored {len(backup_images)} images and labels from backup")
        else:
            log("Warning: Backup directory not found, cannot restore")
    
    log(f"Starting dataset split for {dataset_path}")
    log(f"Parameters: val_ratio={args.val_ratio}, test_ratio={args.test_ratio}, " +
        f"max_test_ratio={args.max_test_ratio}, seed={args.seed}, dry_run={args.dry_run}")
    
    # Split the dataset
    split_stats = split_dataset(
        dataset_path, 
        val_ratio=args.val_ratio, 
        test_ratio=args.test_ratio,
        max_test_ratio=args.max_test_ratio,
        move_files=not args.dry_run,
        seed=args.seed
    )
    
    if split_stats is None:
        log("Error: Failed to split dataset")
        return 1
    
    # Update YAML configuration
    if not args.dry_run:
        update_yaml_config(dataset_path, split_stats)
        
        # Validate the split
        validation_passed = validate_dataset_split(dataset_path)
        if validation_passed:
            log("Dataset split validation passed!")
        else:
            log("Dataset split validation failed. Please check the logs above for details.")
    
    log("Operation completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())