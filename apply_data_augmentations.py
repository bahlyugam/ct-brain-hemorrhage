#!/usr/bin/env python3
import os
import glob
import yaml
import cv2
import numpy as np
import shutil
from tqdm import tqdm
import random
from pathlib import Path
import re

def adjust_bbox(bboxes, transform, img_shape, angle=None):
    """Adjust YOLO bbox for geometric transforms"""
    h, w = img_shape[:2]
    adjusted = []
    for bbox in bboxes:
        cls, x, y, bw, bh = map(float, bbox.strip().split())

        if transform == 'horizontal_flip':
            x = 1.0 - x
        elif transform == 'vertical_flip':
            y = 1.0 - y
        elif transform == 'rotation' and angle:
            # Convert YOLO to absolute center coords
            abs_x = x * w
            abs_y = y * h
            if angle == 90:
                abs_x, abs_y = abs_y, w - abs_x
            elif angle == 180:
                abs_x, abs_y = w - abs_x, h - abs_y
            elif angle == 270:
                abs_x, abs_y = h - abs_y, abs_x
            x = abs_x / w
            y = abs_y / h

        adjusted.append(f"{int(cls)} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}")
    return adjusted

def load_dataset_structure(dataset_dir):
    """Load dataset structure and yaml file"""
    # Verify that dataset_dir exists
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory {dataset_dir} not found")
    
    # Check if data.yaml exists
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"data.yaml not found in {dataset_dir}")
    
    # Load data.yaml
    with open(yaml_path, 'r') as file:
        data_yaml = yaml.safe_load(file)
    
    # Check for required folders
    required_folders = ['train']
    missing_folders = [folder for folder in required_folders if not os.path.exists(os.path.join(dataset_dir, folder))]
    if missing_folders:
        raise FileNotFoundError(f"Missing required folders: {', '.join(missing_folders)}")
    
    return data_yaml

def extract_patient_instance_info(filename):
    """Extract patient ID and instance number from filename.
    Assumes a naming pattern like '294761551_114_png.rf.41e2a56a9d874bb1419fa384d8ffd92c.jpg'
    where 294761551_114 is the patient_instance combination.
    
    Returns:
        str: patient_instance combination or None if pattern not found
    """
    # Extract the base filename without extension and path
    base_name = os.path.basename(filename)
    
    # Using regex to find the patient_instance combination
    # Pattern looks for digits followed by underscore followed by digits at the start of the filename
    match = re.search(r'^(\d+_\d+)', base_name)
    if match:
        return match.group(1)  # Return the full match as one identifier
    return None

def get_image_files(dataset_dir):
    """Get all image files from train, valid, and test folders"""
    image_files = []
    
    for split in ['train', 'valid', 'test']:
        split_dir = os.path.join(dataset_dir, split)
        images_dir = os.path.join(split_dir, 'images')
        
        if os.path.exists(images_dir):
            for img_path in glob.glob(os.path.join(images_dir, '*.jpg')) + \
                           glob.glob(os.path.join(images_dir, '*.jpeg')) + \
                           glob.glob(os.path.join(images_dir, '*.png')):
                label_path = os.path.join(split_dir, 'labels', os.path.basename(img_path).rsplit('.', 1)[0] + '.txt')
                
                if os.path.exists(label_path):
                    # Extract patient ID and instance number
                    patient_info = extract_patient_instance_info(img_path)
                    image_files.append((img_path, label_path, split, patient_info))
    
    return image_files

def group_images_by_patient_instance(image_files):
    """Group images by patient_instance combination"""
    groups = {}
    for img_path, label_path, split, patient_instance in image_files:
        if patient_instance is None:
            continue  # Skip files that don't match the pattern
        key = patient_instance  # Use the patient_instance string directly as the key
        if key not in groups:
            groups[key] = []
        groups[key].append((img_path, label_path, split))
    return groups

def apply_gaussian_blur(image, ksize=5):
    """Apply Gaussian blur augmentation"""
    return cv2.GaussianBlur(image, (ksize, ksize), 0)

def apply_random_gamma(image):
    """Apply random gamma correction"""
    gamma = random.uniform(0.5, 1.5)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(0, 256)]).astype(np.uint8)
    return cv2.LUT(image, table)

def apply_gaussian_noise(image, mean=0, std=10):
    noise = np.random.normal(mean, std, image.shape).astype(np.uint8)
    return cv2.add(image, noise)

def apply_random_brightness_contrast(image):
    """Apply random brightness and contrast adjustment"""
    brightness = random.uniform(0.5, 1.5)
    contrast = random.uniform(0.5, 1.5)
    
    # Apply brightness
    bright_img = cv2.convertScaleAbs(image, alpha=brightness, beta=0)
    
    # Apply contrast
    mean = np.mean(bright_img)
    contrast_img = cv2.convertScaleAbs(bright_img, alpha=contrast, beta=(1.0 - contrast) * mean)
    
    return contrast_img

def apply_horizontal_flip(image):
    return cv2.flip(image, 1)

def apply_vertical_flip(image):
    return cv2.flip(image, 0)

def apply_rotation(image, angle=None):
    if angle is None:
        angle = random.choice([90, 180, 270])
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def apply_hsv_jitter(image, hgain=0.015, sgain=0.7, vgain=0.4):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(img)
    h = (h + random.uniform(-hgain, hgain) * 360) % 360
    s *= random.uniform(1 - sgain, 1 + sgain)
    v *= random.uniform(1 - vgain, 1 + vgain)
    hsv_aug = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv_aug.astype(np.uint8), cv2.COLOR_HSV2BGR)

def generate_augmentation_name(original_name, aug_type):
    """Generate a new filename for the augmented image"""
    name_parts = os.path.basename(original_name).rsplit('.', 1)
    base_name = name_parts[0]
    extension = name_parts[1] if len(name_parts) > 1 else 'jpg'
    
    return f"{base_name}_{aug_type}.{extension}"

def augment_dataset(dataset_dir):
    print(f"Loading dataset from {dataset_dir}...")
    data_yaml = load_dataset_structure(dataset_dir)
    image_files = get_image_files(dataset_dir)
    print(f"Found {len(image_files)} images with corresponding labels")

    # Group images by patient_instance combination
    grouped_images = group_images_by_patient_instance(image_files)
    print(f"Found {len(grouped_images)} patient/instance combinations")

    # Define all available augmentation functions
    # This makes it easy to add/remove augmentations later
    augmentation_functions = {
        'random_gamma': apply_random_gamma,
        'brightness_contrast': apply_random_brightness_contrast,
        'gaussian_blur': apply_gaussian_blur,
        'gaussian_noise': apply_gaussian_noise,
        'horizontal_flip': apply_horizontal_flip,
        'vertical_flip': apply_vertical_flip,
        'rotation': apply_rotation,
        'hsv_jitter': apply_hsv_jitter
    }
    
    # Configure which augmentations to apply
    # To apply to all images in a group, add to 'apply_to_all'
    # To apply to only one random image per group, add to 'apply_to_one'
    apply_to_all = []  # No augmentations applied to all images
    apply_to_one = ['random_gamma']  # Only random_gamma applied to one image per group

    # Process each group (patient_instance combination)
    for patient_instance, group in tqdm(grouped_images.items(), desc="Processing groups"):
        # Apply augmentations to all images in the group (if any)
        if apply_to_all:
            for img_path, label_path, split in group:
                image = cv2.imread(img_path)
                if image is None:
                    print(f"Warning: Could not read {img_path}, skipping...")
                    continue

                with open(label_path, 'r') as file:
                    labels = file.readlines()

                # Apply augmentations to all images
                for aug_name in apply_to_all:
                    if aug_name in augmentation_functions:
                        aug_func = augmentation_functions[aug_name]
                        aug_image = aug_func(image.copy())
                        aug_img_filename = generate_augmentation_name(img_path, aug_name)
                        aug_label_filename = generate_augmentation_name(label_path, aug_name)

                        cv2.imwrite(os.path.join(os.path.dirname(img_path), os.path.basename(aug_img_filename)), aug_image)
                        with open(os.path.join(os.path.dirname(label_path), os.path.basename(aug_label_filename)), 'w') as file:
                            file.writelines(labels)
        
        # Apply augmentations to only one randomly selected image in each group
        if apply_to_one and group:  # Ensure the group is not empty
            selected_img_path, selected_label_path, selected_split = random.choice(group)
            image = cv2.imread(selected_img_path)
            
            if image is None:
                print(f"Warning: Could not read {selected_img_path} for single-image augmentation, skipping...")
                continue
                
            with open(selected_label_path, 'r') as file:
                labels = file.readlines()
            
            # Apply augmentations to one image per group
            for aug_name in apply_to_one:
                if aug_name in augmentation_functions:
                    aug_func = augmentation_functions[aug_name]
                    aug_image = aug_func(image.copy())
                    aug_img_filename = generate_augmentation_name(selected_img_path, aug_name)
                    aug_label_filename = generate_augmentation_name(selected_label_path, aug_name)
                    
                    cv2.imwrite(os.path.join(os.path.dirname(selected_img_path), os.path.basename(aug_img_filename)), aug_image)
                    with open(os.path.join(os.path.dirname(selected_label_path), os.path.basename(aug_label_filename)), 'w') as file:
                        file.writelines(labels)
                    
                    print(f"Applied {aug_name} to one image from patient_instance {patient_instance}")

    print("Augmentation complete!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Apply data augmentations to a YOLO dataset")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the dataset directory")
    args = parser.parse_args()
    
    augment_dataset(args.dataset_dir)