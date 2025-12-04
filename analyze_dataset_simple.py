#!/usr/bin/env python3
"""
Simple script to analyze brain CT hemorrhage dataset metrics.
Provides detailed statistics including original/augmented images, normal/abnormal cases, etc.
"""

import os
import yaml
from collections import defaultdict
from PIL import Image

def count_images_by_prefix(directory):
    """Count thin and thick slice images in a directory."""
    if not os.path.exists(directory):
        return 0, 0

    files = os.listdir(directory)
    thin_count = sum(1 for f in files if f.startswith('thin_') and (f.endswith('.png') or f.endswith('.jpg')))
    thick_count = sum(1 for f in files if f.startswith('thick_') and (f.endswith('.png') or f.endswith('.jpg')))

    return thin_count, thick_count

def get_image_resolution(directory):
    """Get the resolution of the first image found."""
    if not os.path.exists(directory):
        return None

    for file in os.listdir(directory):
        if file.endswith('.png') or file.endswith('.jpg'):
            try:
                img_path = os.path.join(directory, file)
                with Image.open(img_path) as img:
                    return img.size
            except:
                continue
    return None

def count_images_with_annotations(label_dir):
    """Count images that have any annotations (abnormal) vs no annotations (normal)."""
    if not os.path.exists(label_dir):
        return 0, 0

    abnormal_images = set()
    normal_images = set()

    for label_file in os.listdir(label_dir):
        if not label_file.endswith('.txt'):
            continue

        label_path = os.path.join(label_dir, label_file)
        image_name = label_file.replace('.txt', '')

        try:
            with open(label_path, 'r') as f:
                content = f.read().strip()
                if content:  # Has annotations
                    abnormal_images.add(image_name)
                else:  # Empty label file = normal
                    normal_images.add(image_name)
        except:
            continue

    return len(abnormal_images), len(normal_images)

def is_augmented_image(filename):
    """Check if image is augmented based on filename pattern."""
    # Augmented images typically have .rf. or specific augmentation markers
    return '.rf.' in filename

def count_original_vs_augmented(image_dir):
    """Count original vs augmented images."""
    if not os.path.exists(image_dir):
        return 0, 0

    original_count = 0
    augmented_count = 0

    for file in os.listdir(image_dir):
        if file.endswith('.png') or file.endswith('.jpg'):
            if is_augmented_image(file):
                augmented_count += 1
            else:
                original_count += 1

    return original_count, augmented_count

def analyze_dataset(data_yaml_path):
    """Analyze the dataset and return comprehensive metrics."""

    if not os.path.exists(data_yaml_path):
        print(f"Error: {data_yaml_path} not found!")
        return None

    # Load YAML config
    with open(data_yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    base_path = config.get('path', '')
    names = config.get('names', [])

    metrics = {
        'dataset_path': base_path,
        'classes': names,
        'num_classes': len(names),
        'splits': {}
    }

    # Analyze each split
    for split in ['train', 'valid', 'test']:
        img_dir = os.path.join(base_path, split, 'images')
        label_dir = os.path.join(base_path, split, 'labels')

        thin_count, thick_count = count_images_by_prefix(img_dir)
        abnormal_count, normal_count = count_images_with_annotations(label_dir)
        original_count, augmented_count = count_original_vs_augmented(img_dir)

        total_images = thin_count + thick_count

        metrics['splits'][split] = {
            'total_images': total_images,
            'thin_slices': thin_count,
            'thick_slices': thick_count,
            'abnormal_images': abnormal_count,
            'normal_images': normal_count,
            'original_images': original_count,
            'augmented_images': augmented_count
        }

        # Get resolution from first split that has images
        if total_images > 0 and 'resolution' not in metrics:
            resolution = get_image_resolution(img_dir)
            if resolution:
                metrics['resolution'] = resolution

    return metrics

def print_metrics(metrics, dataset_name):
    """Print metrics in the requested format."""

    print("\n" + "="*80)
    print(f"DATASET ANALYSIS: {dataset_name}")
    print("="*80)

    print(f"\nDataset Path: {metrics['dataset_path']}")
    print(f"Number of Classes: {metrics['num_classes']}")
    print(f"Classes: {', '.join(metrics['classes'])}")

    # Calculate totals
    total_original_images = sum(s['original_images'] for s in metrics['splits'].values())
    total_augmented_images = sum(s['augmented_images'] for s in metrics['splits'].values())
    total_images = sum(s['total_images'] for s in metrics['splits'].values())

    train_abnormal = metrics['splits']['train']['abnormal_images']
    train_normal = metrics['splits']['train']['normal_images']

    total_thin = sum(s['thin_slices'] for s in metrics['splits'].values())
    total_thick = sum(s['thick_slices'] for s in metrics['splits'].values())

    print("\n" + "="*80)
    print("SUMMARY METRICS")
    print("="*80)

    print(f"\nOriginal Images (abnormal):     {train_abnormal}")
    print(f"Original Images (normal):       {train_normal}")
    print(f"Normal Cases:                   {train_normal}")
    print(f"Abnormal Cases:                 {train_abnormal}")
    print(f"Augmented Images (abnormal):    {total_augmented_images}")
    print(f"Augmented Images (normal):      0")  # Typically only abnormal are augmented
    print(f"Total Images:                   {total_images}")

    if 'resolution' in metrics:
        print(f"Image Resolution:               {metrics['resolution'][0]}x{metrics['resolution'][1]}")

    print(f"\nTraining Images:                {metrics['splits']['train']['total_images']}")
    print(f"Validation Images:              {metrics['splits']['valid']['total_images']}")
    print(f"Testing Images:                 {metrics['splits']['test']['total_images']}")

    print(f"\nThick Slice Images (5mm):       {total_thick}")
    print(f"Thin Slice Images (0.625mm):    {total_thin}")

    # Detailed breakdown per split
    print("\n" + "="*80)
    print("DETAILED BREAKDOWN BY SPLIT")
    print("="*80)

    for split in ['train', 'valid', 'test']:
        s = metrics['splits'][split]
        print(f"\n{split.upper()}:")
        print(f"  Total Images:       {s['total_images']}")
        print(f"  Original:           {s['original_images']}")
        print(f"  Augmented:          {s['augmented_images']}")
        print(f"  Thin Slices:        {s['thin_slices']}")
        print(f"  Thick Slices:       {s['thick_slices']}")
        print(f"  Abnormal (labeled): {s['abnormal_images']}")
        print(f"  Normal (no labels): {s['normal_images']}")

    print("\n" + "="*80)

def main():
    """Main function to analyze both original and augmented datasets."""

    # Paths to datasets - check archived location first
    original_path = "/Users/yugambahl/Desktop/brain_ct/data/training_datasets/archived/current_20251105/UAT_CT_BRAIN_HEMORRHAGE.v2i.yolov8_combined/data.yaml"
    augmented_path = "/Users/yugambahl/Desktop/brain_ct/data/training_datasets/archived/current_20251105/UAT_CT_BRAIN_HEMORRHAGE.v2i.yolov8_combined_augmented/data.yaml"

    # If not found, check processed/roboflow_versions
    if not os.path.exists(original_path):
        original_path = "/Users/yugambahl/Desktop/brain_ct/data/processed/roboflow_versions/UAT_CT_BRAIN_HEMORRHAGE.v2i.yolov8/data.yaml"

    # Check which datasets exist
    original_exists = os.path.exists(original_path)
    augmented_exists = os.path.exists(augmented_path)

    if not original_exists and not augmented_exists:
        print("Error: No datasets found!")
        print(f"Checked:\n  - {original_path}\n  - {augmented_path}")
        return

    # Analyze original dataset
    if original_exists:
        print("\n" + "#"*80)
        print("# ORIGINAL DATASET")
        print("#"*80)
        metrics = analyze_dataset(original_path)
        if metrics:
            print_metrics(metrics, "Original Dataset")

    # Analyze augmented dataset
    if augmented_exists:
        print("\n" + "#"*80)
        print("# AUGMENTED DATASET")
        print("#"*80)
        metrics = analyze_dataset(augmented_path)
        if metrics:
            print_metrics(metrics, "Augmented Dataset")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
