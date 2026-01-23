#!/usr/bin/env python3
"""
Analyze and compare V2 and V3 datasets for hemorrhage and normal images.
"""

import os
from pathlib import Path
from collections import defaultdict

def analyze_yolo_labels(label_dir):
    """Analyze YOLO format labels to count hemorrhage types and normal images."""
    stats = {
        'total_images': 0,
        'hemorrhage_images': 0,
        'normal_images': 0,
        'class_counts': defaultdict(int),
        'images_per_class': defaultdict(set)
    }

    # Get corresponding image directory
    images_dir = label_dir.replace('/labels', '/images')

    # Get all images
    image_files = []
    if os.path.exists(images_dir):
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(Path(images_dir).glob(ext))

    stats['total_images'] = len(image_files)

    # Analyze labels
    for img_file in image_files:
        label_file = Path(label_dir) / (img_file.stem + '.txt')

        if not label_file.exists() or label_file.stat().st_size == 0:
            # No label file or empty = normal image
            stats['normal_images'] += 1
        else:
            # Has labels = hemorrhage image
            stats['hemorrhage_images'] += 1

            # Count classes in this image
            with open(label_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    class_id = int(line.split()[0])
                    stats['class_counts'][class_id] += 1
                    stats['images_per_class'][class_id].add(img_file.name)

    return stats

def analyze_dataset_split(base_path):
    """Analyze train/valid/test splits."""
    splits = {}

    for split in ['train', 'valid', 'test']:
        label_dir = os.path.join(base_path, split, 'labels')
        if os.path.exists(label_dir):
            splits[split] = analyze_yolo_labels(label_dir)

    return splits

def print_stats(name, splits):
    """Print statistics for a dataset."""
    print(f"\n{'='*80}")
    print(f"{name}")
    print(f"{'='*80}")

    # Class names mapping (based on documentation)
    class_names = {
        0: 'EDH (Epidural)',
        1: 'HC (Contusion)',
        2: 'IPH (Intraparenchymal)',
        3: 'IVH (Intraventricular)',
        4: 'SAH (Subarachnoid)',
        5: 'SDH (Subdural)'
    }

    total_images = 0
    total_hemorrhage = 0
    total_normal = 0
    total_class_counts = defaultdict(int)
    total_images_per_class = defaultdict(set)

    for split_name, stats in splits.items():
        print(f"\n{split_name.upper()}:")
        print(f"  Total images: {stats['total_images']:,}")
        print(f"  Hemorrhage images: {stats['hemorrhage_images']:,} ({stats['hemorrhage_images']/stats['total_images']*100:.1f}%)")
        print(f"  Normal images: {stats['normal_images']:,} ({stats['normal_images']/stats['total_images']*100:.1f}%)")

        total_images += stats['total_images']
        total_hemorrhage += stats['hemorrhage_images']
        total_normal += stats['normal_images']

        for class_id, count in stats['class_counts'].items():
            total_class_counts[class_id] += count
            total_images_per_class[class_id].update(stats['images_per_class'][class_id])

    print(f"\nOVERALL SUMMARY:")
    print(f"  Total images: {total_images:,}")
    print(f"  Hemorrhage images: {total_hemorrhage:,} ({total_hemorrhage/total_images*100:.1f}%)")
    print(f"  Normal images: {total_normal:,} ({total_normal/total_images*100:.1f}%)")

    print(f"\nHEMORRHAGE TYPE BREAKDOWN:")
    print(f"  {'Class':<25} {'Instances':<12} {'Images':<12}")
    print(f"  {'-'*50}")

    for class_id in sorted(total_class_counts.keys()):
        class_name = class_names.get(class_id, f'Class {class_id}')
        instances = total_class_counts[class_id]
        images = len(total_images_per_class[class_id])
        print(f"  {class_name:<25} {instances:<12,} {images:<12,}")

    total_instances = sum(total_class_counts.values())
    total_unique_hemorrhage_images = len(set().union(*[imgs for imgs in total_images_per_class.values()]))
    print(f"  {'-'*50}")
    print(f"  {'TOTAL':<25} {total_instances:<12,} {total_unique_hemorrhage_images:<12,}")

def main():
    print("\n" + "="*80)
    print("DATASET COMPARISON: V2 vs V3")
    print("="*80)

    # V2 Dataset
    v2_path = "/Users/yugambahl/Desktop/brain_ct/data/training_datasets/archived/current_20251105/UAT_CT_BRAIN_HEMORRHAGE.v2i.yolov8_combined"
    print(f"\nAnalyzing V2 dataset: {v2_path}")
    v2_splits = analyze_dataset_split(v2_path)
    print_stats("V2 DATASET", v2_splits)

    # V3 Additional Download (just train folder with images only)
    v3_download_path = "/Users/yugambahl/Desktop/brain_ct/data/training_datasets/roboflow_downloads/UAT_CT_BRAIN_HEMORRHAGE_V3.v3i.coco/train"
    print(f"\n\n{'='*80}")
    print("V3 ADDITIONAL DOWNLOAD")
    print(f"{'='*80}")
    print(f"\nPath: {v3_download_path}")

    # Count V3 images
    v3_image_count = 0
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        v3_image_count += len(list(Path(v3_download_path).glob(ext)))

    print(f"\nTotal images in V3 download: {v3_image_count:,}")
    print("Note: V3 download contains only images without YOLO labels")
    print("      (needs to be annotated or has COCO annotations)")

    # Combined V3 stats
    print(f"\n\n{'='*80}")
    print("COMBINED V3 DATASET (V2 + V3 Download)")
    print(f"{'='*80}")

    v2_total = sum(s['total_images'] for s in v2_splits.values())
    v3_total = v2_total + v3_image_count

    print(f"\nV2 images: {v2_total:,}")
    print(f"V3 additional download: {v3_image_count:,}")
    print(f"Total V3: {v3_total:,}")
    print(f"Growth: +{v3_image_count:,} (+{v3_image_count/v2_total*100:.1f}%)")

if __name__ == '__main__':
    main()
