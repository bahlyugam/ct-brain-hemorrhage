#!/usr/bin/env python3
"""
Comprehensive analysis comparing V2 and V3 datasets with hemorrhage type breakdown.
"""

import json
import os
from pathlib import Path
from collections import defaultdict

def analyze_v2_dataset():
    """Analyze V2 YOLO dataset."""
    v2_path = "/Users/yugambahl/Desktop/brain_ct/data/training_datasets/archived/current_20251105/UAT_CT_BRAIN_HEMORRHAGE.v2i.yolov8_combined"

    stats = {
        'total_images': 0,
        'normal_images': 0,
        'class_image_counts': defaultdict(int),  # Number of images per class
        'class_instance_counts': defaultdict(int)  # Number of instances per class
    }

    # Class names
    class_names = {
        0: 'EDH',
        1: 'HC',
        2: 'IPH',
        3: 'IVH',
        4: 'SAH',
        5: 'SDH'
    }

    # Analyze all splits
    for split in ['train', 'valid', 'test']:
        images_dir = os.path.join(v2_path, split, 'images')
        labels_dir = os.path.join(v2_path, split, 'labels')

        if not os.path.exists(images_dir):
            continue

        # Get all images
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(Path(images_dir).glob(ext))

        stats['total_images'] += len(image_files)

        # Analyze each image
        for img_file in image_files:
            label_file = Path(labels_dir) / (img_file.stem + '.txt')

            if not label_file.exists() or label_file.stat().st_size == 0:
                stats['normal_images'] += 1
            else:
                # Track which classes appear in this image
                classes_in_image = set()

                with open(label_file, 'r') as f:
                    for line in f:
                        class_id = int(line.split()[0])
                        stats['class_instance_counts'][class_id] += 1
                        classes_in_image.add(class_id)

                # Count image once per class it contains
                for class_id in classes_in_image:
                    stats['class_image_counts'][class_id] += 1

    return stats, class_names

def analyze_v3_coco():
    """Analyze V3 COCO format annotations."""
    coco_file = "/Users/yugambahl/Desktop/brain_ct/data/training_datasets/roboflow_downloads/UAT_CT_BRAIN_HEMORRHAGE_V3.v3i.coco/train/_annotations.coco.json"

    with open(coco_file, 'r') as f:
        coco_data = json.load(f)

    stats = {
        'total_images': len(coco_data['images']),
        'normal_images': 0,
        'class_image_counts': defaultdict(int),
        'class_instance_counts': defaultdict(int)
    }

    # Create category mapping
    category_map = {}
    class_names = {}
    for cat in coco_data['categories']:
        category_map[cat['id']] = cat['name']
        class_names[cat['id']] = cat['name']

    # Track which images have annotations
    images_with_annotations = set()
    image_classes = defaultdict(set)

    # Count annotations
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        category_id = ann['category_id']

        images_with_annotations.add(image_id)
        image_classes[image_id].add(category_id)
        stats['class_instance_counts'][category_id] += 1

    # Count images per class
    for image_id, classes in image_classes.items():
        for class_id in classes:
            stats['class_image_counts'][class_id] += 1

    # Count normal images (no annotations)
    stats['normal_images'] = stats['total_images'] - len(images_with_annotations)

    return stats, class_names, category_map

def print_comparison_table(v2_stats, v2_classes, v3_stats, v3_classes, v3_category_map):
    """Print comprehensive comparison table."""

    print("\n" + "="*100)
    print("COMPREHENSIVE DATASET COMPARISON: V2 vs V3")
    print("="*100)

    # Map V3 COCO category names to standard class names
    v3_to_standard = {}
    for cat_id, cat_name in v3_category_map.items():
        # Normalize category names
        name_upper = cat_name.upper()
        if 'EDH' in name_upper or 'EPIDURAL' in name_upper:
            v3_to_standard[cat_id] = 'EDH'
        elif 'HC' in name_upper or 'CONTUSION' in name_upper:
            v3_to_standard[cat_id] = 'HC'
        elif 'IPH' in name_upper or 'INTRAPARENCHYMAL' in name_upper:
            v3_to_standard[cat_id] = 'IPH'
        elif 'IVH' in name_upper or 'INTRAVENTRICULAR' in name_upper:
            v3_to_standard[cat_id] = 'IVH'
        elif 'SAH' in name_upper or 'SUBARACHNOID' in name_upper:
            v3_to_standard[cat_id] = 'SAH'
        elif 'SDH' in name_upper or 'SUBDURAL' in name_upper:
            v3_to_standard[cat_id] = 'SDH'
        else:
            v3_to_standard[cat_id] = cat_name

    # Consolidate V3 stats by standard class names
    v3_consolidated = {
        'class_image_counts': defaultdict(int),
        'class_instance_counts': defaultdict(int)
    }

    for cat_id, count in v3_stats['class_image_counts'].items():
        std_name = v3_to_standard.get(cat_id, v3_category_map[cat_id])
        v3_consolidated['class_image_counts'][std_name] += count

    for cat_id, count in v3_stats['class_instance_counts'].items():
        std_name = v3_to_standard.get(cat_id, v3_category_map[cat_id])
        v3_consolidated['class_instance_counts'][std_name] += count

    # Print table
    print("\n┌─────────────────────────────────────────────────────────────────────────────────────────────┐")
    print("│                          IMAGES PER HEMORRHAGE TYPE                                         │")
    print("└─────────────────────────────────────────────────────────────────────────────────────────────┘")
    print()

    header = f"{'Hemorrhage Type':<25} {'V2 Images':<15} {'V3 Images':<15} {'Total (V2+V3)':<15} {'Growth':<15}"
    print(header)
    print("─" * len(header))

    # Standard class order
    standard_classes = ['EDH', 'HC', 'IPH', 'IVH', 'SAH', 'SDH']

    total_v2_hemorrhage = 0
    total_v3_hemorrhage = 0

    for class_name in standard_classes:
        # Find V2 count
        v2_count = 0
        for class_id, name in v2_classes.items():
            if name == class_name:
                v2_count = v2_stats['class_image_counts'].get(class_id, 0)
                break

        # Get V3 count
        v3_count = v3_consolidated['class_image_counts'].get(class_name, 0)

        total = v2_count + v3_count
        growth = f"+{v3_count}" if v3_count > 0 else "0"

        total_v2_hemorrhage += v2_count
        total_v3_hemorrhage += v3_count

        print(f"{class_name:<25} {v2_count:<15,} {v3_count:<15,} {total:<15,} {growth:<15}")

    print("─" * len(header))

    # Hemorrhage subtotal
    hemorrhage_total = total_v2_hemorrhage + total_v3_hemorrhage
    hemorrhage_growth = f"+{total_v3_hemorrhage}"
    print(f"{'Subtotal (Hemorrhage)':<25} {total_v2_hemorrhage:<15,} {total_v3_hemorrhage:<15,} {hemorrhage_total:<15,} {hemorrhage_growth:<15}")

    # Normal images
    v2_normal = v2_stats['normal_images']
    v3_normal = v3_stats['normal_images']
    total_normal = v2_normal + v3_normal
    normal_growth = f"+{v3_normal}"
    print(f"{'Normal (No Hemorrhage)':<25} {v2_normal:<15,} {v3_normal:<15,} {total_normal:<15,} {normal_growth:<15}")

    print("─" * len(header))

    # Total
    v2_total = v2_stats['total_images']
    v3_total = v3_stats['total_images']
    grand_total = v2_total + v3_total
    total_growth = f"+{v3_total} (+{v3_total/v2_total*100:.1f}%)"
    print(f"{'TOTAL IMAGES':<25} {v2_total:<15,} {v3_total:<15,} {grand_total:<15,} {total_growth:<15}")

    # Print instance counts
    print("\n\n┌─────────────────────────────────────────────────────────────────────────────────────────────┐")
    print("│                     ANNOTATION INSTANCES PER HEMORRHAGE TYPE                                │")
    print("│                    (Total bounding boxes, not unique images)                                │")
    print("└─────────────────────────────────────────────────────────────────────────────────────────────┘")
    print()

    print(header.replace('Images', 'Instances'))
    print("─" * len(header))

    total_v2_instances = 0
    total_v3_instances = 0

    for class_name in standard_classes:
        # Find V2 count
        v2_count = 0
        for class_id, name in v2_classes.items():
            if name == class_name:
                v2_count = v2_stats['class_instance_counts'].get(class_id, 0)
                break

        # Get V3 count
        v3_count = v3_consolidated['class_instance_counts'].get(class_name, 0)

        total = v2_count + v3_count
        growth = f"+{v3_count}" if v3_count > 0 else "0"

        total_v2_instances += v2_count
        total_v3_instances += v3_count

        print(f"{class_name:<25} {v2_count:<15,} {v3_count:<15,} {total:<15,} {growth:<15}")

    print("─" * len(header))

    total_instances = total_v2_instances + total_v3_instances
    instances_growth = f"+{total_v3_instances} (+{total_v3_instances/total_v2_instances*100:.1f}%)"
    print(f"{'TOTAL INSTANCES':<25} {total_v2_instances:<15,} {total_v3_instances:<15,} {total_instances:<15,} {instances_growth:<15}")

    # Print percentages
    print("\n\n┌─────────────────────────────────────────────────────────────────────────────────────────────┐")
    print("│                              DATASET COMPOSITION                                            │")
    print("└─────────────────────────────────────────────────────────────────────────────────────────────┘")
    print()

    print(f"{'Category':<25} {'V2 %':<15} {'V3 %':<15} {'Combined %':<15}")
    print("─" * 70)

    v2_hem_pct = (total_v2_hemorrhage / v2_total * 100) if v2_total > 0 else 0
    v3_hem_pct = (total_v3_hemorrhage / v3_total * 100) if v3_total > 0 else 0
    combined_hem_pct = (hemorrhage_total / grand_total * 100) if grand_total > 0 else 0

    print(f"{'Hemorrhage Images':<25} {v2_hem_pct:<15.1f} {v3_hem_pct:<15.1f} {combined_hem_pct:<15.1f}")

    v2_norm_pct = (v2_normal / v2_total * 100) if v2_total > 0 else 0
    v3_norm_pct = (v3_normal / v3_total * 100) if v3_total > 0 else 0
    combined_norm_pct = (total_normal / grand_total * 100) if grand_total > 0 else 0

    print(f"{'Normal Images':<25} {v2_norm_pct:<15.1f} {v3_norm_pct:<15.1f} {combined_norm_pct:<15.1f}")

    print("="*100)

def analyze_v4_count():
    """Count total images in V4."""
    v4_path = "/Users/yugambahl/Desktop/brain_ct/data/training_datasets/v4/negative_feedback/png"

    v4_count = 0
    if os.path.exists(v4_path):
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            v4_count += len(list(Path(v4_path).glob(ext)))

    return v4_count

def main():
    # Analyze datasets
    print("\nAnalyzing V2 dataset...")
    v2_stats, v2_classes = analyze_v2_dataset()

    print("Analyzing V3 COCO annotations...")
    v3_stats, v3_classes, v3_category_map = analyze_v3_coco()

    print("Counting V4 images...")
    v4_count = analyze_v4_count()

    # Print comparison
    print_comparison_table(v2_stats, v2_classes, v3_stats, v3_classes, v3_category_map)

    # Print V4 info
    print("\n\n┌─────────────────────────────────────────────────────────────────────────────────────────────┐")
    print("│                              V4 DATASET (UNANNOTATED)                                      │")
    print("└─────────────────────────────────────────────────────────────────────────────────────────────┘")
    print()
    print(f"Location: data/training_datasets/v4/negative_feedback/png")
    print(f"Total Images: {v4_count:,}")
    print(f"Status: Unannotated (negative feedback images)")
    print()
    print("="*100)

    # Print summary
    v2_total = v2_stats['total_images']
    v3_total = v3_stats['total_images']
    combined_v3 = v2_total + v3_total
    full_v4 = combined_v3 + v4_count

    print("\n\n┌─────────────────────────────────────────────────────────────────────────────────────────────┐")
    print("│                              VERSION PROGRESSION SUMMARY                                    │")
    print("└─────────────────────────────────────────────────────────────────────────────────────────────┘")
    print()
    print(f"V2:                       {v2_total:>8,} images")
    print(f"V3 (V2 + new download):   {combined_v3:>8,} images  (+{v3_total:,} = +{v3_total/v2_total*100:.1f}%)")
    print(f"V4 (V3 + neg feedback):   {full_v4:>8,} images  (+{v4_count:,} = +{v4_count/combined_v3*100:.1f}%)")
    print()
    print(f"Total Growth (V2 → V4):   +{full_v4-v2_total:,} images (+{(full_v4-v2_total)/v2_total*100:.1f}%)")
    print()
    print("="*100)

if __name__ == '__main__':
    main()
