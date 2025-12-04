"""
Analyze class distribution in the dataset to determine:
1. Total dataset size
2. Class imbalance ratios
3. Optimal YOLO model recommendation
4. Class weight calculation for focal loss
"""

import os
import yaml
from collections import defaultdict
from pathlib import Path


def analyze_class_distribution(data_yaml_path):
    """
    Analyze class distribution across all splits.

    Returns:
        Dictionary with class statistics and recommendations
    """
    with open(data_yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    base_path = config.get('path', '')
    class_names = config.get('names', [])
    num_classes = config.get('nc', 0)

    print("\n" + "="*80)
    print("DATASET AND CLASS DISTRIBUTION ANALYSIS")
    print("="*80)

    # Initialize counters
    splits = ['train', 'valid', 'test']
    split_stats = {}

    # Per-class instance counts (bounding boxes)
    class_instances = defaultdict(int)
    # Per-class image counts (images containing class)
    class_images = defaultdict(int)
    # Images with no hemorrhage
    no_hemorrhage_images = 0
    total_images = 0
    total_instances = 0

    for split in splits:
        split_dir = os.path.join(base_path, split)
        img_dir = os.path.join(split_dir, 'images')
        label_dir = os.path.join(split_dir, 'labels')

        if not os.path.exists(img_dir):
            continue

        split_class_instances = defaultdict(int)
        split_class_images = defaultdict(int)
        split_no_hemorrhage = 0
        split_total_images = 0
        split_total_instances = 0

        # Process each image
        for img_file in os.listdir(img_dir):
            if not img_file.endswith(('.jpg', '.jpeg', '.png')):
                continue

            split_total_images += 1

            # Get corresponding label file
            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(label_dir, label_file)

            if not os.path.exists(label_path):
                split_no_hemorrhage += 1
                continue

            # Parse labels
            with open(label_path, 'r') as f:
                lines = f.readlines()

            if len(lines) == 0:
                split_no_hemorrhage += 1
                continue

            # Track classes in this image
            classes_in_image = set()

            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    split_class_instances[class_id] += 1
                    classes_in_image.add(class_id)

            # Count images per class
            for class_id in classes_in_image:
                split_class_images[class_id] += 1

            split_total_instances += len(lines)

        split_stats[split] = {
            'total_images': split_total_images,
            'no_hemorrhage_images': split_no_hemorrhage,
            'hemorrhage_images': split_total_images - split_no_hemorrhage,
            'total_instances': split_total_instances,
            'class_instances': dict(split_class_instances),
            'class_images': dict(split_class_images),
        }

        # Accumulate global stats
        total_images += split_total_images
        no_hemorrhage_images += split_no_hemorrhage
        total_instances += split_total_instances

        for class_id, count in split_class_instances.items():
            class_instances[class_id] += count

        for class_id, count in split_class_images.items():
            class_images[class_id] += count

    # Print detailed statistics
    print(f"\n{'='*80}")
    print("OVERALL DATASET STATISTICS")
    print(f"{'='*80}")
    print(f"Total images: {total_images:,}")
    print(f"  - With hemorrhage: {total_images - no_hemorrhage_images:,} ({(total_images - no_hemorrhage_images)/total_images*100:.1f}%)")
    print(f"  - No hemorrhage: {no_hemorrhage_images:,} ({no_hemorrhage_images/total_images*100:.1f}%)")
    print(f"Total bounding box instances: {total_instances:,}")
    print(f"Average instances per image: {total_instances/total_images:.2f}")

    # Per-split statistics
    for split in splits:
        if split not in split_stats:
            continue

        stats = split_stats[split]
        print(f"\n{split.upper()} SET:")
        print(f"  Total images: {stats['total_images']:,}")
        print(f"  Hemorrhage images: {stats['hemorrhage_images']:,}")
        print(f"  No hemorrhage images: {stats['no_hemorrhage_images']:,}")
        print(f"  Total instances: {stats['total_instances']:,}")

    # Class distribution
    print(f"\n{'='*80}")
    print("PER-CLASS DISTRIBUTION")
    print(f"{'='*80}")
    print(f"{'Class':<30} {'Instances':<12} {'Images':<12} {'% of Total':<12}")
    print(f"{'-'*80}")

    for class_id in range(num_classes):
        class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
        instances = class_instances.get(class_id, 0)
        images = class_images.get(class_id, 0)
        percentage = (instances / total_instances * 100) if total_instances > 0 else 0

        print(f"{class_name:<30} {instances:<12} {images:<12} {percentage:<12.2f}%")

    # Calculate class imbalance ratios
    print(f"\n{'='*80}")
    print("CLASS IMBALANCE ANALYSIS")
    print(f"{'='*80}")

    if class_instances:
        max_instances = max(class_instances.values())
        min_instances = min(class_instances.values())

        print(f"Most common class: {max_instances:,} instances")
        print(f"Least common class: {min_instances:,} instances")
        print(f"Imbalance ratio: {max_instances/min_instances:.2f}:1")

        print(f"\nPer-class imbalance ratios:")
        for class_id in range(num_classes):
            class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
            instances = class_instances.get(class_id, 0)
            if instances > 0:
                ratio = max_instances / instances
                print(f"  {class_name:<30} {ratio:.2f}:1 (needs {ratio:.2f}x weight)")

    # Calculate recommended class weights
    class_weights = calculate_class_weights(class_instances, num_classes)

    print(f"\n{'='*80}")
    print("RECOMMENDED CLASS WEIGHTS (Inverse Frequency)")
    print(f"{'='*80}")
    for class_id in range(num_classes):
        class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
        weight = class_weights.get(class_id, 1.0)
        print(f"  {class_name:<30} {weight:.4f}")

    # Model recommendation
    recommend_model(total_images, total_instances, num_classes)

    return {
        'total_images': total_images,
        'total_instances': total_instances,
        'no_hemorrhage_images': no_hemorrhage_images,
        'class_instances': dict(class_instances),
        'class_images': dict(class_images),
        'class_weights': class_weights,
        'split_stats': split_stats,
        'num_classes': num_classes,
        'class_names': class_names,
    }


def calculate_class_weights(class_instances, num_classes, method='inverse_freq'):
    """
    Calculate class weights for handling imbalance.

    Methods:
    - 'inverse_freq': weight = total / (num_classes * class_freq)
    - 'effective_samples': weight based on effective number of samples
    """
    if not class_instances:
        return {i: 1.0 for i in range(num_classes)}

    total_instances = sum(class_instances.values())

    if method == 'inverse_freq':
        # Standard inverse frequency weighting
        weights = {}
        for class_id in range(num_classes):
            instances = class_instances.get(class_id, 1)  # Avoid division by zero
            weights[class_id] = total_instances / (num_classes * instances)

        # Normalize weights so average is 1.0
        avg_weight = sum(weights.values()) / len(weights)
        weights = {k: v / avg_weight for k, v in weights.items()}

        return weights

    elif method == 'effective_samples':
        # Effective number of samples weighting (better for extreme imbalance)
        # From "Class-Balanced Loss Based on Effective Number of Samples"
        beta = 0.9999  # Hyperparameter (0.9999 for large datasets)

        weights = {}
        for class_id in range(num_classes):
            instances = class_instances.get(class_id, 1)
            effective_num = (1 - beta ** instances) / (1 - beta)
            weights[class_id] = 1.0 / effective_num

        # Normalize
        avg_weight = sum(weights.values()) / len(weights)
        weights = {k: v / avg_weight for k, v in weights.items()}

        return weights

    return {i: 1.0 for i in range(num_classes)}


def recommend_model(total_images, total_instances, num_classes):
    """
    Recommend optimal YOLO model based on dataset size.
    """
    print(f"\n{'='*80}")
    print("YOLO MODEL RECOMMENDATION")
    print(f"{'='*80}")

    print(f"\nDataset characteristics:")
    print(f"  Total images: {total_images:,}")
    print(f"  Total instances: {total_instances:,}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Instances per image: {total_instances/total_images:.2f}")

    # Model comparison
    models = {
        'yolov8n': {
            'params': '3.2M',
            'flops': '8.7G',
            'speed_cpu': '80.4ms',
            'speed_gpu': '0.99ms',
            'map50': '52.3%',
            'min_images': 1000,
            'recommended_images': '1K-5K',
            'use_case': 'Edge deployment, real-time inference, limited compute'
        },
        'yolov8s': {
            'params': '11.2M',
            'flops': '28.6G',
            'speed_cpu': '128.4ms',
            'speed_gpu': '1.20ms',
            'map50': '61.0%',
            'min_images': 3000,
            'recommended_images': '3K-10K',
            'use_case': 'Balanced speed/accuracy, general purpose'
        },
        'yolov8m': {
            'params': '25.9M',
            'flops': '78.9G',
            'speed_cpu': '234.7ms',
            'speed_gpu': '1.83ms',
            'map50': '67.2%',
            'min_images': 5000,
            'recommended_images': '5K-20K',
            'use_case': 'Medical imaging, moderate dataset size'
        },
        'yolov8l': {
            'params': '43.7M',
            'flops': '165.2G',
            'speed_cpu': '375.2ms',
            'speed_gpu': '2.39ms',
            'map50': '69.8%',
            'min_images': 8000,
            'recommended_images': '8K-30K',
            'use_case': 'Large datasets, clinical accuracy priority'
        },
        'yolov8x': {
            'params': '68.2M',
            'flops': '257.8G',
            'speed_cpu': '479.1ms',
            'speed_gpu': '3.53ms',
            'map50': '70.6%',
            'min_images': 10000,
            'recommended_images': '10K+',
            'use_case': 'Maximum accuracy, research, offline processing'
        },
    }

    print(f"\n{'Model':<12} {'Params':<10} {'FLOPs':<10} {'mAP50':<10} {'Recommended For':<30}")
    print(f"{'-'*80}")

    for model_name, specs in models.items():
        print(f"{model_name:<12} {specs['params']:<10} {specs['flops']:<10} {specs['map50']:<10} {specs['recommended_images']:<30}")

    # Recommendation logic
    print(f"\n{'='*80}")
    print("RECOMMENDATION FOR YOUR DATASET")
    print(f"{'='*80}")

    if total_images < 3000:
        recommended = 'yolov8n'
        reason = "Small dataset - use lightweight model to avoid overfitting"
    elif total_images < 5000:
        recommended = 'yolov8s'
        reason = "Small-medium dataset - balanced model"
    elif total_images < 8000:
        recommended = 'yolov8m'
        reason = "Medium dataset - good capacity for medical imaging"
    elif total_images < 12000:
        recommended = 'yolov8l'
        reason = "Large dataset - high capacity for complex patterns"
    else:
        recommended = 'yolov8x'
        reason = "Very large dataset - maximum accuracy"

    # Adjust for medical imaging
    if num_classes >= 5 and total_images >= 5000:
        if recommended in ['yolov8s', 'yolov8m']:
            recommended = 'yolov8m'
            reason += " + medical imaging benefits from larger models"
        elif recommended == 'yolov8n':
            recommended = 'yolov8s'
            reason += " + medical imaging needs more capacity"

    print(f"\n🎯 RECOMMENDED MODEL: {recommended.upper()}")
    print(f"   Reason: {reason}")
    print(f"\n   Your dataset ({total_images:,} images) is well-suited for {recommended}")

    # Alternative recommendations
    print(f"\n📊 ALTERNATIVES:")

    if total_images >= 8000:
        print(f"   • {recommended} (RECOMMENDED) - Best balance for your dataset")
        if recommended != 'yolov8l':
            print(f"   • yolov8l - Higher accuracy, slower inference")
        if recommended != 'yolov8x':
            print(f"   • yolov8x - Maximum accuracy, research/offline use")
    else:
        print(f"   • {recommended} (RECOMMENDED) - Optimal for your dataset size")

        # Suggest data augmentation
        if total_images < 5000:
            print(f"\n   ⚠️  Consider:")
            print(f"      - Your dataset is on the smaller side for medical imaging")
            print(f"      - Heavy augmentation is crucial (already implemented ✓)")
            print(f"      - May benefit from transfer learning or pretrained weights")

    # Medical imaging specific advice
    print(f"\n💡 MEDICAL IMAGING CONSIDERATIONS:")
    print(f"   • Hemorrhage detection requires high precision")
    print(f"   • False negatives are critical in medical diagnosis")
    print(f"   • Recommend: {recommended} or larger for production use")
    print(f"   • Test-time augmentation can boost performance +1-2%")

    return recommended


if __name__ == "__main__":
    # Analyze the balanced dataset
    data_yaml = "/Users/yugambahl/Desktop/brain_ct/data/UAT_CT_BRAIN_HEMORRHAGE.v2i.yolov8_combined/data.yaml"

    if not os.path.exists(data_yaml):
        print(f"Error: {data_yaml} not found")
        print("Please update the path to your data.yaml file")
    else:
        results = analyze_class_distribution(data_yaml)

        # Save results
        import json
        output_file = "/Users/yugambahl/Desktop/brain_ct/dataset_analysis.json"

        # Convert to JSON-serializable format
        json_results = {
            'total_images': results['total_images'],
            'total_instances': results['total_instances'],
            'no_hemorrhage_images': results['no_hemorrhage_images'],
            'class_instances': results['class_instances'],
            'class_images': results['class_images'],
            'class_weights': results['class_weights'],
            'num_classes': results['num_classes'],
            'class_names': results['class_names'],
        }

        with open(output_file, 'w') as f:
            json.dump(json_results, f, indent=2)

        print(f"\n{'='*80}")
        print(f"✓ Analysis saved to: {output_file}")
        print(f"{'='*80}\n")
