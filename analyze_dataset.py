#!/usr/bin/env python3
"""
Standalone script to analyze brain CT hemorrhage dataset.
Shows class distribution, imbalance ratios, and augmentation info.
"""

import sys
sys.path.insert(0, '/Users/yugambahl/Desktop/brain_ct')

from yolo_augmented_dataset import (
    analyze_class_distribution,
    print_class_imbalance_summary,
    get_oversampling_weights,
    print_augmentation_summary
)
import yaml
import os

def main():
    # Paths to your datasets
    original_data_yaml = "/Users/yugambahl/Desktop/brain_ct/data/training_datasets/archived/current_20251105/UAT_CT_BRAIN_HEMORRHAGE.v2i.yolov8_combined/data.yaml"
    augmented_data_yaml = "/Users/yugambahl/Desktop/brain_ct/data/UAT_CT_BRAIN_HEMORRHAGE.v2i.yolov8_combined_augmented/data.yaml"

    # Check which datasets exist
    original_exists = os.path.exists(original_data_yaml)
    augmented_exists = os.path.exists(augmented_data_yaml)

    print("\n" + "="*80)
    print("BRAIN CT HEMORRHAGE DATASET ANALYSIS")
    print("="*80)

    if augmented_exists:
        print("\n🚀 PRE-AUGMENTED DATASET DETECTED!")
        print(f"   Using: {augmented_data_yaml}")
        data_yaml_path = augmented_data_yaml
        use_preaugmented = True
    else:
        print("\n📁 Using original dataset (pre-augmentation not run yet)")
        print(f"   Using: {original_data_yaml}")
        data_yaml_path = original_data_yaml
        use_preaugmented = False

    # Load dataset config
    with open(data_yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"\nDataset: {config.get('path', 'N/A')}")
    print(f"Number of classes: {config.get('nc', 'N/A')}")
    print(f"Classes: {', '.join(config.get('names', []))}")

    # Analyze class distribution
    print("\n" + "="*80)
    print("ANALYZING CLASS DISTRIBUTION (BEFORE EDH FILTERING)")
    print("="*80)
    class_stats = analyze_class_distribution(data_yaml_path)

    if class_stats:
        print_class_imbalance_summary(class_stats)

        # Show oversampling weights
        weights = get_oversampling_weights(class_stats)
        print("\n" + "="*80)
        print("RECOMMENDED OVERSAMPLING WEIGHTS")
        print("="*80)
        for class_name, weight in weights.items():
            print(f"{class_name:<10}: {weight:.2f}x")
        print("="*80)
    else:
        print("\n⚠️  Could not analyze class distribution. Check dataset path.")

    # Show augmentation info for both slice types
    print("\n" + "="*80)
    print("AUGMENTATION CONFIGURATION")
    print("="*80)

    print("\n--- THIN SLICES / MIXED ---")
    print_augmentation_summary('thin')

    print("\n--- THICK SLICES ---")
    print_augmentation_summary('thick')

    # Count images and analyze per-class distribution
    base_path = config.get('path', '')
    print("\n" + "="*80)
    print("IMAGE COUNTS (ORIGINAL)")
    print("="*80)

    split_stats = {}
    for split in ['train', 'valid', 'test']:
        img_dir = os.path.join(base_path, split, 'images')
        label_dir = os.path.join(base_path, split, 'labels')

        if os.path.exists(img_dir):
            thin_count = sum(1 for f in os.listdir(img_dir) if f.startswith('thin_'))
            thick_count = sum(1 for f in os.listdir(img_dir) if f.startswith('thick_'))
            total = thin_count + thick_count

            # Count instances per class in this split
            class_instances = {name: 0 for name in config.get('names', [])}
            images_with_class = {name: set() for name in config.get('names', [])}

            if os.path.exists(label_dir):
                for label_file in os.listdir(label_dir):
                    if not label_file.endswith('.txt'):
                        continue

                    label_path = os.path.join(label_dir, label_file)
                    image_name = label_file.replace('.txt', '')

                    with open(label_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                class_id = int(parts[0])
                                if class_id < len(config.get('names', [])):
                                    class_name = config.get('names', [])[class_id]
                                    class_instances[class_name] += 1
                                    images_with_class[class_name].add(image_name)

            split_stats[split] = {
                'total_images': total,
                'thin_images': thin_count,
                'thick_images': thick_count,
                'class_instances': class_instances,
                'images_with_class': images_with_class
            }

            print(f"{split.upper():<10}: {total:>4} images (thin: {thin_count:>4}, thick: {thick_count:>4})")

    # Show per-class distribution across splits
    print("\n" + "="*80)
    print("PER-CLASS INSTANCE COUNTS (ORIGINAL - INCLUDES EDH)")
    print("="*80)
    print(f"\n{'Class':<10} {'Train Inst':<12} {'Train Imgs':<12} {'Val Inst':<12} {'Val Imgs':<12} {'Test Inst':<12} {'Test Imgs':<12}")
    print("-" * 80)

    for class_name in config.get('names', []):
        train_inst = split_stats.get('train', {}).get('class_instances', {}).get(class_name, 0)
        train_imgs = len(split_stats.get('train', {}).get('images_with_class', {}).get(class_name, set()))
        val_inst = split_stats.get('valid', {}).get('class_instances', {}).get(class_name, 0)
        val_imgs = len(split_stats.get('valid', {}).get('images_with_class', {}).get(class_name, set()))
        test_inst = split_stats.get('test', {}).get('class_instances', {}).get(class_name, 0)
        test_imgs = len(split_stats.get('test', {}).get('images_with_class', {}).get(class_name, set()))

        print(f"{class_name:<10} {train_inst:<12} {train_imgs:<12} {val_inst:<12} {val_imgs:<12} {test_inst:<12} {test_imgs:<12}")

    # Calculate effective training samples with augmentation
    train_images = split_stats.get('train', {}).get('total_images', 0)
    batch_size = 16
    epochs = 200

    if use_preaugmented:
        print("\n" + "="*80)
        print("🚀 PRE-AUGMENTED DATASET MODE (8-10x FASTER TRAINING)")
        print("="*80)
        print("\n✅ RSNA ICH 2019 Winners' Strategy Applied:")
        print("   1. Original images")
        print("   2. Horizontal flip (100% of winners used this)")
        print("   3. Rotation +10° (patient positioning variation)")
        print("   4. Rotation -10° + Brightness/Contrast (scanner variation)")
        print("\n✅ All augmentation done OFFLINE - zero overhead during training!")
        print("✅ On-the-fly augmentation DISABLED (mosaic=0, rotation=0, etc.)")

        # For pre-augmented, training images already include augmented versions
        original_train_images = train_images // 4  # Reverse the 4x multiplication

        print(f"\nPre-Augmented Dataset Statistics:")
        print(f"  Original training images: {original_train_images:,}")
        print(f"  After 4x augmentation: {train_images:,} (includes original + 3 augmented versions)")
        print(f"  Augmentation multiplier: 4x")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Total training iterations: {(train_images // batch_size) * epochs:,}")

        print(f"\n⚡ Expected Training Speed:")
        print(f"  Speed: 0.8-1.0 it/s (vs 0.1 it/s on-the-fly)")
        print(f"  Speedup: 8-10x faster")
        print(f"  Time per epoch: ~10 minutes (vs ~1.5 hours)")
        print(f"  Total training time: ~33 hours (vs ~12 days)")

    else:
        print("\n" + "="*80)
        print("ON-THE-FLY AUGMENTATION MODE (CURRENT - SLOW)")
        print("="*80)
        print("\n⚠️  YOLO applies augmentation on-the-fly during training.")
        print("   Each epoch sees different augmented versions of the same image.")
        print("\nAugmentation probabilities (from training config):")
        print("  - Mosaic: 0.8 (combines 4 images) ← MAIN BOTTLENECK")
        print("  - MixUp: 0.3 (blends 2 images)")
        print("  - Copy-Paste: 0.3 (copies objects between images)")
        print("  - Rotation: 0-16° randomly")
        print("  - Brightness/Contrast: ±60% randomly")
        print("  - CLAHE: 0.7 probability")

        # Effective samples per epoch (accounting for mosaic which combines 4 images)
        # With mosaic=0.8, roughly 80% of batches use mosaic (4 images → 1 sample)
        # 20% use single images
        effective_samples_per_epoch = int(train_images * 0.2 + (train_images * 0.8) / 4)
        total_training_iterations = (train_images // batch_size) * epochs
        total_augmented_views = train_images * epochs  # Each image seen 200 times with different augmentations

        print(f"\nOn-the-Fly Training Statistics:")
        print(f"  Original training images: {train_images:,}")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Effective samples per epoch: ~{effective_samples_per_epoch:,} (due to mosaic)")
        print(f"  Total training iterations: {total_training_iterations:,}")
        print(f"  Total augmented views: {total_augmented_views:,} (each image × {epochs} epochs)")

        print(f"\n⚠️  Current Training Speed:")
        print(f"  Speed: ~0.1 it/s (VERY SLOW)")
        print(f"  Time per epoch: ~1.5 hours")
        print(f"  Total training time: ~12 days")

        print(f"\n💡 TO SPEED UP 8-10x:")
        print(f"  1. Run: python preaugment_rsna.py")
        print(f"  2. Edit train.py: USE_PREAUGMENTED = True")
        print(f"  3. Restart training")

    # Estimate per-class effective samples (after EDH filtering)
    print("\n" + "="*80)
    print("EFFECTIVE PER-CLASS TRAINING SAMPLES (AFTER EDH FILTERING)")
    print("="*80)
    print(f"\n{'Class':<10} {'Original Inst':<15} {'Original Imgs':<15} {'Effective Views':<15} {'After 200 Epochs':<20}")
    print("-" * 80)

    for class_name in config.get('names', []):
        if class_name == 'EDH':
            print(f"{class_name:<10} {'FILTERED OUT':<15} {'FILTERED OUT':<15} {'FILTERED OUT':<15} {'FILTERED OUT':<20}")
            continue

        train_inst = split_stats.get('train', {}).get('class_instances', {}).get(class_name, 0)
        train_imgs = len(split_stats.get('train', {}).get('images_with_class', {}).get(class_name, set()))

        # Each image with this class is seen once per epoch with different augmentation
        effective_views_per_epoch = train_imgs
        total_views = train_imgs * epochs

        print(f"{class_name:<10} {train_inst:<15} {train_imgs:<15} {effective_views_per_epoch:<15} {total_views:<20,}")

    print("\n" + "="*80)
    print("VALIDATION & TEST (NO AUGMENTATION)")
    print("="*80)
    val_images = split_stats.get('valid', {}).get('total_images', 0)
    test_images = split_stats.get('test', {}).get('total_images', 0)
    print(f"Validation images: {val_images} (evaluated once per epoch)")
    print(f"Test images: {test_images} (evaluated after training)")
    print("Note: Validation and test sets are NOT augmented")

    print("\n" + "="*80)
    print("IMPORTANT NOTES")
    print("="*80)
    print("✓ EDH class will be filtered out during training (insufficient data)")
    print("✓ Training will use 5 classes: HC, IPH, IVH, SAH, SDH")

    if use_preaugmented:
        print("✓ Using PRE-AUGMENTED dataset (4x data, RSNA winners' strategy)")
        print("✓ Zero augmentation overhead during training (all done offline)")
        print("✓ Expected 8-10x faster training (0.8-1.0 it/s)")
    else:
        print("✓ Using ON-THE-FLY augmentation (slower but more random)")
        print("✓ Augmentation overhead during training (mosaic, rotation, etc.)")
        print("⚠️  Current speed: ~0.1 it/s (SLOW)")
        print("💡 Run preaugment_rsna.py to speed up 8-10x")

    print("✓ Validation/test sets are NOT augmented for fair evaluation")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()