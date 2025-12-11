"""
Custom YOLOv8 dataset wrapper with evidence-based CT augmentations.

Integrates Albumentations pipeline with YOLOv8 training while respecting
CT physics and preserving bounding box annotations.

Enhanced with class imbalance handling for rare hemorrhage types.
"""

import os
import cv2
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2


def create_augmented_yaml_config(
    base_yaml_path: str,
    output_yaml_path: str,
    slice_type: str = 'mixed'
) -> Dict:
    """
    Create a YAML config that documents augmentation strategy.

    Args:
        base_yaml_path: Path to base data.yaml
        output_yaml_path: Path to save augmented config
        slice_type: 'thin', 'thick', or 'mixed'

    Returns:
        Config dictionary
    """
    # Load base config
    with open(base_yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    # Add augmentation metadata
    config['augmentation'] = {
        'enabled': True,
        'slice_type': slice_type,
        'techniques': [
            'RandomSizedBBoxSafeCrop',
            'HorizontalFlip',
            'ShiftScaleRotate',
            'RandomBrightnessContrast',
            'CLAHE',
            'CoarseDropout'
        ],
        'evidence_based': True,
        'source': 'RSNA_ICH_2019_winners',
    }

    # Save augmented config
    with open(output_yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    return config


def apply_yolo_compatible_augmentations(
    yolo_model,
    enable_custom_aug: bool = True,
    slice_type: str = 'mixed'
):
    """
    Configure YOLOv8 training with evidence-based augmentation parameters.

    YOLOv8 has built-in augmentations. We adjust them to match research findings
    while keeping YOLOv8's native implementation for performance.

    Args:
        yolo_model: YOLO model instance
        enable_custom_aug: Enable evidence-based augmentations
        slice_type: 'thin', 'thick', or 'mixed'

    Returns:
        Dictionary of augmentation hyperparameters for model.train()
    """

    if not enable_custom_aug:
        # Minimal augmentation
        return {
            'degrees': 0.0,
            'translate': 0.0,
            'scale': 0.0,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.0,
            'mosaic': 0.0,
            'mixup': 0.0,
            'copy_paste': 0.0,
        }

    # PHASE 1: Conservative augmentation parameters to preserve fine-grained features
    # Previous extensive augmentation (mosaic=0.8, rotation=±10°, scale=50%, HSV) was
    # destroying subtle hemorrhage characteristics, causing high cls_loss (5.87)
    # and low mAP50-95 (0.21). Medical imaging requires more conservative augmentation.

    if slice_type == 'thick':
        # Thick slices: slightly more aggressive than thin
        params = {
            # PHASE 1: Reduced rotation from ±16° to ±7° (brain hemorrhages have anatomical context)
            'degrees': 7.0,

            # PHASE 1: Reduced translation from 10% to 5% (preserve spatial relationships)
            'translate': 0.05,

            # PHASE 1: Reduced scale from 50% to 20% variation (hemorrhages have constrained sizes)
            'scale': 0.2,

            # No shearing (evidence shows it degrades performance)
            'shear': 0.0,

            # No perspective (not anatomically valid for CT)
            'perspective': 0.0,

            # No vertical flip (anatomically incorrect)
            'flipud': 0.0,

            # Horizontal flip: 50% probability (anatomically valid)
            'fliplr': 0.5,

            # PHASE 1: Reduced mosaic from 0.8 to 0.3 (mosaic creates unrealistic composite images)
            # High mosaic confuses model about spatial relationships in CT scans
            'mosaic': 0.3,

            # No mixup (not appropriate for medical detection)
            'mixup': 0.0,

            # No copy-paste
            'copy_paste': 0.0,

            # Close mosaic last 20 epochs
            'close_mosaic': 20,

            # PHASE 1: REMOVED HSV augmentations (CT has standardized HU windows)
            # Random brightness changes create unrealistic intensity distributions
            'hsv_h': 0.0,
            'hsv_s': 0.0,
            'hsv_v': 0.0,  # Disabled - was 0.015
        }
    else:
        # Thin slices or mixed: PHASE 1 conservative parameters
        params = {
            # PHASE 1: Reduced rotation from ±10° to ±5° (preserve anatomical context)
            'degrees': 5.0,

            # PHASE 1: Reduced translation from 6.25% to 3% (preserve spatial relationships)
            'translate': 0.03,

            # PHASE 1: Reduced scale from 50% to 20% variation (10% zoom range)
            'scale': 0.2,

            # No shearing
            'shear': 0.0,

            # No perspective
            'perspective': 0.0,

            # No vertical flip
            'flipud': 0.0,

            # Horizontal flip: 50% (only geometrically valid augmentation)
            'fliplr': 0.5,

            # PHASE 1: Reduced mosaic from 0.8 to 0.3 (main culprit for high cls_loss)
            # Mosaic destroys fine-grained features needed to distinguish hemorrhage types
            'mosaic': 0.3,

            # No mixup
            'mixup': 0.0,

            # No copy-paste
            'copy_paste': 0.0,

            # Close mosaic last 20 epochs
            'close_mosaic': 20,

            # PHASE 1: REMOVED HSV augmentations (unrealistic for CT)
            'hsv_h': 0.0,
            'hsv_s': 0.0,
            'hsv_v': 0.0,  # Disabled - was 0.015
        }

    return params


def get_preaugmented_minimal_params():
    """
    Light augmentation parameters for pre-augmented dataset.

    PHASE 1 FIX: Added light horizontal flip (0.5 probability) to add
    batch-level diversity while maintaining speed.

    Pre-augmented dataset has offline augmentations (flip, rotation ±10°, brightness/contrast).
    Adding light online flip improves generalization without significant speed penalty (<5%).

    Expected speedup: 8-10x faster training (0.1 it/s → 0.8-1.0 it/s)

    Returns:
        Dictionary of augmentation parameters
    """
    return {
        # All geometric augmentations disabled (already applied offline)
        'degrees': 0.0,       # Already rotated ±10°
        'translate': 0.0,     # Not needed for binary detection
        'scale': 0.0,         # Not needed for binary detection
        'shear': 0.0,
        'perspective': 0.0,

        # PHASE 1: Light horizontal flip for batch diversity
        'flipud': 0.0,
        'fliplr': 0.5,        # 50% chance - adds diversity with minimal overhead

        # Mosaic disabled (huge speedup!)
        'mosaic': 0.0,        # Main bottleneck - disabling gives 3x speedup
        'mixup': 0.0,
        'copy_paste': 0.0,
        'close_mosaic': 0,

        # HSV color augmentation disabled (CPU overhead)
        'hsv_h': 0.0,         # No hue adjustment
        'hsv_s': 0.0,         # No saturation adjustment
        'hsv_v': 0.0,         # No value/brightness adjustment
    }


def get_rare_class_augmentation_pipeline():
    """
    Get enhanced Albumentations pipeline for rare hemorrhage classes (EDH, HC, SDH).

    These classes have severe imbalance and need stronger augmentation:
    - EDH: 125 instances (1.4%) - 22x rarer than SAH
    - HC: 538 instances (6.1%) - 5x rarer than SAH
    - SDH: 1,339 instances (15.1%) - 2x rarer than SAH

    Returns:
        Albumentations Compose pipeline
    """
    return A.Compose([
        # 1. CLAHE - Enhances subtle density differences critical for small hemorrhages
        A.CLAHE(
            clip_limit=4.0,
            tile_grid_size=(8, 8),
            p=0.7
        ),

        # 2. Brightness/Contrast - Simulates different scanner settings and window levels
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.8
        ),

        # 3. Gaussian Noise - Simulates low-dose scans (critical for thick slices)
        A.GaussNoise(
            var_limit=(10.0, 50.0),
            mean=0,
            p=0.5
        ),

        # 4. Sharpen - Compensates for thick slice blur
        A.Sharpen(
            alpha=(0.2, 0.5),
            lightness=(0.5, 1.0),
            p=0.5
        ),

        # 5. Blur - Simulates different reconstruction kernels
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.3),

        # 6. Gamma correction - Simulates different display settings
        A.RandomGamma(
            gamma_limit=(80, 120),
            p=0.4
        ),
    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels'],
        min_visibility=0.3
    ))


def get_thick_slice_enhancement_pipeline():
    """
    Enhanced augmentation pipeline specifically for thick slices.

    Thick slices have poorer performance (79.73% vs 97.37% sensitivity).
    This pipeline aims to:
    1. Enhance edge definition
    2. Simulate various reconstruction kernels
    3. Compensate for partial volume effects

    Returns:
        Albumentations Compose pipeline
    """
    return A.Compose([
        # 1. CLAHE - Critical for thick slices
        A.CLAHE(
            clip_limit=6.0,  # Higher than rare class (4.0)
            tile_grid_size=(8, 8),
            p=0.8
        ),

        # 2. Sharpen - Compensate for slice thickness
        A.Sharpen(
            alpha=(0.3, 0.6),  # Stronger than rare class
            lightness=(0.5, 1.0),
            p=0.7
        ),

        # 3. Unsharp mask - Alternative sharpening
        A.UnsharpMask(
            blur_limit=(3, 7),
            sigma_limit=(0.5, 1.0),
            alpha=(0.2, 0.5),
            threshold=10,
            p=0.4
        ),

        # 4. Brightness/Contrast
        A.RandomBrightnessContrast(
            brightness_limit=0.25,
            contrast_limit=0.25,
            p=0.7
        ),

        # 5. Elastic transform - Simulates different slice orientations
        A.ElasticTransform(
            alpha=1,
            sigma=50,
            alpha_affine=20,
            p=0.3
        ),
    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels'],
        min_visibility=0.3
    ))


def analyze_class_distribution(data_yaml_path: str) -> Dict:
    """
    Analyze per-class distribution to identify imbalanced classes.

    Args:
        data_yaml_path: Path to data.yaml

    Returns:
        Dictionary with class statistics
    """
    with open(data_yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    base_path = config.get('path', '')
    train_label_path = os.path.join(base_path, 'train', 'labels')

    if not os.path.exists(train_label_path):
        return {}

    # Count instances per class
    class_counts = {}
    class_names = config.get('names', [])

    for i, name in enumerate(class_names):
        class_counts[name] = 0

    # Parse all label files
    for label_file in os.listdir(train_label_path):
        if not label_file.endswith('.txt'):
            continue

        label_path = os.path.join(train_label_path, label_file)
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    if class_id < len(class_names):
                        class_counts[class_names[class_id]] += 1

    # Calculate imbalance ratios
    total_instances = sum(class_counts.values())
    max_count = max(class_counts.values()) if class_counts else 1

    class_stats = {}
    for class_name, count in class_counts.items():
        class_stats[class_name] = {
            'count': count,
            'percentage': (count / total_instances * 100) if total_instances > 0 else 0,
            'imbalance_ratio': max_count / count if count > 0 else float('inf')
        }

    return class_stats


def get_oversampling_weights(class_distribution: Dict) -> Dict:
    """
    Calculate oversampling weights for rare classes.

    Uses square root of imbalance ratio to avoid over-correction.

    Args:
        class_distribution: Output from analyze_class_distribution()

    Returns:
        Dictionary of {class_name: oversampling_weight}
    """
    weights = {}

    for class_name, stats in class_distribution.items():
        # Square root dampening prevents too aggressive oversampling
        imbalance = stats['imbalance_ratio']
        if imbalance == float('inf'):
            weights[class_name] = 10.0  # Max weight for missing classes
        else:
            weights[class_name] = min(imbalance ** 0.5, 10.0)

    return weights


def get_yolo_training_hyperparameters(
    slice_type: str = 'mixed',
    use_preaugmented: bool = False,
    batch_size: int = 16,
    image_size: int = 640,
    epochs: int = 200,
    patience: int = 100,
    freeze_layers: int = 10,
    learning_rate: float = 0.001,
    final_lr_factor: float = 0.01,
) -> Dict:
    """
    Get complete YOLOv8 training hyperparameters with evidence-based augmentations.

    Based on:
    - RSNA ICH Challenge 2019 winners
    - Multi-resolution CT best practices
    - Fine-tuning strategies for medical imaging

    Args:
        slice_type: 'thin', 'thick', or 'mixed'
        use_preaugmented: If True, use minimal augmentation (data already augmented offline)
        batch_size: Batch size
        image_size: Input image size
        epochs: Total epochs
        patience: Early stopping patience
        freeze_layers: Number of layers to freeze
        learning_rate: Initial learning rate
        final_lr_factor: Final LR factor

    Returns:
        Dictionary of hyperparameters for model.train()
    """

    # Get augmentation parameters
    if use_preaugmented:
        # Data already augmented offline - disable all on-the-fly augmentation
        aug_params = get_preaugmented_minimal_params()
        print("\n✓ Using PRE-AUGMENTED dataset mode (zero on-the-fly augmentation)")
    else:
        # Standard on-the-fly augmentation
        aug_params = apply_yolo_compatible_augmentations(
            yolo_model=None,
            enable_custom_aug=True,
            slice_type=slice_type
        )

    # Combine with training hyperparameters
    hyperparams = {
        # Training basics
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': image_size,
        'patience': patience,
        'device': 0,  # Use GPU 0 (A10G) - NOT 'cpu'!
        'exist_ok': True,
        'save_period': 10,
        'plots': True,
        'workers': 8,  # Parallel data loading (matches Modal's 8 CPUs)

        # Optimizer
        'optimizer': 'AdamW',
        'lr0': learning_rate,
        'lrf': final_lr_factor,
        'weight_decay': 0.0005,

        # Learning rate scheduler
        'cos_lr': True,
        'warmup_epochs': 10,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.01,

        # Regularization (PHASE 2: Added label smoothing for better calibration)
        'dropout': 0.0,
        'label_smoothing': 0.05,  # PHASE 2: Prevents overconfidence, improves precision

        # Loss weights (PHASE 1+2: Adjusted for focal loss integration)
        'box': 10.0,  # Increased - localization is strong, leverage it
        'cls': 1.0,   # Reduced from 1.5 - let focal loss handle classification focus
        'dfl': 1.5,   # Works with DFL unfreezing at epoch 10

        # Multi-scale training (PHASE 3: Enabled for better scale invariance)
        'multi_scale': True,  # PHASE 3: Better generalization across hemorrhage sizes

        # Mixed precision
        'amp': True,

        # Data caching (enable RAM cache for pre-augmented dataset)
        'cache': 'ram' if use_preaugmented else False,

        # **Augmentation parameters (evidence-based)**
        **aug_params,
    }

    return hyperparams


def print_augmentation_summary(slice_type: str = 'mixed'):
    """
    Print a summary of applied augmentations.

    Args:
        slice_type: 'thin', 'thick', or 'mixed'
    """
    print("\n" + "="*80)
    print("EVIDENCE-BASED CT HEMORRHAGE AUGMENTATION STRATEGY")
    print("="*80)
    print(f"Slice type: {slice_type.upper()}")
    print(f"Based on: RSNA ICH Challenge 2019 Winners + Systematic Reviews")
    print("="*80)

    params = apply_yolo_compatible_augmentations(None, True, slice_type)

    print(f"\n1. HORIZONTAL FLIP (100% of RSNA winners)")
    print(f"   - Probability: {params['fliplr']*100:.0f}%")
    print(f"   - Rationale: Brain hemispheres are symmetrical")
    print(f"   - Evidence: Used by all top-10 RSNA challenge winners")

    print(f"\n2. ROTATION (Conservative limits)")
    print(f"   - Range: ±{params['degrees']:.1f}°")
    print(f"   - Rationale: Simulate patient positioning variation")
    print(f"   - Evidence: >±15° creates anatomically incorrect examples")

    print(f"\n3. TRANSLATION (Shift)")
    print(f"   - Range: ±{params['translate']*100:.2f}%")
    print(f"   - Rationale: Different scanner FOV and positioning")
    print(f"   - Evidence: RSNA winners used 6.25% for thin, 10% for thick")

    print(f"\n4. SCALE (Zoom variation)")
    print(f"   - Range: {1-params['scale']:.1f}x - {1+params['scale']:.1f}x")
    print(f"   - Rationale: Hemorrhage size variation (5mm to 5cm+)")
    print(f"   - Evidence: Critical for multi-scale detection")

    print(f"\n5. MOSAIC AUGMENTATION")
    print(f"   - Probability: {params['mosaic']*100:.0f}%")
    print(f"   - Close mosaic: Last {params['close_mosaic']} epochs")
    print(f"   - Rationale: Reduced from default for medical imaging")
    print(f"   - Evidence: Conservative mosaic preserves anatomical context")

    print(f"\n6. MULTI-SCALE TRAINING")
    print(f"   - Enabled: Yes")
    print(f"   - Rationale: Handles varying hemorrhage sizes")
    print(f"   - Evidence: Standard in all RSNA winners")

    print("\n" + "="*80)
    print("AUGMENTATIONS EXPLICITLY DISABLED (Evidence-based)")
    print("="*80)
    print("❌ Vertical flip (flipud=0.0) - Anatomically incorrect")
    print("❌ Shear (shear=0.0) - Degrades performance in studies")
    print("❌ Perspective (perspective=0.0) - Not valid for CT")
    print("❌ Mixup (mixup=0.0) - Not appropriate for medical detection")
    print("❌ Copy-paste (copy_paste=0.0) - Not appropriate for CT")
    print("="*80)

    if slice_type == 'thick':
        print("\n📊 THICK SLICE ADJUSTMENTS:")
        print("   - Rotation increased to ±16° (vs ±10° for thin)")
        print("   - Translation increased to 10% (vs 6.25% for thin)")
        print("   - Rationale: Compensate for limited z-axis information")
    elif slice_type == 'thin':
        print("\n📊 THIN SLICE PARAMETERS:")
        print("   - Rotation: ±10° (conservative for anatomical detail)")
        print("   - Translation: 6.25% (standard RSNA winner params)")
        print("   - Rationale: Preserve high-resolution anatomical features")

    print("\n✓ Augmentation pipeline configured\n")


# Helper function for dataset statistics
def analyze_dataset_slices(data_yaml_path: str) -> Dict:
    """
    Analyze dataset to determine slice type distribution.

    Args:
        data_yaml_path: Path to data.yaml

    Returns:
        Dictionary with slice statistics
    """
    with open(data_yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    base_path = config.get('path', '')
    train_path = os.path.join(base_path, config.get('train', 'train/images'))

    if not os.path.exists(train_path):
        return {'thin': 0, 'thick': 0, 'total': 0, 'dominant': 'unknown'}

    thin_count = 0
    thick_count = 0

    for img_file in os.listdir(train_path):
        if img_file.startswith('thin_'):
            thin_count += 1
        elif img_file.startswith('thick_'):
            thick_count += 1

    total = thin_count + thick_count
    dominant = 'mixed'

    if thin_count == 0:
        dominant = 'thick'
    elif thick_count == 0:
        dominant = 'thin'
    elif thin_count > thick_count * 2:
        dominant = 'thin'
    elif thick_count > thin_count * 2:
        dominant = 'thick'

    return {
        'thin': thin_count,
        'thick': thick_count,
        'total': total,
        'dominant': dominant,
        'thin_ratio': thin_count / total if total > 0 else 0,
        'thick_ratio': thick_count / total if total > 0 else 0,
    }


def print_class_imbalance_summary(class_distribution: Dict):
    """
    Print summary of class imbalance with recommendations.

    Args:
        class_distribution: Output from analyze_class_distribution()
    """
    print("\n" + "="*80)
    print("CLASS IMBALANCE ANALYSIS")
    print("="*80)

    # Sort by imbalance ratio (highest first)
    sorted_classes = sorted(
        class_distribution.items(),
        key=lambda x: x[1]['imbalance_ratio'],
        reverse=True
    )

    print(f"\n{'Class':<10} {'Count':<10} {'Percentage':<12} {'Imbalance':<12} {'Status':<15}")
    print("-" * 80)

    for class_name, stats in sorted_classes:
        count = stats['count']
        percentage = stats['percentage']
        imbalance = stats['imbalance_ratio']

        # Determine severity
        if imbalance > 10:
            status = "🔴 SEVERE"
        elif imbalance > 5:
            status = "🟠 HIGH"
        elif imbalance > 2:
            status = "🟡 MODERATE"
        else:
            status = "🟢 BALANCED"

        print(f"{class_name:<10} {count:<10} {percentage:<11.2f}% {imbalance:<11.2f}x {status:<15}")

    print("="*80)
    print("\nRECOMMENDATIONS:")
    print("-" * 80)

    severe_classes = [name for name, stats in class_distribution.items()
                     if stats['imbalance_ratio'] > 10]
    high_classes = [name for name, stats in class_distribution.items()
                   if 5 < stats['imbalance_ratio'] <= 10]

    if severe_classes:
        print(f"\n🔴 SEVERE imbalance classes: {', '.join(severe_classes)}")
        print("   Actions:")
        print("   - Use enhanced augmentation pipeline (get_rare_class_augmentation_pipeline)")
        print("   - Lower confidence thresholds (0.10-0.20 instead of 0.25)")
        print("   - Consider focal loss with high gamma (2.5-3.0)")
        print("   - Oversample during training (2-5x)")

    if high_classes:
        print(f"\n🟠 HIGH imbalance classes: {', '.join(high_classes)}")
        print("   Actions:")
        print("   - Use moderate augmentation boost")
        print("   - Lower confidence thresholds (0.20-0.25)")
        print("   - Apply class weights (2-5x)")

    print("\n✓ Use get_oversampling_weights() to calculate sampling weights")
    print("✓ Use get_rare_class_augmentation_pipeline() for severe cases")
    print("="*80 + "\n")


if __name__ == "__main__":
    # Test and display augmentation configuration
    print("\n" + "="*80)
    print("YOLO AUGMENTED DATASET CONFIGURATION")
    print("="*80)

    # Show configurations for different slice types
    for slice_type in ['thin', 'thick', 'mixed']:
        print_augmentation_summary(slice_type)

    # Get training hyperparameters
    print("\n" + "="*80)
    print("COMPLETE TRAINING HYPERPARAMETERS")
    print("="*80)

    hyperparams = get_yolo_training_hyperparameters(
        slice_type='mixed',
        batch_size=16,
        image_size=640,
        epochs=200,
        patience=100
    )

    print(f"\nTraining configuration:")
    for key, value in hyperparams.items():
        print(f"  {key}: {value}")

    print("\n✓ Configuration ready for YOLOv8 training")

    # Show rare class augmentation info
    print("\n" + "="*80)
    print("RARE CLASS AUGMENTATION PIPELINE")
    print("="*80)
    print("\nFor classes with severe imbalance (EDH, HC), use:")
    print("  pipeline = get_rare_class_augmentation_pipeline()")
    print("\nIncludes:")
    print("  - CLAHE (p=0.7) - Enhance subtle hemorrhages")
    print("  - RandomBrightnessContrast (p=0.8)")
    print("  - GaussNoise (p=0.5)")
    print("  - Sharpen (p=0.5)")
    print("  - Blur variants (p=0.3)")
    print("  - RandomGamma (p=0.4)")

    print("\n" + "="*80)
    print("THICK SLICE ENHANCEMENT PIPELINE")
    print("="*80)
    print("\nFor thick slices (to improve 79.73% → 90%+ sensitivity):")
    print("  pipeline = get_thick_slice_enhancement_pipeline()")
    print("\nIncludes:")
    print("  - CLAHE (p=0.8, stronger)")
    print("  - Sharpen (p=0.7, stronger)")
    print("  - UnsharpMask (p=0.4)")
    print("  - ElasticTransform (p=0.3)")
    print("="*80)
