"""
Model factory for creating YOLO and RF-DETR models with unified interface.

Provides a single entry point for model creation, enabling seamless
switching between different architectures via command-line arguments.
"""

from typing import Optional
from models.base_model import BaseModelWrapper
from models.yolo_model import YOLOModelWrapper
from models.rfdetr_model import RFDETRModelWrapper


def create_model(
    model_type: str = 'yolo',
    variant: str = 'medium',
    num_classes: int = 4,
    pretrained: bool = True,
    resolution: int = 640
) -> BaseModelWrapper:
    """
    Factory function to create object detection models.

    Args:
        model_type: Type of model ('yolo' or 'rfdetr')
        variant: Model variant
            - For YOLO: 'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'
            - For RF-DETR: 'nano', 'small', 'medium'
        num_classes: Number of classes (default: 4 for filtered dataset)
        pretrained: Load pretrained COCO weights (default: True)

    Returns:
        Model wrapper instance (YOLOModelWrapper or RFDETRModelWrapper)

    Raises:
        ValueError: If model_type is unknown

    Examples:
        >>> # Create YOLOv8m model
        >>> model = create_model('yolo', 'yolov8m', num_classes=4)

        >>> # Create RF-DETR Medium model
        >>> model = create_model('rfdetr', 'medium', num_classes=4)

        >>> # Train the model
        >>> model.train(data_path='/path/to/data', epochs=200)
    """
    model_type = model_type.lower()

    if model_type == 'yolo':
        print(f"\n{'='*80}")
        print(f"CREATING YOLO MODEL: {variant}")
        print(f"{'='*80}")
        print(f"Classes: {num_classes}")
        print(f"Pretrained: {pretrained}")
        print(f"{'='*80}\n")

        return YOLOModelWrapper(
            variant=variant,
            num_classes=num_classes,
            pretrained=pretrained
        )

    elif model_type == 'rfdetr':
        print(f"\n{'='*80}")
        print(f"CREATING RF-DETR MODEL: {variant}")
        print(f"{'='*80}")
        print(f"Classes: {num_classes}")
        print(f"Pretrained: {pretrained}")
        print(f"Resolution: {resolution}x{resolution}")
        print(f"{'='*80}\n")

        return RFDETRModelWrapper(
            variant=variant,
            num_classes=num_classes,
            pretrained=pretrained,
            resolution=resolution
        )

    else:
        raise ValueError(
            f"Unknown model type: '{model_type}'. "
            f"Choose from: 'yolo', 'rfdetr'"
        )


def get_recommended_config(model_type: str, variant: str, dataset_size: int) -> dict:
    """
    Get recommended training configuration for a model.

    Args:
        model_type: Type of model ('yolo' or 'rfdetr')
        variant: Model variant
        dataset_size: Number of training images

    Returns:
        Dictionary with recommended training parameters

    Examples:
        >>> config = get_recommended_config('yolo', 'yolov8m', dataset_size=5565)
        >>> print(config['batch_size'])
        24
    """
    if model_type == 'yolo':
        return {
            'batch_size': 24,
            'learning_rate': 0.005,
            'epochs': 200,
            'patience': 50,
            'imgsz': 640,
            'optimizer': 'AdamW',
            'warmup_epochs': 5,
            'amp': True,
            'workers': 8,
        }

    elif model_type == 'rfdetr':
        return {
            # Batch & Memory (OOM fixes)
            'batch_size': 16,
            'grad_accum_steps': 4,  # Effective batch = 64
            'gradient_checkpointing': True,  # NEW - Saves ~40% memory
            'resolution': 640,
            'workers': 4,
            'pin_memory': False,
            'amp': True,

            # Learning Rates (Anti-overfitting)
            'lr': 5e-5,  # Decoder learning rate
            'lr_encoder': 5e-6,  # NEW - 10x lower for pretrained encoder
            'warmup_epochs': 10,
            'weight_decay': 1e-3,  # 10x stronger regularization
            'dropout': 0.1,

            # EMA (Better final model)
            'use_ema': True,  # NEW - +2-5% mAP improvement

            # Early Stopping (Native RF-DETR)
            'early_stopping': True,  # NEW - Enable native early stopping
            'early_stopping_patience': 50,  # FIX - Was 'patience' (wrong name)
            'early_stopping_min_delta': 0.001,  # NEW - mAP improvement threshold
            'early_stopping_use_ema': True,  # NEW - Track using EMA weights

            # Checkpoints (Native RF-DETR)
            'output_dir': '/model',  # FIX - Was 'save_dir' (wrong name)
            'checkpoint_interval': 10,  # NEW - Save every 10 epochs

            # Training Duration
            'epochs': 200,
        }

    else:
        return {}


def compare_models(yolo_metrics: dict, rfdetr_metrics: dict) -> dict:
    """
    Compare metrics between YOLO and RF-DETR models.

    Args:
        yolo_metrics: YOLO validation metrics
        rfdetr_metrics: RF-DETR validation metrics

    Returns:
        Comparison dictionary with improvements/differences

    Examples:
        >>> comparison = compare_models(yolo_metrics, rfdetr_metrics)
        >>> print(comparison['mAP50_improvement'])
        +15.3%
    """
    comparison = {
        'model_comparison': 'YOLO vs RF-DETR',
    }

    # Compare mAP50
    if 'mAP50' in yolo_metrics and 'mAP50' in rfdetr_metrics:
        yolo_map50 = yolo_metrics['mAP50']
        rfdetr_map50 = rfdetr_metrics['mAP50']
        improvement = ((rfdetr_map50 - yolo_map50) / yolo_map50) * 100
        comparison['mAP50'] = {
            'yolo': yolo_map50,
            'rfdetr': rfdetr_map50,
            'improvement_pct': improvement,
        }

    # Compare mAP50-95
    if 'mAP50-95' in yolo_metrics and 'mAP50-95' in rfdetr_metrics:
        yolo_map = yolo_metrics['mAP50-95']
        rfdetr_map = rfdetr_metrics['mAP50-95']
        improvement = ((rfdetr_map - yolo_map) / yolo_map) * 100
        comparison['mAP50-95'] = {
            'yolo': yolo_map,
            'rfdetr': rfdetr_map,
            'improvement_pct': improvement,
        }

    # Compare precision
    if 'precision' in yolo_metrics and 'precision' in rfdetr_metrics:
        comparison['precision'] = {
            'yolo': yolo_metrics['precision'],
            'rfdetr': rfdetr_metrics['precision'],
        }

    # Compare recall
    if 'recall' in yolo_metrics and 'recall' in rfdetr_metrics:
        comparison['recall'] = {
            'yolo': yolo_metrics['recall'],
            'rfdetr': rfdetr_metrics['recall'],
        }

    return comparison


# Model registry for easy lookup
MODEL_REGISTRY = {
    'yolo': {
        'variants': ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'],
        'recommended': 'yolov8m',
        'params': {
            'yolov8n': '3.2M',
            'yolov8s': '11.2M',
            'yolov8m': '25.9M',
            'yolov8l': '43.7M',
            'yolov8x': '68.2M',
        },
        'data_format': 'YOLO (normalized xywh)',
    },
    'rfdetr': {
        'variants': ['nano', 'small', 'medium'],
        'recommended': 'medium',
        'params': {
            'nano': '~8M',
            'small': '~15-20M',
            'medium': '~29M',
        },
        'data_format': 'COCO (absolute xyxy)',
    },
}


def print_model_info(model_type: str) -> None:
    """
    Print information about available model variants.

    Args:
        model_type: Type of model ('yolo' or 'rfdetr')
    """
    if model_type not in MODEL_REGISTRY:
        print(f"Unknown model type: {model_type}")
        return

    info = MODEL_REGISTRY[model_type]
    print(f"\n{'='*80}")
    print(f"{model_type.upper()} MODEL VARIANTS")
    print(f"{'='*80}")
    print(f"Data format: {info['data_format']}")
    print(f"Recommended variant: {info['recommended']}")
    print(f"\nAvailable variants:")
    for variant in info['variants']:
        params = info['params'].get(variant, 'unknown')
        marker = '✓' if variant == info['recommended'] else ' '
        print(f"  {marker} {variant:15s} - {params:10s} parameters")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    # Demo usage
    print("Model Factory Demo\n")

    # Print available models
    print_model_info('yolo')
    print_model_info('rfdetr')

    # Example model creation
    print("\nCreating YOLO model...")
    yolo_model = create_model('yolo', 'yolov8m', num_classes=4)
    print(yolo_model.get_model_info())

    print("\nCreating RF-DETR model...")
    rfdetr_model = create_model('rfdetr', 'medium', num_classes=4)
    print(rfdetr_model.get_model_info())

    # Get recommended configs
    print("\nRecommended YOLO config:")
    print(get_recommended_config('yolo', 'yolov8m', 5565))

    print("\nRecommended RF-DETR config:")
    print(get_recommended_config('rfdetr', 'medium', 5565))
