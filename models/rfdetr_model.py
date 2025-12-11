"""
RF-DETR model wrapper implementing BaseModelWrapper interface.

Wraps Roboflow's RF-DETR for unified training interface.
"""

import os
import json
import torch
from typing import Dict, Any, List, Optional
from pathlib import Path

try:
    # Roboflow RF-DETR package (will be installed via Modal)
    from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium
    RFDETR_AVAILABLE = True
except ImportError:
    print("Warning: rfdetr not installed. Will be available in Modal environment.")
    RFDETR_AVAILABLE = False

from models.base_model import BaseModelWrapper


class RFDETRModelWrapper(BaseModelWrapper):
    """
    Wrapper for RF-DETR models (Roboflow).

    Supports: RF-DETR Nano, Small, Medium
    """

    def __init__(self, variant: str = 'medium', num_classes: int = 4, pretrained: bool = True, resolution: int = 640):
        """
        Initialize RF-DETR model using official rfdetr package.

        Args:
            variant: RF-DETR variant (nano, small, medium)
            num_classes: Number of classes (not used - inferred from COCO JSON)
            pretrained: Load COCO pretrained weights (default: True)
            resolution: Image resolution (default: 640)

        Note: RF-DETR models automatically detect num_classes from COCO JSON during training.
        """
        super().__init__(variant, num_classes)

        if not RFDETR_AVAILABLE:
            print("rfdetr package not available in local environment. Will be initialized in Modal.")
            self.model = None
            return

        # Initialize model based on variant (using official API)
        variant_lower = variant.lower()
        if variant_lower == 'nano':
            self.model = RFDETRNano(resolution=resolution)
        elif variant_lower == 'small':
            self.model = RFDETRSmall(resolution=resolution)
        elif variant_lower == 'medium':
            self.model = RFDETRMedium(resolution=resolution)
        else:
            raise ValueError(f"Unknown RF-DETR variant: {variant}. "
                           f"Choose from: nano, small, medium")

        print(f"Loaded RF-DETR {variant} model (resolution={resolution}, pretrained={pretrained})")

    def train(
        self,
        data_path: str,
        epochs: int = 200,
        batch_size: int = 8,
        learning_rate: float = None,  # Handled automatically by rfdetr
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train RF-DETR model using built-in .train() method.

        Note: RF-DETR has built-in training with automatic learning rate, optimizer,
        loss functions, and validation. Most hyperparameters are handled automatically.

        Args:
            data_path: Path to COCO dataset directory (containing train/, valid/, test/)
            epochs: Number of epochs (default: 200)
            batch_size: Batch size per GPU (default: 8, keep batch × grad_accum = 16)
            learning_rate: Not used - RF-DETR handles this automatically
            **kwargs: Additional RF-DETR training parameters
                - grad_accum_steps: Gradient accumulation steps (default: 2)
                  Note: Keep batch_size × grad_accum_steps = 16 for consistent training

        Returns:
            Training results dictionary
        """
        print(f"\n{'='*80}")
        print(f"TRAINING RF-DETR {self.variant.upper()}")
        print(f"{'='*80}")
        print(f"Dataset: {data_path}")
        print(f"Epochs: {epochs}, Batch: {batch_size}")
        print(f"{'='*80}\n")

        if self.model is None:
            print("ERROR: Model not initialized. Cannot train.")
            return {'results': {}, 'metrics': {}}

        # RF-DETR training using built-in method
        # The package handles: lr scheduling, optimizer, loss, validation, checkpointing
        grad_accum_steps = kwargs.get('grad_accum_steps', 2)

        print(f"Starting RF-DETR training (batch={batch_size}, grad_accum={grad_accum_steps}, effective_batch={batch_size * grad_accum_steps})")
        print("Note: Learning rate, optimizer, and loss are handled automatically by rfdetr package")
        print("\n" + "="*80)
        print("TRAINING PROGRESS (Simplified View)")
        print("="*80)
        print("Epoch | Batch | Loss    | CE Loss | BBox Loss | GIoU Loss | LR      | Time/Batch")
        print("-"*80)

        try:
            # Run training with RF-DETR's native WandB integration
            print("\nNote: RF-DETR training output is verbose (this is normal)")
            print("✓ 'num_classes mismatch' - Expected, reinitializing for 4 classes")
            print("✓ PyTorch warnings - Deprecation notices, safe to ignore")
            print("✓ Using RF-DETR's native WandB integration (better metrics logging)\n")

            # Pass all kwargs to RF-DETR (including lr, weight_decay, dropout, EMA, early stopping, etc.)
            train_kwargs = {
                'dataset_dir': data_path,
                'epochs': epochs,
                'batch_size': batch_size,
                'grad_accum_steps': grad_accum_steps,
                'wandb': True,  # Enable RF-DETR's native WandB integration
                'project': 'brain-ct-hemorrhage',  # WandB project name
                'run': f'rfdetr_{self.variant}_aug3x_native_ema',  # WandB run name
                **kwargs  # Include all other parameters (EMA, early stopping, etc.)
            }

            # Add resume support if checkpoint exists
            if hasattr(self, 'resume_checkpoint') and self.resume_checkpoint:
                train_kwargs['resume'] = self.resume_checkpoint
                print(f"✓ Resuming from checkpoint: {self.resume_checkpoint}\n")

            results = self.model.train(**train_kwargs)

            self.training_results = results

            return {
                'results': results,
                'metrics': self._extract_metrics(results) if results else {},
            }

        except Exception as e:
            print(f"ERROR during RF-DETR training: {e}")
            return {
                'results': {},
                'metrics': {},
                'error': str(e)
            }

    def _extract_metrics(self, results: Any) -> Dict[str, Any]:
        """
        Extract metrics from RF-DETR training results.

        Args:
            results: RF-DETR training results object

        Returns:
            Dictionary of metrics
        """
        try:
            # RF-DETR returns metrics in results object
            # Extract standard COCO metrics
            if hasattr(results, 'metrics'):
                return results.metrics
            elif isinstance(results, dict):
                return results.get('metrics', {})
            else:
                return {}
        except Exception as e:
            print(f"Warning: Could not extract metrics: {e}")
            return {}

    def validate(
        self,
        data_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Validate RF-DETR model.

        Args:
            data_path: Path to COCO validation dataset
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for matching
            **kwargs: Additional validation parameters

        Returns:
            Validation metrics
        """
        print(f"\n{'='*80}")
        print(f"VALIDATING RF-DETR {self.variant.upper()}")
        print(f"{'='*80}\n")

        if self.model is None:
            print("Model not loaded. Returning mock validation results.")
            return {}

        # Placeholder for RF-DETR validation
        # TODO: Implement COCO-style validation
        val_config = {
            'dataset_dir': data_path,
            'conf_threshold': conf_threshold,
            'iou_threshold': iou_threshold,
        }

        print(f"Validation config: {json.dumps(val_config, indent=2)}")

        return {}

    def predict(
        self,
        image_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Run inference on image using RF-DETR.

        Args:
            image_path: Path to image
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold (note: DETR uses bipartite matching, not NMS)
            **kwargs: Additional prediction parameters

        Returns:
            List of detections
        """
        if self.model is None:
            print("Model not loaded. Cannot run inference.")
            return []

        # Run inference using Roboflow API
        results = self.model.infer(
            image_path,
            confidence=conf_threshold,
        )

        # Parse results to standard format
        detections = []
        for prediction in results:
            detection = {
                'class_id': prediction.get('class_id', 0),
                'class_name': prediction.get('class', 'unknown'),
                'confidence': prediction.get('confidence', 0.0),
                'bbox': [
                    prediction.get('x', 0) - prediction.get('width', 0) / 2,
                    prediction.get('y', 0) - prediction.get('height', 0) / 2,
                    prediction.get('x', 0) + prediction.get('width', 0) / 2,
                    prediction.get('y', 0) + prediction.get('height', 0) / 2,
                ],  # Convert center format to xyxy
            }
            detections.append(detection)

        return detections

    def save(self, path: str) -> None:
        """
        Save RF-DETR model weights.

        Args:
            path: Path to save checkpoint
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)

        if self.model is None:
            print("Model not loaded. Cannot save.")
            return

        # Placeholder for saving logic
        # TODO: Implement RF-DETR checkpoint saving
        print(f"Saving RF-DETR model to: {path}")
        print("Note: RF-DETR checkpoint saving requires custom implementation")

    def load(self, path: str) -> None:
        """
        Load RF-DETR checkpoint for resume training.

        Uses RF-DETR's native resume parameter in next train() call.

        Args:
            path: Path to checkpoint file
        """
        if not os.path.exists(path):
            print(f"Checkpoint not found: {path}")
            return

        # Store checkpoint path for use in next train() call
        # RF-DETR's native resume parameter will handle loading
        self.resume_checkpoint = path
        print(f"Will resume from checkpoint: {path}")
        print("Note: Checkpoint will be loaded in next train() call via RF-DETR's native resume")

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get RF-DETR model information.

        Returns:
            Model info dictionary
        """
        info = super().get_model_info()

        # Add RF-DETR-specific info
        variant_params = {
            'nano': '~8M',
            'small': '~15-20M',
            'medium': '~29M',
        }

        info.update({
            'architecture': 'RF-DETR (Real-time DEtection TRansformer)',
            'parameters': variant_params.get(self.variant, 'unknown'),
            'backbone': 'Vision Transformer',
            'model_id': getattr(self, 'model_id', None),
        })

        return info


# Helper function for COCO dataset validation
def validate_coco_dataset(dataset_dir: str) -> bool:
    """
    Validate COCO dataset structure.

    Args:
        dataset_dir: Path to COCO dataset directory

    Returns:
        True if valid, False otherwise
    """
    required_files = [
        'train/_annotations.coco.json',
        'valid/_annotations.coco.json',
    ]

    for file_path in required_files:
        full_path = os.path.join(dataset_dir, file_path)
        if not os.path.exists(full_path):
            print(f"Missing required file: {full_path}")
            return False

    print(f"COCO dataset structure validated: {dataset_dir}")
    return True
