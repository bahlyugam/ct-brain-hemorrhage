"""
YOLO model wrapper implementing BaseModelWrapper interface.

Wraps Ultralytics YOLOv8 for unified training interface.
"""

import os
from typing import Dict, Any, List, Optional
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("Warning: ultralytics not installed. Install with: pip install ultralytics")
    YOLO = None

from models.base_model import BaseModelWrapper


class YOLOModelWrapper(BaseModelWrapper):
    """
    Wrapper for YOLOv8 models (Ultralytics).

    Supports: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
    """

    def __init__(self, variant: str = 'yolov8m', num_classes: int = 4, pretrained: bool = True):
        """
        Initialize YOLO model.

        Args:
            variant: YOLO variant (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
            num_classes: Number of classes (default: 4)
            pretrained: Load COCO pretrained weights (default: True)
        """
        super().__init__(variant, num_classes)

        if YOLO is None:
            raise ImportError("ultralytics not installed. Install with: pip install ultralytics")

        # Load model
        model_path = f"{variant}.pt" if pretrained else f"{variant}.yaml"
        self.model = YOLO(model_path)

        print(f"Loaded YOLO model: {variant} (pretrained={pretrained})")

    def train(
        self,
        data_path: str,
        epochs: int = 200,
        batch_size: int = 24,
        learning_rate: float = 0.005,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train YOLO model.

        Args:
            data_path: Path to data.yaml file
            epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Learning rate
            **kwargs: Additional YOLO training parameters
                - imgsz: Image size (default: 640)
                - patience: Early stopping patience (default: 50)
                - save: Save checkpoints (default: True)
                - save_period: Save every N epochs (default: 5)
                - project: Project name for saving results
                - name: Run name
                - device: GPU device (default: 0)
                - workers: Number of workers (default: 8)
                - amp: Use mixed precision (default: True)

        Returns:
            Training results dictionary
        """
        print(f"\n{'='*80}")
        print(f"TRAINING YOLO {self.variant.upper()}")
        print(f"{'='*80}")
        print(f"Dataset: {data_path}")
        print(f"Epochs: {epochs}, Batch: {batch_size}, LR: {learning_rate}")
        print(f"{'='*80}\n")

        # Default training config
        train_config = {
            'data': data_path,
            'epochs': epochs,
            'batch': batch_size,
            'lr0': learning_rate,
            'imgsz': kwargs.get('imgsz', 640),
            'patience': kwargs.get('patience', 50),
            'save': kwargs.get('save', True),
            'save_period': kwargs.get('save_period', 5),
            'device': kwargs.get('device', 0),
            'workers': kwargs.get('workers', 8),
            'amp': kwargs.get('amp', True),
            'project': kwargs.get('project', 'runs/detect'),
            'name': kwargs.get('name', f'{self.variant}_4class'),
            'exist_ok': kwargs.get('exist_ok', True),
        }

        # Add any additional kwargs
        for key, value in kwargs.items():
            if key not in train_config:
                train_config[key] = value

        # Train
        results = self.model.train(**train_config)
        self.training_results = results

        return {
            'results': results,
            'metrics': self._extract_metrics(results),
        }

    def validate(
        self,
        data_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Validate YOLO model.

        Args:
            data_path: Path to data.yaml file
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
            **kwargs: Additional validation parameters

        Returns:
            Validation metrics
        """
        print(f"\n{'='*80}")
        print(f"VALIDATING YOLO {self.variant.upper()}")
        print(f"{'='*80}\n")

        val_config = {
            'data': data_path,
            'conf': conf_threshold,
            'iou': iou_threshold,
            'split': kwargs.get('split', 'val'),
            'device': kwargs.get('device', 0),
        }

        results = self.model.val(**val_config)

        return self._extract_metrics(results)

    def predict(
        self,
        image_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Run inference on image.

        Args:
            image_path: Path to image
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold
            **kwargs: Additional prediction parameters

        Returns:
            List of detections
        """
        results = self.model.predict(
            source=image_path,
            conf=conf_threshold,
            iou=iou_threshold,
            device=kwargs.get('device', 0),
            verbose=kwargs.get('verbose', False),
        )

        # Parse results
        detections = []
        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                detection = {
                    'class_id': int(boxes.cls[i]),
                    'class_name': result.names[int(boxes.cls[i])],
                    'confidence': float(boxes.conf[i]),
                    'bbox': boxes.xyxy[i].cpu().numpy().tolist(),  # [x_min, y_min, x_max, y_max]
                }
                detections.append(detection)

        return detections

    def save(self, path: str) -> None:
        """
        Save YOLO model weights.

        Args:
            path: Path to save checkpoint (.pt file)
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"Saved model to: {path}")

    def load(self, path: str) -> None:
        """
        Load YOLO model weights.

        Args:
            path: Path to checkpoint (.pt file)
        """
        self.model = YOLO(path)
        print(f"Loaded model from: {path}")

    def _extract_metrics(self, results) -> Dict[str, Any]:
        """
        Extract metrics from YOLO results object.

        Args:
            results: YOLO results object

        Returns:
            Dictionary of metrics
        """
        try:
            metrics = {}

            if hasattr(results, 'results_dict'):
                # Training results
                metrics = results.results_dict
            elif hasattr(results, 'box'):
                # Validation results
                box_metrics = results.box
                metrics = {
                    'mAP50': float(box_metrics.map50),
                    'mAP50-95': float(box_metrics.map),
                    'precision': float(box_metrics.mp),
                    'recall': float(box_metrics.mr),
                }

                # Per-class metrics
                if hasattr(box_metrics, 'maps'):
                    metrics['per_class_mAP50-95'] = box_metrics.maps.tolist()
                if hasattr(box_metrics, 'ap50'):
                    metrics['per_class_mAP50'] = box_metrics.ap50.tolist()

            return metrics

        except Exception as e:
            print(f"Warning: Could not extract metrics: {e}")
            return {}

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get YOLO model information.

        Returns:
            Model info dictionary
        """
        info = super().get_model_info()

        if self.model is not None:
            try:
                # Add YOLO-specific info
                info['parameters'] = sum(p.numel() for p in self.model.model.parameters())
                info['architecture'] = 'YOLOv8'
            except:
                pass

        return info
