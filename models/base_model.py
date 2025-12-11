"""
Abstract base class for model wrappers.

Provides a unified interface for both YOLO and RF-DETR models,
allowing seamless switching between architectures.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pathlib import Path


class BaseModelWrapper(ABC):
    """
    Abstract interface for object detection models.

    This class defines the contract that all model wrappers must implement,
    ensuring consistent behavior across different architectures (YOLO, RF-DETR).
    """

    def __init__(self, variant: str, num_classes: int = 4):
        """
        Initialize model wrapper.

        Args:
            variant: Model variant (e.g., 'yolov8m', 'medium' for RF-DETR)
            num_classes: Number of classes (default: 4 for filtered dataset)
        """
        self.variant = variant
        self.num_classes = num_classes
        self.model = None
        self.training_results = None

    @abstractmethod
    def train(
        self,
        data_path: str,
        epochs: int = 200,
        batch_size: int = 16,
        learning_rate: float = 1e-4,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            data_path: Path to dataset (YOLO data.yaml or COCO directory)
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            **kwargs: Additional training parameters

        Returns:
            Dictionary with training results and metrics
        """
        pass

    @abstractmethod
    def validate(
        self,
        data_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Validate the model.

        Args:
            data_path: Path to validation dataset
            conf_threshold: Confidence threshold for predictions
            iou_threshold: IoU threshold for NMS/matching
            **kwargs: Additional validation parameters

        Returns:
            Dictionary with validation metrics (mAP, precision, recall, etc.)
        """
        pass

    @abstractmethod
    def predict(
        self,
        image_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Run inference on a single image.

        Args:
            image_path: Path to input image
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold
            **kwargs: Additional inference parameters

        Returns:
            List of detections, each with format:
            {
                'class_id': int,
                'class_name': str,
                'confidence': float,
                'bbox': [x_min, y_min, x_max, y_max],
            }
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save model weights.

        Args:
            path: Path to save model checkpoint
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load model weights.

        Args:
            path: Path to model checkpoint
        """
        pass

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and statistics.

        Returns:
            Dictionary with model info (parameters, architecture, etc.)
        """
        return {
            'variant': self.variant,
            'num_classes': self.num_classes,
            'architecture': self.__class__.__name__,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(variant='{self.variant}', num_classes={self.num_classes})"
