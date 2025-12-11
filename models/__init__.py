"""
Model wrappers and factory for unified YOLO and RF-DETR training.
"""

from models.base_model import BaseModelWrapper
from models.model_factory import create_model

__all__ = ['BaseModelWrapper', 'create_model']
