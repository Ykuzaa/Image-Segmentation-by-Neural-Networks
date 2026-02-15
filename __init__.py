"""
Image Segmentation with CNN and Stochastic Optimization
"""

__version__ = "1.0.0"
__author__ = "El-Mehdi BOULAALAM, Matthias Herla"

from .data_generation import generate_data, generate_gaussian_filter
from .models import SimpleCNN, UNet
from .training import (
    create_optimizer,
    train_step,
    evaluate,
    compute_iou,
    compute_dice,
    TrainingConfig
)

__all__ = [
    'generate_data',
    'generate_gaussian_filter',
    'SimpleCNN',
    'UNet',
    'create_optimizer',
    'train_step',
    'evaluate',
    'compute_iou',
    'compute_dice',
    'TrainingConfig'
]
