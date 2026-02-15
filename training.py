"""
Training utilities for image segmentation.

Provides optimization loops and metrics computation.
"""

import torch
import torch.nn as nn


def create_optimizer(model, optimizer_name, learning_rate, **kwargs):
    """
    Create an optimizer instance.
    
    Args:
        model (nn.Module): Model to optimize
        optimizer_name (str): Name of optimizer ('SGD', 'Adam', 'RMSProp')
        learning_rate (float): Learning rate
        **kwargs: Additional optimizer hyperparameters
        
    Returns:
        torch.optim.Optimizer: Optimizer instance
        
    Raises:
        ValueError: If optimizer_name is not recognized
    """
    optimizer_name = optimizer_name.upper()
    
    if optimizer_name == 'SGD':
        momentum = kwargs.get('momentum', 0.9)
        return torch.optim.SGD(
            model.parameters(), 
            lr=learning_rate, 
            momentum=momentum
        )
    elif optimizer_name == 'ADAM':
        betas = kwargs.get('betas', (0.9, 0.999))
        eps = kwargs.get('eps', 1e-8)
        return torch.optim.Adam(
            model.parameters(), 
            lr=learning_rate, 
            betas=betas, 
            eps=eps
        )
    elif optimizer_name == 'RMSPROP':
        alpha = kwargs.get('alpha', 0.99)
        return torch.optim.RMSprop(
            model.parameters(), 
            lr=learning_rate, 
            alpha=alpha
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def train_step(model, images, masks, optimizer, criterion, device):
    """
    Single training step.
    
    Args:
        model (nn.Module): Model to train
        images (torch.Tensor): Batch of input images
        masks (torch.Tensor): Batch of ground truth masks
        optimizer (torch.optim.Optimizer): Optimizer
        criterion (nn.Module): Loss function
        device (torch.device): Device to compute on
        
    Returns:
        float: Loss value
    """
    images = images.to(device)
    masks = masks.to(device)
    
    optimizer.zero_grad()
    
    # Forward pass
    predictions = model(images)
    
    # Loss computation
    loss = criterion(predictions, masks)
    
    # Backward pass
    loss.backward()
    
    # Optimizer step
    optimizer.step()
    
    return loss.item()


def evaluate(model, images, masks, criterion, device):
    """
    Evaluate model on a batch.
    
    Args:
        model (nn.Module): Model to evaluate
        images (torch.Tensor): Batch of input images
        masks (torch.Tensor): Batch of ground truth masks
        criterion (nn.Module): Loss function
        device (torch.device): Device to compute on
        
    Returns:
        dict: Dictionary with 'loss' and 'accuracy'
    """
    model.eval()
    
    with torch.no_grad():
        images = images.to(device)
        masks = masks.to(device)
        
        predictions = model(images)
        loss = criterion(predictions, masks)
        
        # Binary accuracy
        binary_predictions = (predictions > 0.5).float()
        accuracy = (binary_predictions == masks).float().mean()
    
    model.train()
    
    return {
        'loss': loss.item(),
        'accuracy': accuracy.item()
    }


def compute_iou(predictions, masks, threshold=0.5):
    """
    Compute Intersection over Union (IoU) metric.
    
    Args:
        predictions (torch.Tensor): Model predictions [0, 1]
        masks (torch.Tensor): Ground truth masks {0, 1}
        threshold (float): Threshold for binary classification
        
    Returns:
        float: IoU score
    """
    binary_predictions = (predictions > threshold).float()
    
    intersection = torch.sum(binary_predictions * masks)
    union = torch.sum(binary_predictions) + torch.sum(masks) - intersection
    
    iou = intersection / (union + 1e-8)
    return iou.item()


def compute_dice(predictions, masks, threshold=0.5):
    """
    Compute Dice coefficient.
    
    Args:
        predictions (torch.Tensor): Model predictions [0, 1]
        masks (torch.Tensor): Ground truth masks {0, 1}
        threshold (float): Threshold for binary classification
        
    Returns:
        float: Dice coefficient
    """
    binary_predictions = (predictions > threshold).float()
    
    intersection = torch.sum(binary_predictions * masks)
    dice = 2 * intersection / (torch.sum(binary_predictions) + torch.sum(masks) + 1e-8)
    
    return dice.item()


class TrainingConfig:
    """Configuration class for training."""
    
    def __init__(self, 
                 batch_size=32,
                 learning_rate=0.001,
                 num_epochs=50,
                 optimizer='Adam',
                 image_size=32,
                 sigma1=0.5,
                 sigma2=0.5,
                 sigma=0.1,
                 **kwargs):
        """
        Initialize training configuration.
        
        Args:
            batch_size (int): Batch size
            learning_rate (float): Learning rate
            num_epochs (int): Number of training epochs
            optimizer (str): Optimizer name
            image_size (int): Size of generated images
            sigma1, sigma2, sigma (float): Texture parameters
            **kwargs: Additional optimizer parameters
        """
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.image_size = image_size
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.sigma = sigma
        self.optimizer_kwargs = kwargs
    
    def __repr__(self):
        return (f"TrainingConfig(batch_size={self.batch_size}, "
                f"lr={self.learning_rate}, optimizer={self.optimizer}, "
                f"epochs={self.num_epochs})")
