"""
Data generation utilities for synthetic textured image segmentation.

Generates grayscale images composed of two regions with different textures,
modeled as stationary Gaussian processes.
"""

import torch
import numpy as np
from scipy.ndimage import convolve


def generate_gaussian_filter(size, sigma):
    """
    Generate a Gaussian filter kernel.
    
    Args:
        size (int): Size of the filter (size x size)
        sigma (float): Standard deviation of the Gaussian
        
    Returns:
        torch.Tensor: Normalized Gaussian kernel
    """
    ax = np.linspace(-size / 2, size / 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / np.sum(kernel)
    return torch.from_numpy(kernel).float()


def generate_data(batch_size, size_image, sigma1, sigma2, sigma, device=None):
    """
    Generate batch of synthetic textured images and ground truth segmentations.
    
    Each image contains two regions with different texture statistics.
    Textures are created by convolving white Gaussian noise with a 
    covariance filter (Gaussian kernel).
    
    Args:
        batch_size (int): Number of images in batch
        size_image (int): Size of square image (size_image x size_image)
        sigma1 (float): Texture parameter for region 1 (filter std)
        sigma2 (float): Texture parameter for region 2 (filter std)
        sigma (float): Texture parameter for both regions (noise std)
        device (torch.device, optional): Device to place tensors on
        
    Returns:
        tuple: (images, ground_truth_masks)
            - images: torch.Tensor of shape (batch_size, 1, size_image, size_image)
            - masks: torch.Tensor of shape (batch_size, 1, size_image, size_image)
              with values in {0, 1}
    """
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dtype = torch.float32
    
    images = []
    masks = []
    
    for _ in range(batch_size):
        # Generate random partition
        mask = torch.randint(0, 2, (size_image, size_image)).float()
        
        # Generate textures as Gaussian processes
        # Texture 1: white noise convolved with Gaussian kernel (sigma1)
        noise1 = torch.randn(size_image, size_image) * sigma
        kernel1 = generate_gaussian_filter(int(3 * sigma1 + 1), sigma1)
        kernel_size = kernel1.shape[0]
        
        # Pad noise for convolution
        pad = kernel_size // 2
        noise1_padded = torch.nn.functional.pad(
            noise1.unsqueeze(0).unsqueeze(0), 
            (pad, pad, pad, pad), 
            mode='reflect'
        )
        
        texture1 = torch.nn.functional.conv2d(
            noise1_padded, 
            kernel1.unsqueeze(0).unsqueeze(0)
        ).squeeze()
        texture1 = texture1[:size_image, :size_image]
        
        # Texture 2
        noise2 = torch.randn(size_image, size_image) * sigma
        kernel2 = generate_gaussian_filter(int(3 * sigma2 + 1), sigma2)
        kernel_size = kernel2.shape[0]
        pad = kernel_size // 2
        
        noise2_padded = torch.nn.functional.pad(
            noise2.unsqueeze(0).unsqueeze(0), 
            (pad, pad, pad, pad), 
            mode='reflect'
        )
        
        texture2 = torch.nn.functional.conv2d(
            noise2_padded, 
            kernel2.unsqueeze(0).unsqueeze(0)
        ).squeeze()
        texture2 = texture2[:size_image, :size_image]
        
        # Combine textures according to mask
        image = mask * texture1 + (1 - mask) * texture2
        
        # Normalize
        image = (image - image.mean()) / (image.std() + 1e-8)
        
        images.append(image.unsqueeze(0).type(dtype))
        masks.append(mask.unsqueeze(0).type(dtype))
    
    images = torch.stack(images).to(device)
    masks = torch.stack(masks).to(device)
    
    return images, masks
