"""
Neural network architectures for image segmentation.

Includes baseline CNN and U-Net implementations.
"""

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    Simple CNN baseline for texture segmentation.
    
    Architecture:
    - Conv layers with ReLU activation
    - Final sigmoid for binary segmentation
    """
    
    def __init__(self, num_channels=16, bias=True):
        """
        Initialize SimpleCNN.
        
        Args:
            num_channels (int): Number of filters in convolutional layers
            bias (bool): Whether to use bias in convolutional layers
        """
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, num_channels, kernel_size=3, padding=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=bias),
            nn.ReLU(inplace=True),
        )
        
        self.classifier = nn.Sequential(
            nn.Conv2d(num_channels, 1, kernel_size=1, bias=bias),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, height, width)
            
        Returns:
            torch.Tensor: Output segmentation map of shape (batch_size, 1, height, width)
        """
        x = self.features(x)
        x = self.classifier(x)
        return x


class UNet(nn.Module):
    """
    U-Net architecture for image segmentation.
    
    Encoder-decoder structure with skip connections.
    Better preserves spatial information through skip connections.
    """
    
    def __init__(self, num_channels=16, bias=True):
        """
        Initialize UNet.
        
        Args:
            num_channels (int): Base number of filters
            bias (bool): Whether to use bias in convolutional layers
        """
        super(UNet, self).__init__()
        
        # Encoder
        self.enc1 = self._conv_block(1, num_channels, bias)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.enc2 = self._conv_block(num_channels, num_channels * 2, bias)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Bottleneck
        self.bottleneck = self._conv_block(num_channels * 2, num_channels * 4, bias)
        
        # Decoder
        self.upconv2 = nn.ConvTranspose2d(
            num_channels * 4, num_channels * 2, kernel_size=2, stride=2, bias=bias
        )
        self.dec2 = self._conv_block(num_channels * 4, num_channels * 2, bias)
        
        self.upconv1 = nn.ConvTranspose2d(
            num_channels * 2, num_channels, kernel_size=2, stride=2, bias=bias
        )
        self.dec1 = self._conv_block(num_channels * 2, num_channels, bias)
        
        # Output
        self.final = nn.Sequential(
            nn.Conv2d(num_channels, 1, kernel_size=1, bias=bias),
            nn.Sigmoid()
        )
    
    def _conv_block(self, in_channels, out_channels, bias):
        """
        Double convolution block with ReLU activation.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            bias (bool): Whether to use bias
            
        Returns:
            nn.Sequential: Convolution block
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=bias),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """
        Forward pass with skip connections.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, height, width)
            
        Returns:
            torch.Tensor: Output segmentation map of shape (batch_size, 1, height, width)
        """
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        # Bottleneck
        b = self.bottleneck(p2)
        
        # Decoder with skip connections
        u2 = self.upconv2(b)
        u2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(u2)
        
        u1 = self.upconv1(d2)
        u1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(u1)
        
        # Output
        out = self.final(d1)
        return out
