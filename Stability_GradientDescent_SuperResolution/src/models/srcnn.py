"""
SRCNN implementation with optional AdaFM layers and correction filter
Based on Dong et al., ECCV 2014
"""

import torch
import torch.nn as nn
from typing import Optional

from .adafm_layers import AdaFMLayer
from .correction_filter import CorrectionFilter


class SimpleSRCNN(nn.Module):
    """
    Simplified SRCNN for super-resolution with optional AdaFM layers
    
    Args:
        use_adafm: Whether to use AdaFM layers
        use_correction: Whether to use correction filter
        num_channels: Number of input/output channels
        num_filters: Number of filters in hidden layers
    """
    
    def __init__(
        self,
        use_adafm: bool = False,
        use_correction: bool = False,
        num_channels: int = 3,
        num_filters: tuple = (64, 32)
    ):
        super().__init__()
        self.use_correction = use_correction
        self.use_adafm = use_adafm
        
        # Correction filter
        if use_correction:
            self.correction = CorrectionFilter(kernel_size=3)
        
        # Feature extraction
        self.conv1 = nn.Conv2d(
            num_channels, num_filters[0], 
            kernel_size=9, padding=4
        )
        self.relu1 = nn.ReLU()
        if use_adafm:
            self.adafm1 = AdaFMLayer(num_filters[0])
        
        # Non-linear mapping
        self.conv2 = nn.Conv2d(
            num_filters[0], num_filters[1], 
            kernel_size=5, padding=2
        )
        self.relu2 = nn.ReLU()
        if use_adafm:
            self.adafm2 = AdaFMLayer(num_filters[1])
        
        # Reconstruction
        self.conv3 = nn.Conv2d(
            num_filters[1], num_channels, 
            kernel_size=5, padding=2
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input low-resolution image [B, C, H, W]
            
        Returns:
            Super-resolved image [B, C, H, W]
        """
        # Apply correction filter if enabled
        if self.use_correction:
            x = self.correction(x)
        
        # Feature extraction
        x = self.relu1(self.conv1(x))
        if self.use_adafm:
            x = self.adafm1(x)
        
        # Non-linear mapping
        x = self.relu2(self.conv2(x))
        if self.use_adafm:
            x = self.adafm2(x)
        
        # Reconstruction
        x = self.conv3(x)
        
        return x
    
    def get_num_params(self) -> int:
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
