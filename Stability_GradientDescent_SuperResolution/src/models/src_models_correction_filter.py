"""
Correction Filter for Super-Resolution
Based on Abu Hussein et al., CVPR 2020
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CorrectionFilter(nn.Module):
    """
    Correction Filter for aligning LR inputs with training domain
    Uses frequency-domain kernel correction
    
    Args:
        kernel_size: Size of the correction kernel
        num_channels: Number of input channels
        init_identity: Initialize as identity filter
    """
    
    def __init__(
        self, 
        kernel_size: int = 3,
        num_channels: int = 3,
        init_identity: bool = True
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_channels = num_channels
        
        # Learnable correction kernel
        self.correction_kernel = nn.Parameter(
            torch.zeros(num_channels, 1, kernel_size, kernel_size)
        )
        
        # Initialize kernel
        if init_identity:
            self._initialize_identity()
        else:
            self._initialize_random()
    
    def _initialize_identity(self):
        """Initialize as identity kernel"""
        mid = self.kernel_size // 2
        with torch.no_grad():
            self.correction_kernel.zero_()
            self.correction_kernel[:, 0, mid, mid] = 1.0
    
    def _initialize_random(self):
        """Initialize with Xavier uniform"""
        nn.init.xavier_uniform_(self.correction_kernel)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply correction filter to input
        
        Args:
            x: Input image [B, C, H, W]
            
        Returns:
            Corrected image [B, C, H, W]
        """
        batch_size, channels, height, width = x.shape
        
        # Apply depthwise convolution for each channel
        corrected = []
        for c in range(channels):
            channel = x[:, c:c+1, :, :]
            kernel = self.correction_kernel[c:c+1]
            
            # Apply convolution with correction kernel
            corrected_channel = F.conv2d(
                channel, 
                kernel,
                padding=self.kernel_size // 2
            )
            corrected.append(corrected_channel)
        
        # Concatenate channels
        output = torch.cat(corrected, dim=1)
        
        return output
    
    def get_kernel_spectrum(self) -> torch.Tensor:
        """
        Get frequency spectrum of correction kernels
        Useful for visualization and analysis
        
        Returns:
            Frequency spectrum of kernels
        """
        with torch.no_grad():
            # Apply 2D FFT to each kernel
            kernels = self.correction_kernel.squeeze(1)
            spectrum = torch.fft.fft2(kernels)
            magnitude = torch.abs(spectrum)
            
        return magnitude