"""
Adaptive Feature Modification (AdaFM) Layers
Based on He et al., CVPR 2019
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaFMLayer(nn.Module):
    """
    Adaptive Feature Modification Layer
    Modulates features adaptively based on input statistics
    
    Args:
        num_features: Number of feature channels
        reduction_ratio: Channel reduction ratio for efficiency
    """
    
    def __init__(self, num_features: int, reduction_ratio: int = 4):
        super().__init__()
        self.num_features = num_features
        
        # Feature-wise affine parameters
        self.gamma = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        
        # Adaptive modulation network
        self.modulation = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                num_features, 
                num_features // reduction_ratio, 
                kernel_size=1
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                num_features // reduction_ratio, 
                num_features * 2, 
                kernel_size=1
            ),
            nn.Sigmoid()
        )
        
        # Initialize modulation network
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for modulation network"""
        for m in self.modulation.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with adaptive feature modulation
        
        Args:
            x: Input features [B, C, H, W]
            
        Returns:
            Modulated features [B, C, H, W]
        """
        # Compute adaptive modulation factors
        mod_params = self.modulation(x)
        gamma_mod, beta_mod = torch.chunk(mod_params, 2, dim=1)
        
        # Apply feature modulation
        # y = γ * γ_mod * x + (β + β_mod)
        out = x * (self.gamma * gamma_mod) + (self.beta + beta_mod)
        
        return out
    
    def extra_repr(self) -> str:
        """Extra representation for printing"""
        return f'num_features={self.num_features}'