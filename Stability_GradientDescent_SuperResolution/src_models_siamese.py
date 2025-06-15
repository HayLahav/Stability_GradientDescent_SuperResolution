"""
Siamese Network for measuring perceptual similarity
Used to evaluate the quality of super-resolved images
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SiameseNetwork(nn.Module):
    """
    Siamese Network for perceptual similarity measurement
    
    Args:
        input_channels: Number of input channels
        feature_dim: Dimension of final feature vector
    """
    
    def __init__(self, input_channels: int = 3, feature_dim: int = 128):
        super().__init__()
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(input_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Conv Block 2
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Conv Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Similarity computation head
        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for one branch
        
        Args:
            x: Input image [B, C, H, W]
            
        Returns:
            Feature vector [B, D]
        """
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        return features
    
    def forward(
        self, 
        x1: torch.Tensor, 
        x2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for both branches
        
        Args:
            x1: First input image [B, C, H, W]
            x2: Second input image [B, C, H, W]
            
        Returns:
            similarity: Similarity score [B, 1]
            features: Tuple of feature vectors
        """
        # Extract features
        feat1 = self.forward_one(x1)
        feat2 = self.forward_one(x2)
        
        # Compute similarity from feature difference
        diff = torch.abs(feat1 - feat2)
        similarity = self.fc(diff)
        
        return similarity, (feat1, feat2)
    
    def compute_distance(
        self, 
        x1: torch.Tensor, 
        x2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute L2 distance between feature representations
        
        Args:
            x1: First input image [B, C, H, W]
            x2: Second input image [B, C, H, W]
            
        Returns:
            L2 distance [B]
        """
        feat1 = self.forward_one(x1)
        feat2 = self.forward_one(x2)
        
        distance = torch.norm(feat1 - feat2, p=2, dim=1)
        return distance