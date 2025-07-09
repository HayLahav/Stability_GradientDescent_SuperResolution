"""
Loss functions for super-resolution training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import math


class PSNRLoss(nn.Module):
    """Peak Signal-to-Noise Ratio loss"""
    
    def __init__(self, max_val: float = 1.0):
        super().__init__()
        self.max_val = max_val
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = F.mse_loss(pred, target)
        if mse == 0:
            return torch.tensor(float('inf'))
        
        psnr = 20 * torch.log10(self.max_val / torch.sqrt(mse))
        # Return negative PSNR as loss (to minimize)
        return -psnr


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index loss
    
    Args:
        window_size: Size of the gaussian kernel
        num_channels: Number of image channels
    """
    
    def __init__(self, window_size: int = 11, num_channels: int = 3):
        super().__init__()
        self.window_size = window_size
        self.num_channels = num_channels
        
        # Create gaussian kernel
        self.register_buffer('window', self._create_window(window_size, num_channels))
    
    def _create_window(self, window_size: int, channel: int) -> torch.Tensor:
        """Create 2D gaussian kernel"""
        def gaussian(window_size: int, sigma: float) -> torch.Tensor:
            x = torch.arange(window_size).float() - window_size // 2
            gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
            return gauss / gauss.sum()
        
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        
        return window
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute SSIM loss"""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # Compute means
        mu1 = F.conv2d(pred, self.window, padding=self.window_size//2, groups=self.num_channels)
        mu2 = F.conv2d(target, self.window, padding=self.window_size//2, groups=self.num_channels)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        # Compute variances
        sigma1_sq = F.conv2d(pred*pred, self.window, padding=self.window_size//2, groups=self.num_channels) - mu1_sq
        sigma2_sq = F.conv2d(target*target, self.window, padding=self.window_size//2, groups=self.num_channels) - mu2_sq
        sigma12 = F.conv2d(pred*target, self.window, padding=self.window_size//2, groups=self.num_channels) - mu1_mu2
        
        # Compute SSIM
        ssim_map = ((2*mu1_mu2 + C1) * (2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        # Return 1 - SSIM as loss
        return 1 - ssim_map.mean()


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using pre-trained VGG features
    
    Args:
        feature_layers: VGG layers to use for feature extraction
        weights: Weights for each layer
    """
    
    def __init__(
        self,
        feature_layers: Optional[List[str]] = None,
        weights: Optional[List[float]] = None
    ):
        super().__init__()
        
        if feature_layers is None:
            feature_layers = ['relu1_2', 'relu2_2', 'relu3_3']
        
        if weights is None:
            weights = [1.0] * len(feature_layers)
        
        self.feature_layers = feature_layers
        self.weights = weights
        
        # Load pre-trained VGG
        from torchvision import models
        vgg = models.vgg16(pretrained=True).features
        
        # Extract layers
        self.features = nn.ModuleList()
        layer_map = {
            'relu1_1': 1, 'relu1_2': 3,
            'relu2_1': 6, 'relu2_2': 8,
            'relu3_1': 11, 'relu3_2': 13, 'relu3_3': 15,
            'relu4_1': 18, 'relu4_2': 20, 'relu4_3': 22,
            'relu5_1': 25, 'relu5_2': 27, 'relu5_3': 29
        }
        
        for layer in feature_layers:
            if layer in layer_map:
                idx = layer_map[layer]
                self.features.append(nn.Sequential(*list(vgg.children())[:idx+1]))
        
        # Freeze VGG parameters
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute perceptual loss"""
        loss = 0.0
        
        # Normalize if needed (VGG expects specific normalization)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(pred.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(pred.device)
        
        pred = (pred - mean) / std
        target = (target - mean) / std
        
        # Compute features and losses
        for feature_extractor, weight in zip(self.features, self.weights):
            pred_features = feature_extractor(pred)
            target_features = feature_extractor(target)
            loss += weight * F.mse_loss(pred_features, target_features)
        
        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss function
    
    Args:
        mse_weight: Weight for MSE loss
        psnr_weight: Weight for PSNR loss
        ssim_weight: Weight for SSIM loss
        perceptual_weight: Weight for perceptual loss
    """
    
    def __init__(
        self,
        mse_weight: float = 1.0,
        psnr_weight: float = 0.0,
        ssim_weight: float = 0.0,
        perceptual_weight: float = 0.0
    ):
        super().__init__()
        
        self.mse_weight = mse_weight
        self.psnr_weight = psnr_weight
        self.ssim_weight = ssim_weight
        self.perceptual_weight = perceptual_weight
        
        # Initialize component losses
        if psnr_weight > 0:
            self.psnr_loss = PSNRLoss()
        if ssim_weight > 0:
            self.ssim_loss = SSIMLoss()
        if perceptual_weight > 0:
            self.perceptual_loss = PerceptualLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute combined loss"""
        total_loss = 0.0
        
        if self.mse_weight > 0:
            total_loss += self.mse_weight * F.mse_loss(pred, target)
        
        if self.psnr_weight > 0:
            total_loss += self.psnr_weight * self.psnr_loss(pred, target)
        
        if self.ssim_weight > 0:
            total_loss += self.ssim_weight * self.ssim_loss(pred, target)
        
        if self.perceptual_weight > 0:
            total_loss += self.perceptual_weight * self.perceptual_loss(pred, target)
        
        return total_loss
