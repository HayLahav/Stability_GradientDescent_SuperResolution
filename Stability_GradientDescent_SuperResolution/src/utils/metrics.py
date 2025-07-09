"""
Evaluation metrics for super-resolution
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict
import math


def calculate_psnr(
    img1: torch.Tensor,
    img2: torch.Tensor,
    max_val: float = 1.0
) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio
    
    Args:
        img1: First image [B, C, H, W] or [C, H, W]
        img2: Second image [B, C, H, W] or [C, H, W]
        max_val: Maximum pixel value
        
    Returns:
        PSNR value in dB
    """
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
    
    mse = torch.mean((img1 - img2) ** 2)
    
    if mse == 0:
        return float('inf')
    
    psnr = 20 * math.log10(max_val / math.sqrt(mse.item()))
    return psnr


def calculate_ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    K1: float = 0.01,
    K2: float = 0.03,
    max_val: float = 1.0
) -> float:
    """
    Calculate Structural Similarity Index
    
    Args:
        img1: First image [B, C, H, W]
        img2: Second image [B, C, H, W]
        window_size: Size of gaussian window
        K1, K2: SSIM constants
        max_val: Maximum pixel value
        
    Returns:
        SSIM value
    """
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
    
    C1 = (K1 * max_val) ** 2
    C2 = (K2 * max_val) ** 2
    
    # Create gaussian window
    def gaussian_window(size: int, sigma: float = 1.5) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g.outer(g)
    
    window = gaussian_window(window_size).unsqueeze(0).unsqueeze(0)
    window = window.expand(img1.shape[1], 1, window_size, window_size)
    window = window.to(img1.device)
    
    # Compute local means
    mu1 = torch.nn.functional.conv2d(img1, window, padding=window_size//2, groups=img1.shape[1])
    mu2 = torch.nn.functional.conv2d(img2, window, padding=window_size//2, groups=img2.shape[1])
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # Compute local variances and covariance
    sigma1_sq = torch.nn.functional.conv2d(img1**2, window, padding=window_size//2, groups=img1.shape[1]) - mu1_sq
    sigma2_sq = torch.nn.functional.conv2d(img2**2, window, padding=window_size//2, groups=img2.shape[1]) - mu2_sq
    sigma12 = torch.nn.functional.conv2d(img1*img2, window, padding=window_size//2, groups=img1.shape[1]) - mu1_mu2
    
    # SSIM formula
    ssim_map = ((2*mu1_mu2 + C1) * (2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean().item()


def calculate_mse(
    img1: torch.Tensor,
    img2: torch.Tensor
) -> float:
    """
    Calculate Mean Squared Error
    
    Args:
        img1: First image
        img2: Second image
        
    Returns:
        MSE value
    """
    return torch.mean((img1 - img2) ** 2).item()


def evaluate_model(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: str = 'cuda',
    metrics: Tuple[str, ...] = ('psnr', 'ssim', 'mse')
) -> Dict[str, float]:
    """
    Evaluate model on test set
    
    Args:
        model: Super-resolution model
        test_loader: Test data loader
        device: Device to run on
        metrics: Metrics to compute
        
    Returns:
        Dictionary of metric values
    """
    model.eval()
    model = model.to(device)
    
    results = {metric: [] for metric in metrics}
    
    with torch.no_grad():
        for lr_images, hr_images in test_loader:
            lr_images = lr_images.to(device)
            hr_images = hr_images.to(device)
            
            # Generate SR images
            sr_images = model(lr_images)
            
            # Calculate metrics for each image in batch
            for i in range(lr_images.shape[0]):
                if 'psnr' in metrics:
                    psnr = calculate_psnr(sr_images[i], hr_images[i])
                    results['psnr'].append(psnr)
                
                if 'ssim' in metrics:
                    ssim = calculate_ssim(sr_images[i], hr_images[i])
                    results['ssim'].append(ssim)
                
                if 'mse' in metrics:
                    mse = calculate_mse(sr_images[i], hr_images[i])
                    results['mse'].append(mse)
    
    # Average results
    avg_results = {}
    for metric, values in results.items():
        avg_results[f'{metric}_mean'] = np.mean(values)
        avg_results[f'{metric}_std'] = np.std(values)
    
    return avg_results
