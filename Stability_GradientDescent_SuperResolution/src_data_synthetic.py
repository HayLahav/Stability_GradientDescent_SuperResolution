"""
Synthetic pattern generators for controlled experiments
"""

import torch
import torch.nn.functional as F
import numpy as np


def generate_gradient_pattern(size: int) -> torch.Tensor:
    """
    Generate gradient pattern
    
    Args:
        size: Image size
        
    Returns:
        RGB image tensor [3, size, size]
    """
    x = torch.linspace(0, 1, size).unsqueeze(0).repeat(size, 1)
    y = torch.linspace(0, 1, size).unsqueeze(1).repeat(1, size)
    
    r = x
    g = y
    b = (x + y) / 2
    
    return torch.stack([r, g, b], dim=0)


def generate_checkerboard_pattern(size: int, square_size: int = 8) -> torch.Tensor:
    """
    Generate checkerboard pattern
    
    Args:
        size: Image size
        square_size: Size of each square
        
    Returns:
        RGB image tensor [3, size, size]
    """
    x, y = torch.meshgrid(
        torch.arange(size),
        torch.arange(size),
        indexing='ij'
    )
    
    checker = ((x // square_size + y // square_size) % 2).float()
    
    r = checker
    g = 1 - checker
    b = checker * 0.5
    
    return torch.stack([r, g, b], dim=0)


def generate_circular_pattern(size: int) -> torch.Tensor:
    """
    Generate circular/radial pattern
    
    Args:
        size: Image size
        
    Returns:
        RGB image tensor [3, size, size]
    """
    center = size // 2
    x, y = torch.meshgrid(
        torch.arange(size),
        torch.arange(size),
        indexing='ij'
    )
    
    # Distance from center
    dist = torch.sqrt(
        (x - center).float()**2 + (y - center).float()**2
    ) / (size / 2)
    
    r = torch.clamp(dist, 0, 1)
    g = torch.clamp(1 - dist, 0, 1)
    b = torch.sin(dist * np.pi).clamp(0, 1)
    
    return torch.stack([r, g, b], dim=0)


def generate_random_pattern(size: int, complexity: int = 4) -> torch.Tensor:
    """
    Generate random smooth pattern using upsampling
    
    Args:
        size: Image size
        complexity: Base resolution for random pattern
        
    Returns:
        RGB image tensor [3, size, size]
    """
    # Generate low-res random pattern
    base = torch.rand(3, complexity, complexity)
    
    # Upsample to target size
    pattern = F.interpolate(
        base.unsqueeze(0),
        size=(size, size),
        mode='bilinear',
        align_corners=False
    ).squeeze(0)
    
    # Normalize to [0, 1]
    pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min())
    
    return pattern


def generate_texture_pattern(size: int, frequency: int = 10) -> torch.Tensor:
    """
    Generate texture pattern using sinusoidal functions
    
    Args:
        size: Image size
        frequency: Frequency of the pattern
        
    Returns:
        RGB image tensor [3, size, size]
    """
    x = torch.linspace(0, frequency * 2 * np.pi, size)
    y = torch.linspace(0, frequency * 2 * np.pi, size)
    
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    
    r = (torch.sin(xx) * torch.cos(yy) + 1) / 2
    g = (torch.sin(xx + np.pi/3) * torch.cos(yy + np.pi/3) + 1) / 2
    b = (torch.sin(xx + 2*np.pi/3) * torch.cos(yy + 2*np.pi/3) + 1) / 2
    
    return torch.stack([r, g, b], dim=0)