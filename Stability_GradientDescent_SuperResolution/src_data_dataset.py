"""
Dataset classes for super-resolution
"""

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from typing import Optional, Tuple, List, Callable
import numpy as np
import os
from PIL import Image

from .synthetic_generator import (
    generate_gradient_pattern,
    generate_checkerboard_pattern,
    generate_circular_pattern,
    generate_random_pattern
)


class SyntheticSRDataset(Dataset):
    """
    Synthetic dataset for super-resolution experiments
    Creates LR-HR image pairs with controlled degradation
    
    Args:
        num_samples: Number of samples to generate
        image_size: Size of HR images
        scale_factor: Downscaling factor for LR images
        noise_level: Noise level to add to images
        pattern_types: Types of patterns to generate
    """
    
    def __init__(
        self,
        num_samples: int = 1000,
        image_size: int = 32,
        scale_factor: int = 2,
        noise_level: float = 0.01,
        pattern_types: Optional[List[str]] = None
    ):
        self.num_samples = num_samples
        self.image_size = image_size
        self.scale_factor = scale_factor
        self.lr_size = image_size // scale_factor
        self.noise_level = noise_level
        
        if pattern_types is None:
            pattern_types = ['gradient', 'checkerboard', 'circular', 'random']
        self.pattern_types = pattern_types
        
        # Generate synthetic data
        self.hr_images = []
        self.lr_images = []
        self._generate_data()
    
    def _generate_data(self):
        """Generate synthetic HR-LR image pairs"""
        pattern_generators = {
            'gradient': generate_gradient_pattern,
            'checkerboard': generate_checkerboard_pattern,
            'circular': generate_circular_pattern,
            'random': generate_random_pattern
        }
        
        for i in range(self.num_samples):
            # Select pattern type
            pattern_type = self.pattern_types[i % len(self.pattern_types)]
            generator = pattern_generators[pattern_type]
            
            # Generate HR image
            hr = generator(self.image_size)
            
            # Create LR image by downsampling
            lr = F.interpolate(
                hr.unsqueeze(0), 
                size=(self.lr_size, self.lr_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            
            # Add noise
            if self.noise_level > 0:
                lr += torch.randn_like(lr) * self.noise_level
                hr += torch.randn_like(hr) * self.noise_level
            
            # Clamp to valid range
            lr = torch.clamp(lr, 0, 1)
            hr = torch.clamp(hr, 0, 1)
            
            self.hr_images.append(hr)
            self.lr_images.append(lr)
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.lr_images[idx], self.hr_images[idx]
    
    def add_perturbation(self, idx: int, perturbation_strength: float = 0.1):
        """Add perturbation to specific sample for stability analysis"""
        self.lr_images[idx] += torch.randn_like(self.lr_images[idx]) * perturbation_strength
        self.lr_images[idx] = torch.clamp(self.lr_images[idx], 0, 1)


class RealSRDataset(Dataset):
    """
    Real image dataset for super-resolution
    
    Args:
        root_dir: Root directory containing images
        scale_factor: Downscaling factor
        transform: Optional transform to apply
        image_size: Size to crop/resize images to
    """
    
    def __init__(
        self,
        root_dir: str,
        scale_factor: int = 2,
        transform: Optional[Callable] = None,
        image_size: Optional[int] = None
    ):
        self.root_dir = root_dir
        self.scale_factor = scale_factor
        self.transform = transform
        self.image_size = image_size
        
        # Get list of image files
        self.image_files = []
        for ext in ['*.jpg', '*.png', '*.jpeg']:
            self.image_files.extend(
                [f for f in os.listdir(root_dir) if f.lower().endswith(ext[1:])]
            )
        
        if not self.image_files:
            raise ValueError(f"No images found in {root_dir}")
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load image
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        # Convert to tensor
        image = np.array(image) / 255.0
        image = torch.from_numpy(image).float().permute(2, 0, 1)
        
        # Crop/resize if needed
        if self.image_size:
            # Center crop to square
            _, h, w = image.shape
            min_dim = min(h, w)
            top = (h - min_dim) // 2
            left = (w - min_dim) // 2
            image = image[:, top:top+min_dim, left:left+min_dim]
            
            # Resize to target size
            image = F.interpolate(
                image.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
        # Generate HR and LR pairs
        hr_image = image
        lr_size = hr_image.shape[-1] // self.scale_factor
        
        lr_image = F.interpolate(
            hr_image.unsqueeze(0),
            size=(lr_size, lr_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        
        # Apply transforms if any
        if self.transform:
            lr_image = self.transform(lr_image)
            hr_image = self.transform(hr_image)
        
        return lr_image, hr_image