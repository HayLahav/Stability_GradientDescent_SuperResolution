"""
Synthetic pattern generators for controlled experiments
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from typing import Optional, List, Tuple
import copy


def generate_gradient_pattern(size: int) -> torch.Tensor:
    """
    Generate gradient pattern with multiple orientations
    
    Args:
        size: Image size
        
    Returns:
        RGB image tensor [3, size, size]
    """
    x = torch.linspace(0, 1, size)
    y = torch.linspace(0, 1, size)
    xx, yy = torch.meshgrid(x, y, indexing='ij')

    # Multiple gradient orientations
    r = xx * 0.7 + yy * 0.3
    g = yy * 0.8 + xx * 0.2
    b = (xx + yy) / 2 + 0.1 * torch.sin(5 * xx) * torch.cos(5 * yy)

    return torch.stack([r, g, b], dim=0)


def generate_checkerboard_pattern(size: int, square_size: Optional[int] = None) -> torch.Tensor:
    """
    Generate checkerboard pattern with varying square sizes
    
    Args:
        size: Image size
        square_size: Size of each square (random if None)
        
    Returns:
        RGB image tensor [3, size, size]
    """
    if square_size is None:
        square_size = np.random.choice([4, 6, 8, 12])

    x, y = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
    checker = ((x // square_size + y // square_size) % 2).float()

    # Add some variation
    r = checker + 0.2 * torch.sin(x * 0.1)
    g = (1 - checker) + 0.2 * torch.cos(y * 0.1)
    b = 0.5 + 0.3 * checker + 0.1 * torch.sin((x + y) * 0.05)

    return torch.stack([torch.clamp(r, 0, 1), torch.clamp(g, 0, 1), torch.clamp(b, 0, 1)], dim=0)


def generate_circular_pattern(size: int) -> torch.Tensor:
    """
    Generate circular pattern with multiple frequencies
    
    Args:
        size: Image size
        
    Returns:
        RGB image tensor [3, size, size]
    """
    center = size // 2
    x, y = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')

    dist = torch.sqrt((x - center).float()**2 + (y - center).float()**2) / (size / 2)
    angle = torch.atan2(y - center, x - center)

    r = torch.clamp(torch.sin(dist * 3 * np.pi) + 0.5, 0, 1)
    g = torch.clamp(torch.cos(dist * 2 * np.pi) + 0.5, 0, 1)
    b = torch.clamp(torch.sin(angle * 3) * torch.exp(-dist) + 0.5, 0, 1)

    return torch.stack([r, g, b], dim=0)


def generate_texture_pattern(size: int) -> torch.Tensor:
    """
    Generate texture-like patterns
    
    Args:
        size: Image size
        
    Returns:
        RGB image tensor [3, size, size]
    """
    x = torch.linspace(0, 4 * np.pi, size)
    y = torch.linspace(0, 4 * np.pi, size)
    xx, yy = torch.meshgrid(x, y, indexing='ij')

    r = (torch.sin(xx) * torch.cos(yy * 1.3) + 1) / 2
    g = (torch.sin(xx * 1.7) * torch.cos(yy) + 1) / 2
    b = (torch.sin(xx * 0.8) * torch.cos(yy * 0.9) + 1) / 2

    return torch.stack([r, g, b], dim=0)


def generate_mixed_pattern(size: int) -> torch.Tensor:
    """
    Generate mixed patterns combining multiple types
    
    Args:
        size: Image size
        
    Returns:
        RGB image tensor [3, size, size]
    """
    # Combine gradient and circular
    grad = generate_gradient_pattern(size)
    circ = generate_circular_pattern(size)

    # Blend with spatial variation
    x = torch.linspace(0, 1, size)
    blend_weight = x.unsqueeze(1).repeat(1, size)

    mixed = blend_weight.unsqueeze(0) * grad + (1 - blend_weight.unsqueeze(0)) * circ
    return mixed


def generate_random_pattern(size: int, complexity: int = 6) -> torch.Tensor:
    """
    Generate random pattern with multiple scales
    
    Args:
        size: Image size
        complexity: Base resolution for pattern complexity
        
    Returns:
        RGB image tensor [3, size, size]
    """
    # Generate multiple scales and combine
    pattern = torch.zeros(3, size, size)

    for scale in [complexity//2, complexity, complexity*2]:
        base = torch.rand(3, scale, scale)
        scaled = F.interpolate(base.unsqueeze(0), size=(size, size),
                             mode='bilinear', align_corners=False).squeeze(0)
        pattern += scaled * (1.0 / 3)

    # Normalize
    pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min())
    return pattern


class SyntheticSRDataset(Dataset):
    """
    Synthetic dataset with improved pattern generation and realistic degradation
    """

    def __init__(self, num_samples=1000, image_size=32, scale_factor=2, noise_level=0.01,
                 pattern_types=None, realistic_degradation=True):
        self.num_samples = num_samples
        self.image_size = image_size
        self.scale_factor = scale_factor
        self.lr_size = image_size // scale_factor
        self.noise_level = noise_level
        self.realistic_degradation = realistic_degradation

        if pattern_types is None:
            pattern_types = ['gradient', 'checkerboard', 'circular', 'random', 'texture', 'mixed']
        self.pattern_types = pattern_types

        # Generate data
        self.hr_images = []
        self.lr_images = []
        self._generate_data()

        print(f"✅ Generated {num_samples} synthetic image pairs")
        print(f"   HR size: {image_size}×{image_size}, LR size: {self.lr_size}×{self.lr_size}")
        print(f"   Pattern types: {len(pattern_types)}, Noise level: {noise_level}")

    def _generate_data(self):
        """Generate synthetic HR-LR image pairs with realistic degradation"""
        for i in range(self.num_samples):
            # Select pattern type
            pattern_type = self.pattern_types[i % len(self.pattern_types)]

            # Generate HR image based on pattern type
            if pattern_type == 'gradient':
                hr = generate_gradient_pattern(self.image_size)
            elif pattern_type == 'checkerboard':
                hr = generate_checkerboard_pattern(self.image_size)
            elif pattern_type == 'circular':
                hr = generate_circular_pattern(self.image_size)
            elif pattern_type == 'texture':
                hr = generate_texture_pattern(self.image_size)
            elif pattern_type == 'mixed':
                hr = generate_mixed_pattern(self.image_size)
            else:  # random
                hr = generate_random_pattern(self.image_size)

            # Create LR image with realistic degradation
            if self.realistic_degradation:
                lr = self._realistic_downsampling(hr)
            else:
                lr = F.interpolate(hr.unsqueeze(0), size=(self.lr_size, self.lr_size),
                                 mode='bilinear', align_corners=False).squeeze(0)

            # Add noise
            if self.noise_level > 0:
                lr += torch.randn_like(lr) * self.noise_level
                hr += torch.randn_like(hr) * self.noise_level * 0.5  # Less noise on HR

            # Clamp to valid range
            lr = torch.clamp(lr, 0, 1)
            hr = torch.clamp(hr, 0, 1)

            self.hr_images.append(hr)
            self.lr_images.append(lr)

    def _realistic_downsampling(self, hr_img):
        """Apply realistic degradation model"""
        # Apply slight blur before downsampling
        kernel_size = 3
        sigma = 0.5
        channels = hr_img.shape[0]

        # Create Gaussian kernel
        x = torch.arange(kernel_size).float() - kernel_size // 2
        gaussian_1d = torch.exp(-x.pow(2) / (2 * sigma ** 2))
        gaussian_1d = gaussian_1d / gaussian_1d.sum()
        gaussian_2d = gaussian_1d[:, None] * gaussian_1d[None, :]
        kernel = gaussian_2d.expand(channels, 1, kernel_size, kernel_size)

        # Apply blur
        hr_blurred = F.conv2d(hr_img.unsqueeze(0), kernel,
                             padding=kernel_size//2, groups=channels).squeeze(0)

        # Downsample
        lr = F.interpolate(hr_blurred.unsqueeze(0), size=(self.lr_size, self.lr_size),
                          mode='bilinear', align_corners=False).squeeze(0)

        return lr

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.lr_images[idx], self.hr_images[idx]

    def add_perturbation(self, idx: int, perturbation_strength: float = 0.1):
        """Add perturbation to specific sample for stability analysis"""
        if idx < len(self.lr_images):
            self.lr_images[idx] += torch.randn_like(self.lr_images[idx]) * perturbation_strength
            self.lr_images[idx] = torch.clamp(self.lr_images[idx], 0, 1)

    def visualize_samples(self, num_samples: int = 6):
        """Visualize sample patterns"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, num_samples, figsize=(3*num_samples, 6))

        for i in range(min(num_samples, len(self.hr_images))):
            # HR image
            hr_img = self.hr_images[i].permute(1, 2, 0).numpy()
            axes[0, i].imshow(hr_img)
            axes[0, i].set_title(f'HR {i+1}')
            axes[0, i].axis('off')

            # LR image
            lr_img = self.lr_images[i].permute(1, 2, 0).numpy()
            axes[1, i].imshow(lr_img)
            axes[1, i].set_title(f'LR {i+1}')
            axes[1, i].axis('off')

        plt.suptitle('Synthetic Dataset Samples')
        plt.tight_layout()
        plt.show()


def create_perturbed_dataset(original_dataset, perturbation_idx=0, perturbation_strength=0.1):
    """
    Create a perturbed copy of dataset for stability analysis
    
    Args:
        original_dataset: Original SyntheticSRDataset
        perturbation_idx: Index of sample to perturb
        perturbation_strength: Strength of perturbation
        
    Returns:
        Perturbed dataset copy
    """
    # Create copy
    perturbed_dataset = SyntheticSRDataset(
        num_samples=original_dataset.num_samples,
        image_size=original_dataset.image_size,
        scale_factor=original_dataset.scale_factor,
        noise_level=original_dataset.noise_level,
        pattern_types=original_dataset.pattern_types,
        realistic_degradation=original_dataset.realistic_degradation
    )
    
    # Copy data
    perturbed_dataset.hr_images = [img.clone() for img in original_dataset.hr_images]
    perturbed_dataset.lr_images = [img.clone() for img in original_dataset.lr_images]
    
    # Add perturbation
    perturbed_dataset.add_perturbation(perturbation_idx, perturbation_strength)
    
    return perturbed_dataset
