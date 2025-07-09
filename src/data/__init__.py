"""
Data handling utilities and datasets
"""

# Import from synthetic.py (main implementation)
from .synthetic import (
    SyntheticSRDataset,
    generate_gradient_pattern,
    generate_checkerboard_pattern,
    generate_circular_pattern,
    generate_random_pattern,
    generate_texture_pattern,
    generate_mixed_pattern,
    create_perturbed_dataset
)

# Import from dataset.py (real data support)
from .dataset import RealSRDataset

# Import transforms
from .transforms import (
    RandomCrop,
    RandomFlip,
    RandomRotation,
    Normalize,
    ToTensor
)

__all__ = [
    # Main synthetic dataset
    'SyntheticSRDataset',
    
    # Pattern generators
    'generate_gradient_pattern',
    'generate_checkerboard_pattern', 
    'generate_circular_pattern',
    'generate_random_pattern',
    'generate_texture_pattern',
    'generate_mixed_pattern',
    
    # Utilities
    'create_perturbed_dataset',
    
    # Real data support
    'RealSRDataset',
    
    # Transforms
    'RandomCrop',
    'RandomFlip',
    'RandomRotation',
    'Normalize',
    'ToTensor'
]
