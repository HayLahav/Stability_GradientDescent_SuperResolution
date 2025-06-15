"""
Data transformations for super-resolution
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import random


class RandomCrop:
    """Random crop transformation"""
    
    def __init__(self, size: int):
        self.size = size
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        _, h, w = img.shape
        
        if h < self.size or w < self.size:
            # Pad if needed
            pad_h = max(0, self.size - h)
            pad_w = max(0, self.size - w)
            img = F.pad(img, (0, pad_w, 0, pad_h), mode='reflect')
            _, h, w = img.shape
        
        top = random.randint(0, h - self.size)
        left = random.randint(0, w - self.size)
        
        return img[:, top:top+self.size, left:left+self.size]


class RandomFlip:
    """Random horizontal and vertical flip"""
    
    def __init__(self, h_prob: float = 0.5, v_prob: float = 0.5):
        self.h_prob = h_prob
        self.v_prob = v_prob
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() < self.h_prob:
            img = torch.flip(img, dims=[-1])
        if random.random() < self.v_prob:
            img = torch.flip(img, dims=[-2])
        return img


class RandomRotation:
    """Random 90-degree rotation"""
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        k = random.randint(0, 3)
        return torch.rot90(img, k, dims=[-2, -1])


class Normalize:
    """Normalize image"""
    
    def __init__(
        self,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    ):
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return (img - self.mean) / self.std


class ToTensor:
    """Convert numpy array to tensor"""
    
    def __call__(self, img: np.ndarray) -> torch.Tensor:
        if img.ndim == 2:
            img = img[:, :, np.newaxis]
        
        # HWC to CHW
        img = img.transpose(2, 0, 1)
        return torch.from_numpy(img).float()