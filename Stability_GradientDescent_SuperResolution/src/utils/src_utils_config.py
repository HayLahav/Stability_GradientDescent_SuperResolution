"""
Configuration management utilities
"""

import yaml
import json
from typing import Dict, Any, Optional
from pathlib import Path
import copy


class Config:
    """Configuration class with dot notation access"""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict
        self._convert_to_attributes(config_dict)
    
    def _convert_to_attributes(self, d: Dict[str, Any], parent: Optional[str] = None):
        """Convert dictionary to attributes"""
        for key, value in d.items():
            if isinstance(value, dict):
                # Create nested Config object
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert back to dictionary"""
        result = {}
        for key, value in self.__dict__.items():
            if key.startswith('_'):
                continue
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result
    
    def __getitem__(self, key: str) -> Any:
        """Dictionary-style access"""
        return getattr(self, key)
    
    def __setitem__(self, key: str, value: Any):
        """Dictionary-style setting"""
        setattr(self, key, value)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Safe get with default"""
        return getattr(self, key, default)


def load_config(config_path: str) -> Config:
    """
    Load configuration from YAML or JSON file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Config object
    """
    path = Path(config_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(path, 'r') as f:
        if path.suffix in ['.yaml', '.yml']:
            config_dict = yaml.safe_load(f)
        elif path.suffix == '.json':
            config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")
    
    return Config(config_dict)


def save_config(config: Config, save_path: str):
    """
    Save configuration to file
    
    Args:
        config: Config object
        save_path: Path to save configuration
    """
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    config_dict = config.to_dict() if isinstance(config, Config) else config
    
    with open(path, 'w') as f:
        if path.suffix in ['.yaml', '.yml']:
            yaml.dump(config_dict, f, default_flow_style=False)
        elif path.suffix == '.json':
            json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configurations, with override taking precedence
    
    Args:
        base_config: Base configuration
        override_config: Override configuration
        
    Returns:
        Merged configuration
    """
    merged = copy.deepcopy(base_config)
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


# Default configurations
DEFAULT_TRAINING_CONFIG = {
    'model': {
        'name': 'SimpleSRCNN',
        'use_correction': False,
        'use_adafm': False,
        'num_channels': 3,
        'num_filters': [64, 32]
    },
    'optimizer': {
        'name': 'SGD',
        'lr': 0.01,
        'momentum': 0.9,
        'weight_decay': 0.0
    },
    'training': {
        'epochs': 100,
        'batch_size': 32,
        'num_workers': 4,
        'pin_memory': True
    },
    'data': {
        'num_samples': 1000,
        'image_size': 32,
        'scale_factor': 2,
        'noise_level': 0.01,
        'val_split': 0.2
    },
    'stability': {
        'compute': True,
        'perturbation_idx': 0,
        'perturbation_strength': 0.1
    },
    'loss': {
        'name': 'MSE',
        'weights': {
            'mse': 1.0,
            'psnr': 0.0,
            'ssim': 0.0,
            'perceptual': 0.0
        }
    },
    'logging': {
        'save_dir': 'results',
        'log_interval': 10,
        'save_checkpoint': True,
        'tensorboard': True
    }
}


DEFAULT_ADAFM_CONFIG = {
    'optimizer': {
        'name': 'AdaFM',
        'gamma': 1.0,
        'lam': 1.0,
        'delta': 0.001
    }
}