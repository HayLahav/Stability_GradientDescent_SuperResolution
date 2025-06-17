"""
Training callbacks for monitoring and control
"""

import torch
import os
from typing import Dict, Optional
import numpy as np


class Callback:
    """Base callback class"""
    
    def on_train_begin(self):
        pass
    
    def on_train_end(self):
        pass
    
    def on_epoch_begin(self, epoch: int):
        pass
    
    def on_epoch_end(self, epoch: int, metrics: Dict) -> bool:
        """Return False to stop training"""
        return True
    
    def on_batch_begin(self, batch: int):
        pass
    
    def on_batch_end(self, batch: int, metrics: Dict):
        pass


class EarlyStopping(Callback):
    """
    Early stopping callback
    
    Args:
        monitor: Metric to monitor
        patience: Number of epochs to wait
        mode: 'min' or 'max'
        delta: Minimum change to qualify as improvement
    """
    
    def __init__(
        self,
        monitor: str = 'val_loss',
        patience: int = 10,
        mode: str = 'min',
        delta: float = 0.0001
    ):
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.delta = delta
        
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0
        self.best_epoch = 0
    
    def on_epoch_end(self, epoch: int, metrics: Dict) -> bool:
        if self.monitor not in metrics:
            return True
        
        current = metrics[self.monitor]
        
        if self.mode == 'min':
            improved = current < self.best_value - self.delta
        else:
            improved = current > self.best_value + self.delta
        
        if improved:
            self.best_value = current
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            print(f"Early stopping triggered. Best epoch: {self.best_epoch}")
            return False
        
        return True


class ModelCheckpoint(Callback):
    """
    Save model checkpoints
    
    Args:
        filepath: Path to save checkpoints
        monitor: Metric to monitor
        mode: 'min' or 'max'
        save_best_only: Only save best model
    """
    
    def __init__(
        self,
        filepath: str,
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_best_only: bool = True,
        verbose: bool = True
    ):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.verbose = verbose
        
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        
        # Create directory
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    def on_epoch_end(self, epoch: int, metrics: Dict) -> bool:
        if self.monitor not in metrics:
            return True
        
        current = metrics[self.monitor]
        
        if self.mode == 'min':
            is_best = current < self.best_value
        else:
            is_best = current > self.best_value
        
        if is_best:
            self.best_value = current
        
        if not self.save_best_only or is_best:
            # Get model from trainer (assumes trainer is passed)
            if hasattr(self, 'trainer') and hasattr(self.trainer, 'model'):
                filepath = self.filepath.format(epoch=epoch, **metrics)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.trainer.model.state_dict(),
                    'metrics': metrics,
                    'best_value': self.best_value
                }, filepath)
                
                if self.verbose:
                    print(f"Saved checkpoint to {filepath}")
        
        return True


class LearningRateMonitor(Callback):
    """Monitor learning rate during training"""
    
    def __init__(self):
        self.learning_rates = []
    
    def on_epoch_end(self, epoch: int, metrics: Dict) -> bool:
        if hasattr(self, 'trainer') and hasattr(self.trainer, 'optimizer'):
            optimizer = self.trainer.optimizer
            
            if hasattr(optimizer, 'get_current_lrs'):
                # AdaFM optimizer
                lrs = optimizer.get_current_lrs()
                self.learning_rates.append(lrs)
                print(f"  Learning rates: {lrs}")
            elif hasattr(optimizer, 'param_groups'):
                # Standard optimizer
                lr = optimizer.param_groups[0]['lr']
                self.learning_rates.append({'lr': lr})
                print(f"  Learning rate: {lr:.6f}")
        
        return True


class StabilityMonitor(Callback):
    """
    Monitor stability metrics during training
    
    Args:
        compute_frequency: How often to compute stability metrics
        save_trajectory: Whether to save parameter trajectory
    """
    
    def __init__(
        self,
        compute_frequency: int = 10,
        save_trajectory: bool = False
    ):
        self.compute_frequency = compute_frequency
        self.save_trajectory = save_trajectory
        
        self.stability_history = {
            'parameter_distances': [],
            'empirical_gammas': [],
            'theoretical_gammas': [],
            'gradient_variances': []
        }
        
        if save_trajectory:
            self.trajectory = []
    
    def on_epoch_end(self, epoch: int, metrics: Dict) -> bool:
        if epoch % self.compute_frequency == 0:
            # Extract stability metrics if available
            for key in ['parameter_distance', 'empirical_gamma', 'theoretical_gamma']:
                if key in metrics:
                    self.stability_history[f"{key}s"].append(metrics[key])
            
            # Save trajectory if requested
            if self.save_trajectory and hasattr(self, 'trainer'):
                model_copy = {
                    name: param.cpu().clone()
                    for name, param in self.trainer.model.named_parameters()
                }
                self.trajectory.append(model_copy)
        
        return True
    
    def get_stability_summary(self) -> Dict:
        """Get summary of stability metrics"""
        summary = {}
        
        for key, values in self.stability_history.items():
            if values:
                summary[f"{key}_mean"] = np.mean(values)
                summary[f"{key}_std"] = np.std(values)
                summary[f"{key}_final"] = values[-1]
        
        return summary