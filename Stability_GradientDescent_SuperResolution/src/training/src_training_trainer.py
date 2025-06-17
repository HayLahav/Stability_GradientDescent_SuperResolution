"""
Training classes for stability analysis
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Optional, Callable, Tuple
import numpy as np
from tqdm import tqdm
import copy

from ..stability import StabilityAnalyzer
from ..optimizers import AdaFMOptimizer
from .callbacks import Callback


class Trainer:
    """
    Standard trainer for super-resolution models
    
    Args:
        model: Neural network model
        optimizer: Optimizer instance
        loss_fn: Loss function
        device: Device to train on
        callbacks: List of callbacks
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer,
        loss_fn: nn.Module,
        device: str = 'cuda',
        callbacks: Optional[List[Callback]] = None
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.callbacks = callbacks or []
        
        self.train_losses = []
        self.val_losses = []
        self.epoch = 0
    
    def train_epoch(
        self,
        train_loader: DataLoader
    ) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {self.epoch}')
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
            # Callback on batch end
            for callback in self.callbacks:
                callback.on_batch_end(batch_idx, {'loss': loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def validate(
        self,
        val_loader: DataLoader
    ) -> float:
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        return avg_loss
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 100
    ) -> Dict[str, List[float]]:
        """
        Train model for multiple epochs
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs
            
        Returns:
            Training history
        """
        # Callback on training start
        for callback in self.callbacks:
            callback.on_train_begin()
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Callback on epoch start
            for callback in self.callbacks:
                callback.on_epoch_begin(epoch)
            
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = None
            if val_loader:
                val_loss = self.validate(val_loader)
                self.val_losses.append(val_loss)
            
            # Print progress
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            if val_loss:
                print(f"  Val Loss: {val_loss:.4f}")
            
            # Callback on epoch end
            metrics = {'train_loss': train_loss}
            if val_loss:
                metrics['val_loss'] = val_loss
            
            for callback in self.callbacks:
                if not callback.on_epoch_end(epoch, metrics):
                    print("Early stopping triggered")
                    break
        
        # Callback on training end
        for callback in self.callbacks:
            callback.on_train_end()
        
        history = {'train_loss': self.train_losses}
        if self.val_losses:
            history['val_loss'] = self.val_losses
        
        return history


class StabilityTrainer(Trainer):
    """
    Trainer with stability analysis capabilities
    
    Args:
        model: Neural network model
        optimizer: Optimizer instance
        loss_fn: Loss function
        stability_analyzer: Stability analyzer instance
        perturbed_dataset: Dataset with perturbation
        device: Device to train on
        callbacks: List of callbacks
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer,
        loss_fn: nn.Module,
        stability_analyzer: StabilityAnalyzer,
        perturbed_dataset: Optional[Dataset] = None,
        device: str = 'cuda',
        callbacks: Optional[List[Callback]] = None
    ):
        super().__init__(model, optimizer, loss_fn, device, callbacks)
        self.stability_analyzer = stability_analyzer
        self.perturbed_dataset = perturbed_dataset
        
        # For parallel training on perturbed dataset
        if perturbed_dataset:
            self.model_prime = copy.deepcopy(model).to(device)
            if isinstance(optimizer, AdaFMOptimizer):
                self.optimizer_prime = AdaFMOptimizer(
                    self.model_prime.parameters(),
                    gamma=optimizer.gamma,
                    delta=optimizer.delta,
                    single_variable=True
                )
            else:
                self.optimizer_prime = type(optimizer)(
                    self.model_prime.parameters(),
                    **optimizer.defaults
                )
        
        # Stability metrics storage
        self.parameter_distances = []
        self.empirical_gammas = []
        self.theoretical_gammas = []
    
    def train_epoch_with_stability(
        self,
        train_loader: DataLoader,
        perturbed_loader: Optional[DataLoader] = None
    ) -> Tuple[float, Dict[str, float]]:
        """Train epoch with stability tracking"""
        # Train main model
        train_loss = self.train_epoch(train_loader)
        
        # Train perturbed model if available
        if perturbed_loader and hasattr(self, 'model_prime'):
            self.model_prime.train()
            
            for inputs, targets in perturbed_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                self.optimizer_prime.zero_grad()
                outputs = self.model_prime(inputs)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                self.optimizer_prime.step()
        
        # Compute stability metrics
        stability_metrics = {}
        if hasattr(self, 'model_prime'):
            # Parameter distance
            param_dist = self.stability_analyzer.compute_parameter_distance(
                self.model, self.model_prime
            )
            stability_metrics['parameter_distance'] = param_dist
            self.parameter_distances.append(param_dist)
            
            # Get current iteration count and learning rate
            T = self.epoch * len(train_loader)
            m = len(train_loader.dataset)
            
            if isinstance(self.optimizer, AdaFMOptimizer):
                lr = self.optimizer.get_current_lrs()['eta_x']
            elif hasattr(self.optimizer, 'param_groups'):
                lr = self.optimizer.param_groups[0]['lr']
            else:
                lr = 0.01  # Default
            
            # Theoretical gamma
            theo_gamma = self.stability_analyzer.compute_theoretical_gamma_general(
                T, m, lr
            )
            stability_metrics['theoretical_gamma'] = theo_gamma
            self.theoretical_gammas.append(theo_gamma)
        
        return train_loss, stability_metrics
    
    def fit_with_stability(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        perturbed_loader: Optional[DataLoader] = None,
        num_epochs: int = 100,
        test_samples: Optional[torch.Tensor] = None
    ) -> Dict[str, List[float]]:
        """
        Train with stability analysis
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            perturbed_loader: Perturbed data loader
            num_epochs: Number of epochs
            test_samples: Test samples for empirical gamma
            
        Returns:
            Extended training history with stability metrics
        """
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train with stability
            train_loss, stability_metrics = self.train_epoch_with_stability(
                train_loader, perturbed_loader
            )
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = None
            if val_loader:
                val_loss = self.validate(val_loader)
                self.val_losses.append(val_loss)
            
            # Compute empirical gamma if test samples provided
            if test_samples is not None and hasattr(self, 'model_prime'):
                emp_gamma = self.stability_analyzer.compute_empirical_gamma(
                    self.model, self.model_prime, test_samples.to(self.device)
                )
                stability_metrics['empirical_gamma'] = emp_gamma
                self.empirical_gammas.append(emp_gamma)
            
            # Print progress
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            if val_loss:
                print(f"  Val Loss: {val_loss:.4f}")
            
            if stability_metrics:
                print("  Stability Metrics:")
                for key, value in stability_metrics.items():
                    print(f"    {key}: {value:.6f}")
        
        # Compile history
        history = {
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            'parameter_distance': self.parameter_distances,
            'empirical_gamma': self.empirical_gammas,
            'theoretical_gamma': self.theoretical_gammas
        }
        
        return history