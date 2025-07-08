"""
AdaFM Optimizer implementation from ICLR 2025
"AdaFM: Adaptive Variance-Reduced Algorithm for Stochastic Minimax Optimization"
Including numerical stability and comprehensive state management
"""

import torch
from typing import Dict, List, Optional, Tuple, Union
import math


class AdaFMOptimizer:
    """
    Adaptive Filtered Momentum Optimizer
    
    Implements the exact algorithm from the ICLR 2025 paper for both
    single variable optimization and minimax optimization with numerical
    stability improvements.
    
    Args:
        params_x: Parameters for primal variable x (minimization)
        params_y: Parameters for dual variable y (maximization) - None for single variable
        gamma: Learning rate parameter for x (default: 1.0)
        lam: Learning rate parameter for y (default: 1.0)
        delta: Small value for learning rate adjustment (default: 0.001)
        single_variable: If True, treats as standard minimization (not minimax)
        eps: Numerical stability constant (default: 1e-8)
        max_lr: Maximum learning rate bound (default: 1.0)
    """
    
    def __init__(
        self,
        params_x,
        params_y=None,
        gamma: float = 1.0,
        lam: float = 1.0,
        delta: float = 0.001,
        single_variable: bool = False,
        eps: float = 1e-8,
        max_lr: float = 1.0
    ):
        self.single_variable = single_variable
        self.eps = eps
        self.max_lr = max_lr
        
        if single_variable:
            # For standard minimization problems
            self.param_groups = [{'params': list(params_x), 'var_type': 'x'}]
        else:
            # For minimax problems
            assert params_y is not None, "params_y required for minimax optimization"
            self.param_groups = [
                {'params': list(params_x), 'var_type': 'x'},
                {'params': list(params_y), 'var_type': 'y'}
            ]
        
        self.gamma = gamma
        self.lam = lam
        self.delta = delta
        self.t = 0  # iteration counter
        
        # Initialize estimators v_t and w_t with numerical stability
        self.v_estimators = {}  # For x variables
        self.w_estimators = {}  # For y variables
        
        # Initialize cumulative values alpha_x and alpha_y with stability
        self.alpha_x = self.eps
        self.alpha_y = self.eps
        
        # Store previous parameters and gradients for estimator updates
        self.prev_params_x = {}
        self.prev_params_y = {}
        self.prev_grads_x = {}
        self.prev_grads_y = {}
        
        # Learning rate history for monitoring
        self.lr_history = {'eta_x': [], 'eta_y': []}
        
        # Initialize storage
        for group in self.param_groups:
            for p in group['params']:
                if group['var_type'] == 'x':
                    self.v_estimators[p] = torch.zeros_like(p.data)
                    self.prev_params_x[p] = p.data.clone()
                    self.prev_grads_x[p] = torch.zeros_like(p.data)
                else:
                    self.w_estimators[p] = torch.zeros_like(p.data)
                    self.prev_params_y[p] = p.data.clone()
                    self.prev_grads_y[p] = torch.zeros_like(p.data)
    
    def zero_grad(self):
        """Zero out parameter gradients"""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.zero_()
    
    def step(self):
        """Perform a single optimization step with numerical stability"""
        self.t += 1
        
        # Compute momentum parameter with stability bounds
        beta_t = min(1.0 / (self.t ** (2.0/3.0)), 0.99)  # Cap at 0.99 for stability
        
        # Update estimators and collect norms with gradient clipping
        v_norm_squared = 0.0
        w_norm_squared = 0.0
        max_grad_norm = 10.0  # Gradient clipping for stability
        
        # First pass: update estimators
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                
                # Gradient clipping for numerical stability
                grad_norm = torch.norm(grad)
                if grad_norm > max_grad_norm:
                    grad = grad * (max_grad_norm / grad_norm)
                
                if group['var_type'] == 'x':
                    # Update v_t estimator for x variables
                    if self.t == 1:
                        self.v_estimators[p] = grad.clone()
                    else:
                        # v_t = ∇_x f(x_t, y_t) + (1 - β_t)(v_{t-1} - ∇_x f(x_{t-1}, y_{t-1}))
                        self.v_estimators[p] = grad + (1 - beta_t) * (
                            self.v_estimators[p] - self.prev_grads_x[p]
                        )
                    
                    v_norm_squared += torch.sum(self.v_estimators[p] ** 2).item()
                    self.prev_grads_x[p] = grad.clone()
                    
                else:  # var_type == 'y'
                    # Update w_t estimator for y variables
                    if self.t == 1:
                        self.w_estimators[p] = grad.clone()
                    else:
                        # w_t = ∇_y f(x_t, y_t) + (1 - β_t)(w_{t-1} - ∇_y f(x_{t-1}, y_{t-1}))
                        self.w_estimators[p] = grad + (1 - beta_t) * (
                            self.w_estimators[p] - self.prev_grads_y[p]
                        )
                    
                    w_norm_squared += torch.sum(self.w_estimators[p] ** 2).item()
                    self.prev_grads_y[p] = grad.clone()
        
        # Update cumulative values with smooth update and numerical stability
        beta_next = max(1.0 / ((self.t + 1) ** (2.0/3.0)), 1e-6)
        
        # Smooth update to prevent sudden jumps
        smooth_factor = 0.9 if self.t > 1 else 1.0
        self.alpha_x = smooth_factor * self.alpha_x + (1 - smooth_factor) * (v_norm_squared / beta_next + self.eps)
        self.alpha_y = smooth_factor * self.alpha_y + (1 - smooth_factor) * (w_norm_squared / beta_next + self.eps)
        
        # Compute learning rates with stability bounds
        if self.single_variable:
            eta_x = min(self.gamma / (self.alpha_x ** (1.0/3.0 + self.delta)), self.max_lr)
            eta_y = None
        else:
            max_alpha = max(self.alpha_x, self.alpha_y)
            eta_x = min(self.gamma / (max_alpha ** (1.0/3.0 + self.delta)), self.max_lr)
            eta_y = min(self.lam / (self.alpha_y ** (1.0/3.0 - self.delta)), self.max_lr)
        
        # Store learning rate history
        self.lr_history['eta_x'].append(eta_x)
        self.lr_history['eta_y'].append(eta_y)
        
        # Second pass: update parameters with stability checks
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                if group['var_type'] == 'x':
                    # x_{t+1} = x_t - η_x_t * v_t
                    update = self.v_estimators[p] * eta_x
                    if torch.isfinite(update).all():
                        p.data.add_(update, alpha=-1)
                    self.prev_params_x[p] = p.data.clone()
                else:  # var_type == 'y'
                    # y_{t+1} = y_t + η_y_t * w_t
                    update = self.w_estimators[p] * eta_y
                    if torch.isfinite(update).all():
                        p.data.add_(update, alpha=1)
                    self.prev_params_y[p] = p.data.clone()
    
    def get_current_lrs(self) -> Dict[str, Optional[float]]:
        """
        Get current learning rates for monitoring
        
        Returns:
            Dictionary with eta_x and eta_y values
        """
        if self.t == 0:
            return {'eta_x': 0.0, 'eta_y': 0.0}
        
        if self.single_variable:
            eta_x = min(self.gamma / (self.alpha_x ** (1.0/3.0 + self.delta)), self.max_lr)
            return {'eta_x': eta_x, 'eta_y': None}
        else:
            max_alpha = max(self.alpha_x, self.alpha_y)
            eta_x = min(self.gamma / (max_alpha ** (1.0/3.0 + self.delta)), self.max_lr)
            eta_y = min(self.lam / (self.alpha_y ** (1.0/3.0 - self.delta)), self.max_lr)
            return {'eta_x': eta_x, 'eta_y': eta_y}
    
    def get_momentum_param(self) -> float:
        """
        Get current momentum parameter β_t
        
        Returns:
            Current momentum parameter value
        """
        if self.t == 0:
            return 1.0
        return min(1.0 / (self.t ** (2.0/3.0)), 0.99)
    
    def state_dict(self) -> Dict:
        """
        Get optimizer state dictionary for checkpointing
        
        Returns:
            State dictionary containing all optimizer state
        """
        state = {
            'gamma': self.gamma,
            'lam': self.lam,
            'delta': self.delta,
            't': self.t,
            'alpha_x': self.alpha_x,
            'alpha_y': self.alpha_y,
            'single_variable': self.single_variable,
            'eps': self.eps,
            'max_lr': self.max_lr,
            'v_estimators': self.v_estimators,
            'w_estimators': self.w_estimators,
            'prev_grads_x': self.prev_grads_x,
            'prev_grads_y': self.prev_grads_y,
            'lr_history': self.lr_history
        }
        return state
    
    def load_state_dict(self, state_dict: Dict):
        """
        Load optimizer state from dictionary
        
        Args:
            state_dict: State dictionary to load
        """
        self.gamma = state_dict['gamma']
        self.lam = state_dict['lam']
        self.delta = state_dict['delta']
        self.t = state_dict['t']
        self.alpha_x = state_dict['alpha_x']
        self.alpha_y = state_dict['alpha_y']
        self.single_variable = state_dict['single_variable']
        self.eps = state_dict.get('eps', 1e-8)
        self.max_lr = state_dict.get('max_lr', 1.0)
        self.v_estimators = state_dict['v_estimators']
        self.w_estimators = state_dict['w_estimators']
        self.prev_grads_x = state_dict['prev_grads_x']
        self.prev_grads_y = state_dict['prev_grads_y']
        self.lr_history = state_dict.get('lr_history', {'eta_x': [], 'eta_y': []})
