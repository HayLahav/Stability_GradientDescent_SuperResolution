"""
Adaptive SGD optimizer for comparison
"""

import torch
from torch.optim.optimizer import Optimizer
from typing import Optional, Callable


class AdaptiveSGD(Optimizer):
    """
    SGD with adaptive learning rate schedules
    
    Args:
        params: Model parameters
        lr: Base learning rate
        momentum: Momentum factor
        weight_decay: Weight decay (L2 penalty)
        lr_schedule: Learning rate schedule function
    """
    
    def __init__(
        self,
        params,
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        lr_schedule: Optional[Callable] = None
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            lr_schedule=lr_schedule
        )
        super().__init__(params, defaults)
        
        self.t = 0  # iteration counter
        
        # Initialize momentum buffers
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if momentum > 0:
                    state['momentum_buffer'] = torch.zeros_like(p.data)
    
    def step(self, closure=None):
        """Perform a single optimization step"""
        loss = None
        if closure is not None:
            loss = closure()
        
        self.t += 1
        
        for group in self.param_groups:
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            
            # Compute adaptive learning rate
            if group['lr_schedule'] is not None:
                lr = group['lr_schedule'](self.t)
            else:
                lr = group['lr']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                
                # Add weight decay
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)
                
                # Apply momentum
                if momentum != 0:
                    state = self.state[p]
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(grad, alpha=1 - momentum)
                    grad = buf
                
                # Update parameters
                p.data.add_(grad, alpha=-lr)
        
        return loss
    
    def get_lr(self) -> float:
        """Get current learning rate"""
        group = self.param_groups[0]
        if group['lr_schedule'] is not None:
            return group['lr_schedule'](self.t)
        return group['lr']


# Common learning rate schedules
def constant_lr(base_lr: float):
    """Constant learning rate"""
    return lambda t: base_lr


def step_decay_lr(base_lr: float, decay_rate: float, decay_steps: int):
    """Step decay learning rate"""
    return lambda t: base_lr * (decay_rate ** (t // decay_steps))


def exponential_decay_lr(base_lr: float, decay_rate: float):
    """Exponential decay learning rate"""
    return lambda t: base_lr * (decay_rate ** t)


def inverse_time_decay_lr(base_lr: float, decay_rate: float = 1.0):
    """Inverse time decay: lr = base_lr / (1 + decay_rate * t)"""
    return lambda t: base_lr / (1 + decay_rate * t)


def polynomial_decay_lr(base_lr: float, power: float = 0.5):
    """Polynomial decay: lr = base_lr / t^power"""
    return lambda t: base_lr / ((t + 1) ** power)
