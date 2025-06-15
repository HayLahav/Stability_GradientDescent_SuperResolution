"""
Unit tests for optimizers
"""

import pytest
import torch
import torch.nn as nn

from src.optimizers import AdaFMOptimizer, AdaptiveSGD
from src.optimizers.adaptive_sgd import polynomial_decay_lr


class SimpleModel(nn.Module):
    """Simple model for testing"""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.linear(x)


class TestAdaFMOptimizer:
    """Test AdaFM optimizer"""
    
    def test_single_variable_init(self):
        """Test single variable initialization"""
        model = SimpleModel()
        optimizer = AdaFMOptimizer(
            model.parameters(),
            gamma=1.0,
            delta=0.001,
            single_variable=True
        )
        assert optimizer.single_variable == True
        assert optimizer.gamma == 1.0
        assert optimizer.delta == 0.001
        assert optimizer.t == 0
    
    def test_minimax_init(self):
        """Test minimax initialization"""
        x = torch.randn(10, requires_grad=True)
        y = torch.randn(10, requires_grad=True)
        
        optimizer = AdaFMOptimizer(
            params_x=[x],
            params_y=[y],
            gamma=1.0,
            lam=1.0,
            delta=0.001,
            single_variable=False
        )
        assert optimizer.single_variable == False
        assert len(optimizer.param_groups) == 2
    
    def test_step_single_variable(self):
        """Test optimization step for single variable"""
        model = SimpleModel()
        optimizer = AdaFMOptimizer(
            model.parameters(),
            gamma=1.0,
            delta=0.001,
            single_variable=True
        )
        
        # Generate dummy data
        x = torch.randn(5, 10)
        y = torch.randn(5, 1)
        
        # Forward and backward
        loss = nn.MSELoss()(model(x), y)
        loss.backward()
        
        # Check initial state
        assert optimizer.t == 0
        assert optimizer.alpha_x == 0.0
        
        # Take step
        optimizer.step()
        
        # Check updated state
        assert optimizer.t == 1
        assert optimizer.alpha_x > 0
    
    def test_momentum_parameter(self):
        """Test momentum parameter computation"""
        model = SimpleModel()
        optimizer = AdaFMOptimizer(model.parameters())
        
        # Initial momentum
        assert optimizer.get_momentum_param() == 1.0
        
        # After steps
        optimizer.t = 8
        beta = optimizer.get_momentum_param()
        assert abs(beta - 0.25) < 0.001  # 1/8^(2/3) = 0.25
    
    def test_learning_rates(self):
        """Test learning rate computation"""
        model = SimpleModel()
        optimizer = AdaFMOptimizer(model.parameters())
        
        # Initial learning rates
        lrs = optimizer.get_current_lrs()
        assert lrs['eta_x'] == 0.0
        assert lrs['eta_y'] is None  # Single variable
        
        # After accumulating some alpha
        optimizer.alpha_x = 1000.0
        lrs = optimizer.get_current_lrs()
        assert lrs['eta_x'] > 0
        assert lrs['eta_x'] < 1.0
    
    def test_state_dict(self):
        """Test state dictionary save/load"""
        model = SimpleModel()
        optimizer = AdaFMOptimizer(model.parameters())
        
        # Take a step
        x = torch.randn(5, 10)
        y = torch.randn(5, 1)
        loss = nn.MSELoss()(model(x), y)
        loss.backward()
        optimizer.step()
        
        # Save state
        state = optimizer.state_dict()
        
        # Create new optimizer and load state
        optimizer2 = AdaFMOptimizer(model.parameters())
        optimizer2.load_state_dict(state)
        
        assert optimizer2.t == optimizer.t
        assert optimizer2.alpha_x == optimizer.alpha_x


class TestAdaptiveSGD:
    """Test adaptive SGD"""
    
    def test_initialization(self):
        """Test initialization"""
        model = SimpleModel()
        optimizer = AdaptiveSGD(
            model.parameters(),
            lr=0.01,
            momentum=0.9
        )
        assert optimizer.defaults['lr'] == 0.01
        assert optimizer.defaults['momentum'] == 0.9
    
    def test_step(self):
        """Test optimization step"""
        model = SimpleModel()
        optimizer = AdaptiveSGD(
            model.parameters(),
            lr=0.01,
            momentum=0.9
        )
        
        # Generate dummy data
        x = torch.randn(5, 10)
        y = torch.randn(5, 1)
        
        # Forward and backward
        loss = nn.MSELoss()(model(x), y)
        loss.backward()
        
        # Take step
        optimizer.step()
        
        # Check iteration counter
        assert optimizer.t == 1
    
    def test_lr_schedule(self):
        """Test learning rate schedule"""
        model = SimpleModel()
        lr_schedule = polynomial_decay_lr(0.1, power=0.5)
        
        optimizer = AdaptiveSGD(
            model.parameters(),
            lr=0.1,
            lr_schedule=lr_schedule
        )
        
        # Check learning rates at different iterations
        optimizer.t = 0
        assert optimizer.get_lr() == 0.1
        
        optimizer.t = 99  # t=100
        lr = optimizer.get_lr()
        expected = 0.1 / 10  # 0.1 / sqrt(100)
        assert abs(lr - expected) < 0.001