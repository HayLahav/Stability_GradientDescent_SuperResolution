"""
Minimax optimization demonstration with AdaFM
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.optimizers import AdaFMOptimizer


class MinimaxProblem:
    """Simple minimax problem for demonstration"""
    
    def __init__(self, dim: int = 20):
        self.dim = dim
        # Random matrix for bilinear term
        self.A = torch.randn(dim, dim) * 0.5
        # Make problem more interesting
        self.A = (self.A + self.A.T) / 2  # Symmetric component
        
    def f(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute f(x,y) = x^T A y + ||x||^2/2 - ||y||^2/2"""
        bilinear = torch.dot(x, self.A @ y)
        reg_x = 0.5 * torch.norm(x)**2
        reg_y = 0.5 * torch.norm(y)**2
        return bilinear + reg_x - reg_y
    
    def grad_x(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Gradient with respect to x"""
        return self.A @ y + x
    
    def grad_y(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Gradient with respect to y"""
        return self.A.T @ x - y


def run_minimax_comparison():
    """Compare AdaFM with standard GDA on minimax problem"""
    
    print("=== AdaFM Minimax Optimization Demo ===\n")
    
    # Initialize problem
    problem = MinimaxProblem(dim=50)
    
    # Methods to compare
    methods = {
        'GDA (lr_x=0.01, lr_y=0.02)': {
            'use_adafm': False,
            'lr_x': 0.01,
            'lr_y': 0.02
        },
        'GDA (lr_x=0.005, lr_y=0.01)': {
            'use_adafm': False,
            'lr_x': 0.005,
            'lr_y': 0.01
        },
        'AdaFM (γ=1.0, λ=1.0)': {
            'use_adafm': True,
            'gamma': 1.0,
            'lam': 1.0,
            'delta': 0.001
        }
    }
    
    # Run experiments
    results = {}
    num_iterations = 500
    
    for method_name, config in methods.items():
        print(f"\nRunning {method_name}...")
        
        # Initialize variables
        x = torch.randn(problem.dim, requires_grad=True)
        y = torch.randn(problem.dim, requires_grad=True)
        
        # Initialize optimizer
        if config['use_adafm']:
            optimizer = AdaFMOptimizer(
                params_x=[x],
                params_y=[y],
                gamma=config['gamma'],
                lam=config['lam'],
                delta=config['delta'],
                single_variable=False
            )
        else:
            # Standard GDA
            optimizer_x = torch.optim.SGD([x], lr=config['lr_x'])
            optimizer_y = torch.optim.SGD([y], lr=config['lr_y'])
        
        # Training history
        history = {
            'f_values': [],
            'x_norms': [],
            'y_norms': [],
            'grad_x_norms': [],
            'grad_y_norms': [],
            'lr_x': [],
            'lr_y': []
        }
        
        # Optimization loop
        for t in range(num_iterations):
            # Compute function value
            f_val = problem.f(x, y)
            history['f_values'].append(f_val.item())
            history['x_norms'].append(torch.norm(x).item())
            history['y_norms'].append(torch.norm(y).item())
            
            # Compute gradients
            x.grad = problem.grad_x(x, y)
            y.grad = -problem.grad_y(x, y)  # Negative for maximization
            
            history['grad_x_norms'].append(torch.norm(x.grad).item())
            history['grad_y_norms'].append(torch.norm(y.grad).item())
            
            # Update parameters
            if config['use_adafm']:
                optimizer.step()
                optimizer.zero_grad()
                
                # Record adaptive learning rates
                current_lrs = optimizer.get_current_lrs()
                history['lr_x'].append(current_lrs['eta_x'])
                history['lr_y'].append(current_lrs['eta_y'])
            else:
                optimizer_x.step()
                optimizer_y.step()
                optimizer_x.zero_grad()
                optimizer_y.zero_grad()
                
                history['lr_x'].append(config['lr_x'])
                history['lr_y'].append(config['lr_y'])
        
        results[method_name] = history
        
        # Print final statistics
        print(f"  Final f(x,y): {history['f_values'][-1]:.6f}")
        print(f"  Final ||∇x||: {history['grad_x_norms'][-1]:.6f}")
        print(f"  Final ||∇y||: {history['grad_y_norms'][-1]:.6f}")
    
    return results


def plot_minimax_results(results: dict):
    """Plot results of minimax optimization"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Function values
    ax = axes[0, 0]
    for name, history in results.items():
        ax.plot(history['f_values'], label=name, linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('f(x,y)')
    ax.set_title('Function Value Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gradient norms
    ax = axes[0, 1]
    for name, history in results.items():
        ax.semilogy(history['grad_x_norms'], label=f'{name} ||∇x||', linewidth=2)
        ax.semilogy(history['grad_y_norms'], '--', label=f'{name} ||∇y||', linewidth=1)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Gradient Norm')
    ax.set_title('Gradient Norms')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Variable norms
    ax = axes[0, 2]
    for name, history in results.items():
        ax.plot(history['x_norms'], label=f'{name} ||x||', linewidth=2)
        ax.plot(history['y_norms'], '--', label=f'{name} ||y||', linewidth=1)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Variable Norm')
    ax.set_title('Variable Norms')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Learning rates (x)
    ax = axes[1, 0]
    for name, history in results.items():
        ax.semilogy(history['lr_x'], label=name, linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('η_x')
    ax.set_title('Learning Rate for x')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Learning rates (y)
    ax = axes[1, 1]
    for name, history in results.items():
        ax.semilogy(history['lr_y'], label=name, linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('η_y')
    ax.set_title('Learning Rate for y')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Learning rate ratio
    ax = axes[1, 2]
    for name, history in results.items():
        ratio = [lr_y/lr_x for lr_x, lr_y in zip(history['lr_x'], history['lr_y'])]
        ax.plot(ratio, label=name, linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('η_y / η_x')
    ax.set_title('Learning Rate Ratio')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/figures/adafm_minimax_experiment.png', dpi=300, bbox_inches='tight')
    plt.show()


def test_function_minimax():
    """Test on toy function from AdaFM paper"""
    
    print("\n=== Toy Function Test ===")
    
    # f(x,y) = 1/2 * y^2 + Lxy - L^2/2 * x^2
    L = 2.0
    
    def f(x, y):
        return 0.5 * y**2 + L * x * y - 0.5 * L**2 * x**2
    
    def grad_x(x, y):
        return L * y - L**2 * x
    
    def grad_y(x, y):
        return y + L * x
    
    # Initialize
    x = torch.tensor(0.1, requires_grad=True)
    y = torch.tensor(0.0, requires_grad=True)
    
    # AdaFM optimizer
    optimizer = AdaFMOptimizer(
        params_x=[x],
        params_y=[y],
        gamma=5.0,
        lam=1.0,
        delta=0.1,
        single_variable=False
    )
    
    trajectory = {'x': [x.item()], 'y': [y.item()]}
    
    # Run optimization
    for _ in range(100):
        x.grad = grad_x(x, y)
        y.grad = -grad_y(x, y)
        
        optimizer.step()
        optimizer.zero_grad()
        
        trajectory['x'].append(x.item())
        trajectory['y'].append(y.item())
    
    # Plot trajectory
    plt.figure(figsize=(8, 6))
    plt.plot(trajectory['x'], trajectory['y'], 'b-', linewidth=2, label='AdaFM')
    plt.plot(trajectory['x'][0], trajectory['y'][0], 'go', markersize=10, label='Start')
    plt.plot(trajectory['x'][-1], trajectory['y'][-1], 'ro', markersize=10, label='End')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Optimization Trajectory on f(x,y) = y²/2 + {L}xy - {L}²x²/2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.savefig('results/figures/toy_function_trajectory.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Final point: x={x.item():.4f}, y={y.item():.4f}")
    print(f"Function value: {f(x, y).item():.6f}")


def main():
    """Run minimax demonstrations"""
    
    # Create output directory
    Path('results/figures').mkdir(parents=True, exist_ok=True)
    
    # Run main comparison
    results = run_minimax_comparison()
    plot_minimax_results(results)
    
    # Test on toy function
    test_function_minimax()
    
    print("\n=== Minimax demonstration completed! ===")


if __name__ == '__main__':
    main()