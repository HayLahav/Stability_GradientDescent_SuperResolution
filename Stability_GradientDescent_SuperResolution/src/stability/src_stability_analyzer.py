"""
Stability analyzer for gradient descent algorithms
Based on Lecture 8: Stability Analysis for Gradient Descent
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Callable
import numpy as np

from .theoretical_bounds import (
    compute_strongly_convex_bound,
    compute_general_bound,
    compute_smooth_bound
)
from .metrics import (
    compute_parameter_distance,
    compute_empirical_gamma
)


class StabilityAnalyzer:
    """
    Analyzes the stability of gradient descent algorithms
    
    Args:
        model: Neural network model
        loss_fn: Loss function
        L: Lipschitz constant
        alpha: Strong convexity parameter (if applicable)
        beta: Smoothness parameter (if applicable)
    """
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable,
        L: float = 1.0,
        alpha: Optional[float] = None,
        beta: Optional[float] = None
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.L = L
        self.alpha = alpha
        self.beta = beta
        
        # Storage for analysis
        self.parameter_distances = []
        self.empirical_gammas = []
        self.theoretical_gammas = []
        
    def compute_parameter_distance(
        self,
        model1: nn.Module,
        model2: nn.Module
    ) -> float:
        """
        Compute ||w - w'|| between two models
        
        Args:
            model1: First model
            model2: Second model
            
        Returns:
            L2 distance between parameters
        """
        return compute_parameter_distance(model1, model2)
    
    def compute_empirical_gamma(
        self,
        model_S: nn.Module,
        model_S_prime: nn.Module,
        test_data: torch.Tensor
    ) -> float:
        """
        Compute empirical stability γ(m) = sup_z |f(w_S, z) - f(w_S', z)|
        
        Args:
            model_S: Model trained on S
            model_S_prime: Model trained on S'
            test_data: Test data for evaluation
            
        Returns:
            Empirical stability measure
        """
        return compute_empirical_gamma(model_S, model_S_prime, test_data)
    
    def compute_theoretical_gamma_strongly_convex(
        self,
        T: int,
        m: int,
        eta: float
    ) -> float:
        """
        Theoretical stability bound for strongly convex case
        
        Args:
            T: Number of iterations
            m: Sample size
            eta: Learning rate
            
        Returns:
            Theoretical stability bound
        """
        if self.alpha is None:
            raise ValueError("Strong convexity parameter α not specified")
        
        return compute_strongly_convex_bound(
            T, m, eta, self.L, self.alpha
        )
    
    def compute_theoretical_gamma_general(
        self,
        T: int,
        m: int,
        eta: float
    ) -> float:
        """
        Theoretical stability bound for general convex case
        
        Args:
            T: Number of iterations
            m: Sample size
            eta: Learning rate
            
        Returns:
            Theoretical stability bound
        """
        return compute_general_bound(T, m, eta, self.L)
    
    def compute_theoretical_gamma_smooth(
        self,
        T: int,
        m: int,
        eta: float
    ) -> float:
        """
        Theoretical stability bound for smooth case
        
        Args:
            T: Number of iterations
            m: Sample size
            eta: Learning rate
            
        Returns:
            Theoretical stability bound
        """
        if self.beta is None:
            raise ValueError("Smoothness parameter β not specified")
        
        return compute_smooth_bound(T, m, eta, self.L, self.beta)
    
    def analyze_trajectory(
        self,
        model_trajectory: List[nn.Module],
        model_prime_trajectory: List[nn.Module]
    ) -> Dict[str, List[float]]:
        """
        Analyze stability along training trajectory
        
        Args:
            model_trajectory: List of model checkpoints
            model_prime_trajectory: List of perturbed model checkpoints
            
        Returns:
            Dictionary of stability metrics over time
        """
        distances = []
        divergence_rates = []
        
        for t, (model_t, model_prime_t) in enumerate(
            zip(model_trajectory, model_prime_trajectory)
        ):
            dist = self.compute_parameter_distance(model_t, model_prime_t)
            distances.append(dist)
            
            if t > 0:
                rate = (distances[t] - distances[t-1]) / distances[t-1]
                divergence_rates.append(rate)
        
        return {
            'distances': distances,
            'divergence_rates': divergence_rates
        }
    
    def compute_stability_certificate(
        self,
        model: nn.Module,
        dataset_size: int,
        num_iterations: int,
        learning_rate: float,
        case: str = 'general'
    ) -> Dict[str, float]:
        """
        Compute stability certificate for given training configuration
        
        Args:
            model: Trained model
            dataset_size: Size of training dataset
            num_iterations: Number of training iterations
            learning_rate: Learning rate used
            case: Type of analysis ('strongly_convex', 'general', 'smooth')
            
        Returns:
            Dictionary with stability bounds and guarantees
        """
        if case == 'strongly_convex':
            gamma = self.compute_theoretical_gamma_strongly_convex(
                num_iterations, dataset_size, learning_rate
            )
        elif case == 'smooth':
            gamma = self.compute_theoretical_gamma_smooth(
                num_iterations, dataset_size, learning_rate
            )
        else:
            gamma = self.compute_theoretical_gamma_general(
                num_iterations, dataset_size, learning_rate
            )
        
        # Compute generalization bound
        # E[F(w_S)] - F(w*) ≤ γ(m) + ε(m)
        optimization_error = self.L**2 / (2 * num_iterations * learning_rate)
        generalization_bound = gamma + optimization_error
        
        return {
            'stability_parameter': gamma,
            'optimization_error': optimization_error,
            'generalization_bound': generalization_bound,
            'case': case,
            'iterations': num_iterations,
            'sample_size': dataset_size,
            'learning_rate': learning_rate
        }
    
    def plot_stability_analysis(
        self,
        save_path: Optional[str] = None
    ):
        """
        Plot stability analysis results
        
        Args:
            save_path: Path to save figure
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot parameter distances
        ax = axes[0, 0]
        if self.parameter_distances:
            ax.plot(self.parameter_distances)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('||w - w\'||')
            ax.set_title('Parameter Distance Evolution')
            ax.grid(True)
        
        # Plot empirical gamma
        ax = axes[0, 1]
        if self.empirical_gammas:
            ax.plot(self.empirical_gammas, label='Empirical')
            if self.theoretical_gammas:
                ax.plot(self.theoretical_gammas, '--', label='Theoretical')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('γ(m)')
            ax.set_title('Stability Parameter')
            ax.legend()
            ax.grid(True)
        
        # Plot convergence rate comparison
        ax = axes[1, 0]
        T_range = np.logspace(1, 5, 100)
        m = 1000
        eta = 0.01
        
        if self.alpha:
            gamma_sc = compute_strongly_convex_bound(
                T_range, m, eta, self.L, self.alpha
            )
            ax.loglog(T_range, gamma_sc, label='Strongly Convex')
        
        gamma_gen = compute_general_bound(T_range, m, eta, self.L)
        ax.loglog(T_range, gamma_gen, label='General Case')
        
        if self.beta:
            gamma_smooth = compute_smooth_bound(
                T_range, m, eta, self.L, self.beta
            )
            ax.loglog(T_range, gamma_smooth, label='Smooth Case')
        
        ax.set_xlabel('Iterations (T)')
        ax.set_ylabel('γ(m)')
        ax.set_title('Theoretical Bounds Comparison')
        ax.legend()
        ax.grid(True)
        
        # Plot sample complexity
        ax = axes[1, 1]
        m_range = np.logspace(2, 5, 100)
        T_fixed = 1000
        
        gamma_m = compute_general_bound(T_fixed, m_range, eta, self.L)
        ax.loglog(m_range, gamma_m)
        ax.set_xlabel('Sample Size (m)')
        ax.set_ylabel('γ(m)')
        ax.set_title('Stability vs Sample Size')
        ax.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()