"""
Stability analysis tools and theoretical bounds
"""

# Import core analyzer first
from .analyzer import StabilityAnalyzer

# Import theoretical bounds functions
from .theoretical_bounds import (
    compute_strongly_convex_bound,
    compute_general_bound,
    compute_smooth_bound,
    compute_time_varying_bound,
    compute_regularized_bound,
    compute_optimal_iterations,
    compute_optimal_learning_rate,
    analyze_price_of_stability
)

# Import metrics functions
from .metrics import (
    compute_parameter_distance,
    compute_empirical_gamma,
    compute_gradient_variance,
    compute_generalization_gap,
    track_parameter_trajectory,
    compute_trajectory_smoothness,
    compute_hessian_eigenvalues
)

__all__ = [
    # Core analyzer
    'StabilityAnalyzer',
    
    # Theoretical bounds
    'compute_strongly_convex_bound',
    'compute_general_bound',
    'compute_smooth_bound',
    'compute_time_varying_bound',
    'compute_regularized_bound',
    'compute_optimal_iterations',
    'compute_optimal_learning_rate',
    'analyze_price_of_stability',
    
    # Metrics
    'compute_parameter_distance',
    'compute_empirical_gamma',
    'compute_gradient_variance',
    'compute_generalization_gap',
    'track_parameter_trajectory',
    'compute_trajectory_smoothness',
    'compute_hessian_eigenvalues'
]
