"""
Theoretical stability bounds from Lecture 8
"""

import numpy as np
from typing import Union


def compute_strongly_convex_bound(
    T: Union[int, np.ndarray],
    m: Union[int, np.ndarray],
    eta: float,
    L: float,
    alpha: float
) -> Union[float, np.ndarray]:
    """
    Compute stability bound for strongly convex case (Theorem 8.1)
    γ(m) = O(L²/(α√T) + L²/(αm))
    
    Args:
        T: Number of iterations
        m: Sample size
        eta: Learning rate
        L: Lipschitz constant
        alpha: Strong convexity parameter
        
    Returns:
        Stability bound γ(m)
    """
    term1 = L**2 / (alpha * np.sqrt(T))
    term2 = L**2 / (alpha * m)
    return term1 + term2


def compute_general_bound(
    T: Union[int, np.ndarray],
    m: Union[int, np.ndarray],
    eta: float,
    L: float
) -> Union[float, np.ndarray]:
    """
    Compute stability bound for general convex case (Theorem 8.3)
    γ(m) = 4ηL²√T + 4ηL²T/m
    
    Args:
        T: Number of iterations
        m: Sample size
        eta: Learning rate
        L: Lipschitz constant
        
    Returns:
        Stability bound γ(m)
    """
    term1 = 4 * eta * L**2 * np.sqrt(T)
    term2 = 4 * eta * L**2 * T / m
    return term1 + term2


def compute_smooth_bound(
    T: Union[int, np.ndarray],
    m: Union[int, np.ndarray],
    eta: float,
    L: float,
    beta: float
) -> Union[float, np.ndarray]:
    """
    Compute stability bound for smooth case (Theorem 8.5)
    γ(m) = 2ηTL²/m
    
    Args:
        T: Number of iterations
        m: Sample size
        eta: Learning rate
        L: Lipschitz constant
        beta: Smoothness parameter
        
    Returns:
        Stability bound γ(m)
    """
    # For smooth case, optimal eta = 1/beta
    if eta > 2/beta:
        print(f"Warning: η={eta} > 2/β={2/beta}. Using η=1/β for stability.")
        eta = 1/beta
    
    return 2 * eta * T * L**2 / m


def compute_time_varying_bound(
    T: int,
    m: int,
    eta_schedule: callable,
    L: float
) -> float:
    """
    Compute stability bound for time-varying step sizes
    γ(m) ≤ (2L²/m)√(Σ η(t)²)
    
    Args:
        T: Number of iterations
        m: Sample size
        eta_schedule: Function that returns η(t) for iteration t
        L: Lipschitz constant
        
    Returns:
        Stability bound γ(m)
    """
    sum_eta_squared = sum(eta_schedule(t)**2 for t in range(T))
    return (2 * L**2 / m) * np.sqrt(sum_eta_squared)


def compute_regularized_bound(
    T: int,
    m: int,
    eta: float,
    L: float,
    lam: float,
    R: float
) -> float:
    """
    Compute stability bound with regularization (Theorem 8.2)
    
    Args:
        T: Number of iterations
        m: Sample size
        eta: Learning rate
        L: Lipschitz constant
        lam: Regularization parameter
        R: Bound on parameter norm
        
    Returns:
        Stability bound with regularization
    """
    # With regularization, effective strong convexity is λ
    term1 = lam * R**2
    term2 = L**2 / (lam * np.sqrt(T))
    term3 = L**2 / (lam * m)
    return term1 + term2 + term3


def compute_optimal_iterations(
    epsilon: float,
    m: int,
    L: float,
    case: str = 'general',
    alpha: float = None
) -> int:
    """
    Compute optimal number of iterations for ε-learning error
    
    Args:
        epsilon: Target learning error
        m: Sample size
        L: Lipschitz constant
        case: 'strongly_convex' or 'general'
        alpha: Strong convexity parameter (if applicable)
        
    Returns:
        Optimal number of iterations T
    """
    if case == 'strongly_convex' and alpha is not None:
        # For strongly convex: T = O(m²) for ε = O(1/m)
        T = int((L**2 / (alpha * epsilon))**2)
    else:
        # For general case: T = O(1/ε²)
        T = int(1 / epsilon**2)
    
    return T


def compute_optimal_learning_rate(
    T: int,
    m: int,
    L: float,
    R: float,
    case: str = 'general'
) -> float:
    """
    Compute optimal learning rate for given configuration
    
    Args:
        T: Number of iterations
        m: Sample size
        L: Lipschitz constant
        R: Bound on parameter norm
        case: Type of optimization problem
        
    Returns:
        Optimal learning rate
    """
    if case == 'strongly_convex':
        # η_t = 1/(αt) for strongly convex
        return 1.0  # Return base rate, actual schedule is 1/(αt)
    elif case == 'smooth':
        # η = 1/β for smooth case
        return 1.0  # Placeholder, should use 1/β
    else:
        # General case: η*T = O(√m)
        return np.sqrt(m) / (L * T)


def analyze_price_of_stability(
    epsilon: float,
    L: float = 1.0,
    alpha: float = 0.1
) -> dict:
    """
    Analyze the price of stability: optimization vs generalization
    
    Args:
        epsilon: Target error
        L: Lipschitz constant
        alpha: Strong convexity parameter
        
    Returns:
        Dictionary with iteration requirements
    """
    # Optimization error: T = O(1/ε)
    T_opt = int(L**2 / (alpha * epsilon))
    
    # Generalization error: T = O(1/ε²)
    T_gen = int(L**2 / (alpha * epsilon**2))
    
    return {
        'optimization_iterations': T_opt,
        'generalization_iterations': T_gen,
        'price_ratio': T_gen / T_opt,
        'epsilon': epsilon
    }