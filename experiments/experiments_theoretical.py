"""
Theoretical validation experiments
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path

from src.stability.theoretical_bounds import (
    compute_strongly_convex_bound,
    compute_general_bound,
    compute_smooth_bound,
    compute_time_varying_bound,
    analyze_price_of_stability
)


def validate_theoretical_bounds():
    """Validate theoretical stability bounds from Lecture 8"""
    
    print("=== Validating Theoretical Bounds ===\n")
    
    # Parameters
    L = 1.0  # Lipschitz constant
    alpha = 0.1  # Strong convexity parameter
    beta = 1.0  # Smoothness parameter
    m = 1000  # Sample size
    
    # Test different scenarios
    T_values = [10, 100, 1000, 10000]
    eta_values = [0.1, 0.01, 0.001]
    
    print("1. Strongly Convex Case (Theorem 8.1):")
    print("   γ(m) = O(L²/(α√T) + L²/(αm))")
    for T in T_values:
        gamma = compute_strongly_convex_bound(T, m, 1.0, L, alpha)
        print(f"   T={T:5d}: γ(m) = {gamma:.6f}")
    
    print("\n2. General Case (Theorem 8.3):")
    print("   γ(m) = 4ηL²√T + 4ηL²T/m")
    for eta in eta_values:
        print(f"   η={eta}:")
        for T in T_values:
            gamma = compute_general_bound(T, m, eta, L)
            print(f"     T={T:5d}: γ(m) = {gamma:.6f}")
    
    print("\n3. Smooth Case (Theorem 8.5):")
    print("   γ(m) = 2ηTL²/m")
    eta = 1.0 / beta  # Optimal for smooth case
    for T in T_values:
        gamma = compute_smooth_bound(T, m, eta, L, beta)
        print(f"   T={T:5d}: γ(m) = {gamma:.6f}")
    
    # Visualize convergence rates
    plt.figure(figsize=(15, 5))
    T_range = np.logspace(1, 5, 100)
    
    # Strongly convex
    plt.subplot(1, 3, 1)
    gamma_sc = compute_strongly_convex_bound(T_range, m, 1.0, L, alpha)
    opt_error_sc = L**2 / (alpha * T_range)
    plt.loglog(T_range, gamma_sc, 'b-', label='Stability Bound', linewidth=2)
    plt.loglog(T_range, opt_error_sc, 'r--', label='Optimization Error', linewidth=2)
    plt.xlabel('Iterations (T)')
    plt.ylabel('Error')
    plt.title('Strongly Convex Case')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # General case
    plt.subplot(1, 3, 2)
    eta_opt = np.sqrt(m) / (L * T_range)
    gamma_gen = compute_general_bound(T_range, m, eta_opt, L)
    opt_error_gen = L**2 / np.sqrt(T_range)
    plt.loglog(T_range, gamma_gen, 'b-', label='Stability Bound', linewidth=2)
    plt.loglog(T_range, opt_error_gen, 'r--', label='Optimization Error', linewidth=2)
    plt.xlabel('Iterations (T)')
    plt.ylabel('Error')
    plt.title('General Case')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Smooth case
    plt.subplot(1, 3, 3)
    gamma_smooth = compute_smooth_bound(T_range, m, 1/beta, L, beta)
    opt_error_smooth = beta * L**2 / T_range
    plt.loglog(T_range, gamma_smooth, 'b-', label='Stability Bound', linewidth=2)
    plt.loglog(T_range, opt_error_smooth, 'r--', label='Optimization Error', linewidth=2)
    plt.xlabel('Iterations (T)')
    plt.ylabel('Error')
    plt.title('Smooth Case')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/figures/theoretical_bounds_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def validate_time_varying_stepsizes():
    """Validate stability with time-varying step sizes"""
    
    print("\n=== Time-Varying Step Sizes ===")
    
    # Different step size schedules
    schedules = {
        'Constant': lambda t: 0.01,
        'Decreasing (1/t)': lambda t: 0.1 / (t + 1),
        'Decreasing (1/√t)': lambda t: 0.1 / np.sqrt(t + 1),
        'Exponential': lambda t: 0.1 * 0.99**t
    }
    
    T = 1000
    m = 1000
    L = 1.0
    
    for name, schedule in schedules.items():
        gamma = compute_time_varying_bound(T, m, schedule, L)
        sum_eta_sq = sum(schedule(t)**2 for t in range(T))
        print(f"\n{name}:")
        print(f"  Σ η(t)² = {sum_eta_sq:.6f}")
        print(f"  γ(m) = {gamma:.6f}")


def analyze_price_of_stability_examples():
    """Analyze the price of stability for different error targets"""
    
    print("\n=== Price of Stability Analysis ===")
    
    epsilons = [0.1, 0.01, 0.001, 0.0001]
    L = 1.0
    alpha = 0.1
    
    results = []
    for eps in epsilons:
        analysis = analyze_price_of_stability(eps, L, alpha)
        results.append(analysis)
        
        print(f"\nε = {eps}:")
        print(f"  Optimization iterations: {analysis['optimization_iterations']}")
        print(f"  Generalization iterations: {analysis['generalization_iterations']}")
        print(f"  Price ratio: {analysis['price_ratio']:.2f}x")
    
    # Visualize price of stability
    plt.figure(figsize=(10, 6))
    
    eps_array = np.array(epsilons)
    opt_iters = np.array([r['optimization_iterations'] for r in results])
    gen_iters = np.array([r['generalization_iterations'] for r in results])
    
    plt.loglog(eps_array, opt_iters, 'b-o', label='Optimization (T = O(1/ε))', linewidth=2, markersize=8)
    plt.loglog(eps_array, gen_iters, 'r-s', label='Generalization (T = O(1/ε²))', linewidth=2, markersize=8)
    
    # Add reference lines
    plt.loglog(eps_array, 1/eps_array, 'b--', alpha=0.5, label='1/ε')
    plt.loglog(eps_array, 1/eps_array**2, 'r--', alpha=0.5, label='1/ε²')
    
    plt.xlabel('Target Error (ε)')
    plt.ylabel('Required Iterations')
    plt.title('The Price of Stability: Optimization vs Generalization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('results/figures/price_of_stability.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Run all theoretical validation experiments"""
    
    # Create output directory
    Path('results/figures').mkdir(parents=True, exist_ok=True)
    
    # Run validations
    validate_theoretical_bounds()
    validate_time_varying_stepsizes()
    analyze_price_of_stability_examples()
    
    print("\n=== Theoretical validation completed! ===")


if __name__ == '__main__':
    main()