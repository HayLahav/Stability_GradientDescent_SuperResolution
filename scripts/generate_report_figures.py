"""
Generate all figures for the project report
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

from src.utils import plot_stability_analysis


def load_all_histories():
    """Load training histories from all experiments"""
    histories = {}
    
    result_dirs = [
        'results/Baseline',
        'results/WithCorrection',
        'results/WithAdaFMOpt',
        'results/FullSystem'
    ]
    
    for result_dir in result_dirs:
        history_path = Path(result_dir) / 'history.pt'
        if history_path.exists():
            name = Path(result_dir).name
            histories[name] = torch.load(history_path)
            print(f"Loaded history for {name}")
    
    return histories


def create_comparison_table(histories):
    """Create comparison table of final metrics"""
    
    print("\n=== Final Metrics Comparison ===")
    print(f"{'Configuration':<20} {'Train Loss':<12} {'Val Loss':<12} {'Gen. Gap':<12} {'Final γ(m)':<12}")
    print("-" * 70)
    
    for name, history in histories.items():
        train_loss = history['train_loss'][-1]
        val_loss = history.get('val_loss', [0])[-1]
        gen_gap = val_loss - train_loss if val_loss else 0
        
        if 'empirical_gamma' in history and history['empirical_gamma']:
            gamma = history['empirical_gamma'][-1]
        else:
            gamma = 0
        
        print(f"{name:<20} {train_loss:<12.4f} {val_loss:<12.4f} {gen_gap:<12.4f} {gamma:<12.4f}")


def plot_combined_analysis(histories):
    """Create combined stability analysis plot"""
    
    # Create main comparison plot
    plot_stability_analysis(
        histories,
        save_path='results/figures/combined_stability_analysis.png',
        figsize=(18, 12)
    )
    
    # Create learning curves only
    plt.figure(figsize=(10, 6))
    for name, history in histories.items():
        plt.plot(history['train_loss'], label=f'{name} (train)', linewidth=2)
        if 'val_loss' in history:
            plt.plot(history['val_loss'], '--', label=f'{name} (val)', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss Comparison', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/figures/learning_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create stability comparison
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for name, history in histories.items():
        if 'parameter_distance' in history and history['parameter_distance']:
            plt.plot(history['parameter_distance'], label=name, linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('||w - w\'||', fontsize=12)
    plt.title('Parameter Distance Evolution', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    for name, history in histories.items():
        if 'empirical_gamma' in history and history['empirical_gamma']:
            plt.semilogy(history['empirical_gamma'], label=name, linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('γ(m)', fontsize=12)
    plt.title('Empirical Stability Bound', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)