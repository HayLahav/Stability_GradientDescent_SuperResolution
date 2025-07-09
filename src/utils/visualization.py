"""
Visualization utilities for stability analysis
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
import seaborn as sns


def plot_stability_analysis(
    histories: Dict[str, Dict],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10)
):
    """
    Plot comprehensive stability analysis results
    
    Args:
        histories: Dictionary of training histories
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    # Plot 1: Training and Test Loss
    ax = axes[0, 0]
    for name, history in histories.items():
        ax.plot(history['train_loss'], label=f'{name} (train)', linestyle='-')
        if 'val_loss' in history:
            ax.plot(history['val_loss'], label=f'{name} (val)', linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Parameter Distance
    ax = axes[0, 1]
    for name, history in histories.items():
        if 'parameter_distance' in history and history['parameter_distance']:
            ax.plot(history['parameter_distance'], label=name)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('||w - w\'||')
    ax.set_title('Parameter Distance (Stability)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Empirical vs Theoretical Gamma
    ax = axes[0, 2]
    for name, history in histories.items():
        if 'empirical_gamma' in history and history['empirical_gamma']:
            ax.semilogy(history['empirical_gamma'], label=f'{name} (empirical)', linestyle='-')
        if 'theoretical_gamma' in history and history['theoretical_gamma']:
            ax.semilogy(history['theoretical_gamma'], label=f'{name} (theoretical)', linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('γ(m)')
    ax.set_title('Stability Bound γ(m)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Generalization Gap
    ax = axes[1, 0]
    for name, history in histories.items():
        if 'val_loss' in history:
            gen_gap = [val - train for train, val in 
                      zip(history['train_loss'], history['val_loss'])]
            ax.plot(gen_gap, label=name)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss - Train Loss')
    ax.set_title('Generalization Gap')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Learning Rate Evolution (if available)
    ax = axes[1, 1]
    for name, history in histories.items():
        if 'learning_rate' in history and history['learning_rate']:
            ax.semilogy(history['learning_rate'], label=name)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Momentum Parameter (if available)
    ax = axes[1, 2]
    for name, history in histories.items():
        if 'momentum_param' in history and history['momentum_param']:
            ax.plot(history['momentum_param'], label=name)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('β(t)')
    ax.set_title('Momentum Parameter')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_convergence_curves(
    losses: Dict[str, List[float]],
    title: str = "Convergence Curves",
    save_path: Optional[str] = None
):
    """Plot convergence curves for different methods"""
    plt.figure(figsize=(10, 6))
    
    for method, loss_values in losses.items():
        plt.semilogy(loss_values, label=method, linewidth=2)
    
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_sample_images(
    lr_images: torch.Tensor,
    sr_images: torch.Tensor,
    hr_images: torch.Tensor,
    num_samples: int = 5,
    save_path: Optional[str] = None
):
    """
    Plot sample LR, SR, and HR images
    
    Args:
        lr_images: Low-resolution images [B, C, H, W]
        sr_images: Super-resolved images [B, C, H, W]
        hr_images: High-resolution images [B, C, H, W]
        num_samples: Number of samples to show
        save_path: Path to save figure
    """
    num_samples = min(num_samples, lr_images.shape[0])
    
    fig, axes = plt.subplots(3, num_samples, figsize=(3*num_samples, 9))
    
    if num_samples == 1:
        axes = axes.reshape(3, 1)
    
    for i in range(num_samples):
        # LR image
        lr_img = lr_images[i].cpu().permute(1, 2, 0).numpy()
        lr_img = np.clip(lr_img, 0, 1)
        axes[0, i].imshow(lr_img)
        axes[0, i].set_title(f'LR {i+1}')
        axes[0, i].axis('off')
        
        # SR image
        sr_img = sr_images[i].cpu().permute(1, 2, 0).numpy()
        sr_img = np.clip(sr_img, 0, 1)
        axes[1, i].imshow(sr_img)
        axes[1, i].set_title(f'SR {i+1}')
        axes[1, i].axis('off')
        
        # HR image
        hr_img = hr_images[i].cpu().permute(1, 2, 0).numpy()
        hr_img = np.clip(hr_img, 0, 1)
        axes[2, i].imshow(hr_img)
        axes[2, i].set_title(f'HR {i+1}')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_theoretical_bounds(
    T_range: np.ndarray,
    m: int,
    L: float,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    save_path: Optional[str] = None
):
    """Plot theoretical stability bounds for different cases"""
    from ..stability.theoretical_bounds import (
        compute_strongly_convex_bound,
        compute_general_bound,
        compute_smooth_bound
    )
    
    plt.figure(figsize=(10, 6))
    
    # General case
    eta_general = np.sqrt(m) / (L * np.sqrt(T_range))
    gamma_general = compute_general_bound(T_range, m, eta_general, L)
    plt.loglog(T_range, gamma_general, 'b-', label='General Case', linewidth=2)
    
    # Strongly convex case
    if alpha is not None:
        gamma_sc = compute_strongly_convex_bound(T_range, m, 1.0, L, alpha)
        plt.loglog(T_range, gamma_sc, 'r--', label='Strongly Convex', linewidth=2)
    
    # Smooth case
    if beta is not None:
        eta_smooth = 1.0 / beta
        gamma_smooth = compute_smooth_bound(T_range, m, eta_smooth, L, beta)
        plt.loglog(T_range, gamma_smooth, 'g-.', label='Smooth', linewidth=2)
    
    plt.xlabel('Number of Iterations (T)')
    plt.ylabel('Stability Bound γ(m)')
    plt.title('Theoretical Stability Bounds')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def create_figure_grid(
    figures: List[plt.Figure],
    titles: List[str],
    cols: int = 2,
    figsize: Optional[Tuple[int, int]] = None
):
    """Create a grid of existing figures"""
    n_figs = len(figures)
    rows = (n_figs + cols - 1) // cols
    
    if figsize is None:
        figsize = (6 * cols, 5 * rows)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if n_figs > 1 else [axes]
    
    for i, (subfig, title) in enumerate(zip(figures, titles)):
        if i < len(axes):
            # Copy content from subfigure
            axes[i].set_title(title)
            # This is simplified - in practice, you'd copy the actual plot data
    
    # Hide unused subplots
    for i in range(n_figs, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig
