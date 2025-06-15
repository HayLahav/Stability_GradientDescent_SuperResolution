"""
Stability metrics and measurements
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional
import numpy as np


def compute_parameter_distance(
    model1: nn.Module,
    model2: nn.Module
) -> float:
    """
    Compute L2 distance between parameters of two models
    ||w - w'||
    
    Args:
        model1: First model
        model2: Second model
        
    Returns:
        L2 distance between parameters
    """
    distance = 0.0
    
    for (p1, p2) in zip(model1.parameters(), model2.parameters()):
        if p1.shape != p2.shape:
            raise ValueError(f"Parameter shapes don't match: {p1.shape} vs {p2.shape}")
        distance += torch.norm(p1 - p2, p=2).item() ** 2
    
    return np.sqrt(distance)


def compute_empirical_gamma(
    model_S: nn.Module,
    model_S_prime: nn.Module,
    test_data: torch.Tensor,
    batch_size: Optional[int] = None
) -> float:
    """
    Compute empirical stability measure
    γ(m) = sup_z |f(w_S, z) - f(w_S', z)|
    
    Args:
        model_S: Model trained on dataset S
        model_S_prime: Model trained on dataset S'
        test_data: Test data to evaluate on
        batch_size: Batch size for processing
        
    Returns:
        Empirical gamma value
    """
    model_S.eval()
    model_S_prime.eval()
    
    with torch.no_grad():
        if batch_size is None:
            # Process all at once
            output_S = model_S(test_data)
            output_S_prime = model_S_prime(test_data)
            
            # Compute supremum of absolute difference
            diff = torch.abs(output_S - output_S_prime)
            gamma = torch.max(diff).item()
        else:
            # Process in batches
            gamma = 0.0
            num_samples = test_data.shape[0]
            
            for i in range(0, num_samples, batch_size):
                batch = test_data[i:i+batch_size]
                output_S = model_S(batch)
                output_S_prime = model_S_prime(batch)
                
                diff = torch.abs(output_S - output_S_prime)
                batch_gamma = torch.max(diff).item()
                gamma = max(gamma, batch_gamma)
    
    return gamma


def compute_gradient_variance(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module
) -> float:
    """
    Compute variance of gradients across batches
    
    Args:
        model: Neural network model
        data_loader: Data loader for computing gradients
        loss_fn: Loss function
        
    Returns:
        Average gradient variance across parameters
    """
    model.train()
    
    # Storage for gradients
    gradient_samples = []
    
    for inputs, targets in data_loader:
        model.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        
        # Collect gradients
        grads = []
        for p in model.parameters():
            if p.grad is not None:
                grads.append(p.grad.view(-1).clone())
        
        if grads:
            all_grads = torch.cat(grads)
            gradient_samples.append(all_grads)
    
    if not gradient_samples:
        return 0.0
    
    # Stack all gradient samples
    gradient_matrix = torch.stack(gradient_samples)
    
    # Compute variance
    variance = torch.var(gradient_matrix, dim=0).mean().item()
    
    return variance


def compute_generalization_gap(
    train_loss: float,
    test_loss: float
) -> float:
    """
    Compute generalization gap
    
    Args:
        train_loss: Training loss
        test_loss: Test loss
        
    Returns:
        Generalization gap
    """
    return test_loss - train_loss


def track_parameter_trajectory(
    models: List[nn.Module]
) -> np.ndarray:
    """
    Track parameter trajectory over training
    
    Args:
        models: List of model checkpoints
        
    Returns:
        Array of parameter vectors [T, D]
    """
    trajectories = []
    
    for model in models:
        params = []
        for p in model.parameters():
            params.append(p.detach().view(-1))
        param_vector = torch.cat(params).cpu().numpy()
        trajectories.append(param_vector)
    
    return np.array(trajectories)


def compute_trajectory_smoothness(
    trajectory: np.ndarray
) -> float:
    """
    Compute smoothness of parameter trajectory
    
    Args:
        trajectory: Parameter trajectory [T, D]
        
    Returns:
        Average smoothness measure
    """
    if len(trajectory) < 2:
        return 0.0
    
    # Compute differences between consecutive points
    diffs = np.diff(trajectory, axis=0)
    
    # Compute norms of differences
    norms = np.linalg.norm(diffs, axis=1)
    
    # Return average norm
    return np.mean(norms)


def compute_hessian_eigenvalues(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    num_eigenvalues: int = 10
) -> Tuple[np.ndarray, float]:
    """
    Compute top eigenvalues of Hessian (approximation)
    Useful for checking strong convexity/smoothness
    
    Args:
        model: Neural network model
        data_loader: Data loader
        loss_fn: Loss function
        num_eigenvalues: Number of top eigenvalues to compute
        
    Returns:
        Top eigenvalues and condition number
    """
    # This is a simplified approximation
    # Full Hessian computation is expensive for neural networks
    
    model.eval()
    
    # Compute gradient covariance as Hessian approximation
    gradients = []
    
    for inputs, targets in data_loader:
        model.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        
        grads = []
        for p in model.parameters():
            if p.grad is not None:
                grads.append(p.grad.view(-1))
        
        if grads:
            all_grads = torch.cat(grads)
            gradients.append(all_grads.cpu().numpy())
    
    if not gradients:
        return np.array([]), 0.0
    
    # Compute covariance
    gradients = np.array(gradients)
    cov = np.cov(gradients.T)
    
    # Compute eigenvalues
    eigenvalues = np.linalg.eigvalsh(cov)
    top_eigenvalues = eigenvalues[-num_eigenvalues:]
    
    # Condition number
    if eigenvalues[0] > 0:
        condition_number = eigenvalues[-1] / eigenvalues[0]
    else:
        condition_number = float('inf')
    
    return top_eigenvalues, condition_number