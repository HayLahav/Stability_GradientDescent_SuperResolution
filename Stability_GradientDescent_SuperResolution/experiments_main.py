"""
Main experiment runner for stability analysis
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any

from src.models import SimpleSRCNN
from src.optimizers import AdaFMOptimizer
from src.optimizers.adaptive_sgd import AdaptiveSGD, polynomial_decay_lr
from src.data import SyntheticSRDataset
from src.training import StabilityTrainer
from src.stability import StabilityAnalyzer
from src.utils import (
    load_config,
    save_config,
    plot_stability_analysis,
    evaluate_model
)
from torch.utils.data import DataLoader, random_split


def setup_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    """Setup experiment components based on configuration"""
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create datasets
    dataset = SyntheticSRDataset(
        num_samples=config['data']['num_samples'],
        image_size=config['data']['image_size'],
        scale_factor=config['data']['scale_factor'],
        noise_level=config['data']['noise_level']
    )
    
    # Split into train/val
    val_size = int(len(dataset) * config['data']['val_split'])
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create perturbed dataset for stability analysis
    perturbed_dataset = None
    if config['stability']['compute']:
        perturbed_dataset = SyntheticSRDataset(
            num_samples=config['data']['num_samples'],
            image_size=config['data']['image_size'],
            scale_factor=config['data']['scale_factor'],
            noise_level=config['data']['noise_level']
        )
        # Add perturbation to specific sample
        perturbed_dataset.add_perturbation(
            config['stability']['perturbation_idx'],
            config['stability']['perturbation_strength']
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory']
    )
    
    perturbed_loader = None
    if perturbed_dataset:
        perturbed_loader = DataLoader(
            perturbed_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=config['training']['num_workers'],
            pin_memory=config['training']['pin_memory']
        )
    
    # Create model
    model = SimpleSRCNN(
        use_adafm=config['model']['use_adafm'],
        use_correction=config['model']['use_correction'],
        num_channels=config['model']['num_channels'],
        num_filters=tuple(config['model']['num_filters'])
    )
    
    # Create optimizer
    if config['optimizer']['name'] == 'AdaFM':
        optimizer = AdaFMOptimizer(
            model.parameters(),
            gamma=config['optimizer']['gamma'],
            delta=config['optimizer']['delta'],
            single_variable=True
        )
    elif config['optimizer']['name'] == 'AdaptiveSGD':
        lr_schedule = polynomial_decay_lr(
            config['optimizer']['lr'],
            power=0.5
        )
        optimizer = AdaptiveSGD(
            model.parameters(),
            lr=config['optimizer']['lr'],
            momentum=config['optimizer']['momentum'],
            weight_decay=config['optimizer']['weight_decay'],
            lr_schedule=lr_schedule
        )
    else:  # Standard SGD
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config['optimizer']['lr'],
            momentum=config['optimizer']['momentum'],
            weight_decay=config['optimizer']['weight_decay']
        )
    
    # Create loss function
    loss_fn = torch.nn.MSELoss()
    
    # Create stability analyzer
    stability_analyzer = StabilityAnalyzer(
        model, loss_fn, L=1.0, alpha=0.1
    )
    
    # Create trainer
    trainer = StabilityTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        stability_analyzer=stability_analyzer,
        perturbed_dataset=perturbed_dataset,
        device=device
    )
    
    return {
        'trainer': trainer,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'perturbed_loader': perturbed_loader,
        'device': device
    }


def run_experiment(config_path: str):
    """Run stability analysis experiment"""
    
    # Load configuration
    config = load_config(config_path)
    config_dict = config.to_dict()
    
    # Setup experiment
    experiment = setup_experiment(config_dict)
    
    # Create output directory
    output_dir = Path(config_dict['logging']['save_dir']) / config_dict['model']['name']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    save_config(config, output_dir / 'config.yaml')
    
    # Get test samples for empirical gamma computation
    test_samples = next(iter(experiment['val_loader']))[0][:5]
    
    # Train with stability analysis
    print("\nStarting training with stability analysis...")
    history = experiment['trainer'].fit_with_stability(
        train_loader=experiment['train_loader'],
        val_loader=experiment['val_loader'],
        perturbed_loader=experiment['perturbed_loader'],
        num_epochs=config_dict['training']['epochs'],
        test_samples=test_samples
    )
    
    # Save training history
    torch.save(history, output_dir / 'history.pt')
    
    # Evaluate final model
    print("\nEvaluating final model...")
    eval_results = evaluate_model(
        experiment['trainer'].model,
        experiment['val_loader'],
        device=experiment['device']
    )
    
    print("\nFinal evaluation results:")
    for metric, value in eval_results.items():
        print(f"  {metric}: {value:.4f}")
    
    # Save evaluation results
    torch.save(eval_results, output_dir / 'eval_results.pt')
    
    # Plot results
    print("\nGenerating plots...")
    plot_stability_analysis(
        {config_dict['model']['name']: history},
        save_path=output_dir / 'stability_analysis.png'
    )
    
    # Save final model
    torch.save({
        'model_state_dict': experiment['trainer'].model.state_dict(),
        'optimizer_state_dict': experiment['trainer'].optimizer.state_dict() 
            if hasattr(experiment['trainer'].optimizer, 'state_dict') else None,
        'config': config_dict,
        'eval_results': eval_results
    }, output_dir / 'final_model.pt')
    
    print(f"\nExperiment completed! Results saved to {output_dir}")
    
    return history, eval_results


def main():
    parser = argparse.ArgumentParser(description='Run stability analysis experiment')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Run experiment
    run_experiment(args.config)


if __name__ == '__main__':
    main()