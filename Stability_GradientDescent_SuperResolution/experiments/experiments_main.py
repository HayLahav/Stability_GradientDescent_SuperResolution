"""
Main experiment runner for stability analysis with comprehensive fixes
"""

import sys
import os
from pathlib import Path

# Add project root to path for absolute imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import torch
import numpy as np
from typing import Dict, Any, Optional
import logging
from datetime import datetime

from src.models.src_models_srcnn import SimpleSRCNN
from src.optimizers.src_optimizers_adafm import AdaFMOptimizer
from src.optimizers.src_optimizers_adaptive_sgd import AdaptiveSGD, polynomial_decay_lr
from src.data.src_data_synthetic import SyntheticSRDataset
from src.training.src_training_trainer import StabilityTrainer
from src.stability.src_stability_analyzer import StabilityAnalyzer
from src.utils.src_utils_config import load_config, save_config
from src.utils.src_utils_visualization import plot_stability_analysis
from src.utils.src_utils_metrics import evaluate_model

def setup_logging(log_dir: Path) -> logging.Logger:
    """Setup comprehensive logging"""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('stability_experiment')
    logger.setLevel(logging.INFO)
    
    # File handler
    log_file = log_dir / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration completeness"""
    required_sections = ['model', 'optimizer', 'training', 'data', 'stability', 'loss', 'logging']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
    
    # Validate specific fields
    if config['model']['name'] not in ['Baseline', 'WithCorrection', 'WithAdaFMOpt', 'FullSystem']:
        raise ValueError(f"Invalid model name: {config['model']['name']}")
    
    if config['optimizer']['name'] not in ['SGD', 'AdaFM', 'AdaptiveSGD']:
        raise ValueError(f"Invalid optimizer name: {config['optimizer']['name']}")
    
    return True


def setup_experiment(config: Dict[str, Any], logger: logging.Logger) -> Dict[str, Any]:
    """Setup experiment components with improved error handling"""
    
    # Validate configuration
    validate_config(config)
    
    # Set device with fallback
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    if device == 'cpu':
        logger.warning("CUDA not available, using CPU. This may be slow.")
    
    # Create datasets with validation
    try:
        dataset = SyntheticSRDataset(
            num_samples=config['data']['num_samples'],
            image_size=config['data']['image_size'],
            scale_factor=config['data']['scale_factor'],
            noise_level=config['data']['noise_level']
        )
        logger.info(f"Created dataset with {len(dataset)} samples")
    except Exception as e:
        logger.error(f"Failed to create dataset: {e}")
        raise
    
    # Split into train/val with proper validation
    val_size = int(len(dataset) * config['data']['val_split'])
    train_size = len(dataset) - val_size
    
    if train_size <= 0 or val_size <= 0:
        raise ValueError(f"Invalid data split: train={train_size}, val={val_size}")
    
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Reproducible splits
    )
    
    # Create perturbed dataset for stability analysis
    perturbed_dataset = None
    if config['stability']['compute']:
        try:
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
            logger.info("Created perturbed dataset for stability analysis")
        except Exception as e:
            logger.error(f"Failed to create perturbed dataset: {e}")
            raise
    
    # Create data loaders with error handling
    try:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=config['training']['num_workers'],
            pin_memory=config['training']['pin_memory'] and device == 'cuda',
            drop_last=True  # Ensure consistent batch sizes
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config['training']['num_workers'],
            pin_memory=config['training']['pin_memory'] and device == 'cuda'
        )
        
        perturbed_loader = None
        if perturbed_dataset:
            perturbed_loader = DataLoader(
                perturbed_dataset,
                batch_size=config['training']['batch_size'],
                shuffle=True,
                num_workers=config['training']['num_workers'],
                pin_memory=config['training']['pin_memory'] and device == 'cuda',
                drop_last=True
            )
    except Exception as e:
        logger.error(f"Failed to create data loaders: {e}")
        raise
    
    # Create model with validation
    try:
        model = SimpleSRCNN(
            use_adafm=config['model']['use_adafm'],
            use_correction=config['model']['use_correction'],
            num_channels=config['model']['num_channels'],
            num_filters=tuple(config['model']['num_filters'])
        )
        model = model.to(device)
        
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Created model with {num_params:,} trainable parameters")
    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        raise
    
    # Create optimizer with comprehensive support
    try:
        if config['optimizer']['name'] == 'AdaFM':
            optimizer = AdaFMOptimizer(
                model.parameters(),
                gamma=config['optimizer']['gamma'],
                lam=config['optimizer'].get('lam', 1.0),
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
        
        logger.info(f"Created {config['optimizer']['name']} optimizer")
    except Exception as e:
        logger.error(f"Failed to create optimizer: {e}")
        raise
    
    # Create loss function
    loss_fn = torch.nn.MSELoss()
    
    # Create stability analyzer with proper parameters
    try:
        stability_analyzer = StabilityAnalyzer(
            model=model,
            loss_fn=loss_fn,
            L=1.0,  # Conservative Lipschitz estimate
            alpha=0.1,  # Weak convexity assumption
            beta=None   # Not necessarily smooth
        )
        logger.info("Created stability analyzer")
    except Exception as e:
        logger.error(f"Failed to create stability analyzer: {e}")
        raise
    
    # Create trainer
    try:
        trainer = StabilityTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            stability_analyzer=stability_analyzer,
            perturbed_dataset=perturbed_dataset,
            device=device
        )
        logger.info("Created stability trainer")
    except Exception as e:
        logger.error(f"Failed to create trainer: {e}")
        raise
    
    return {
        'trainer': trainer,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'perturbed_loader': perturbed_loader,
        'device': device,
        'model': model,
        'optimizer': optimizer,
        'stability_analyzer': stability_analyzer
    }


def run_experiment(config_path: str, seed: Optional[int] = 42) -> None:
    """Run stability analysis experiment with comprehensive error handling and logging"""
    
    # Set random seeds for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    # Load configuration
    try:
        config = load_config(config_path)
        config_dict = config.to_dict()
    except Exception as e:
        print(f"Failed to load config from {config_path}: {e}")
        raise
    
    # Create output directory
    output_dir = Path(config_dict['logging']['save_dir']) / config_dict['model']['name']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    logger.info(f"Starting experiment: {config_dict['model']['name']}")
    logger.info(f"Config loaded from: {config_path}")
    logger.info(f"Random seed: {seed}")
    
    try:
        # Save configuration
        save_config(config, output_dir / 'config.yaml')
        logger.info(f"Configuration saved to {output_dir / 'config.yaml'}")
        
        # Setup experiment
        logger.info("Setting up experiment components...")
        experiment = setup_experiment(config_dict, logger)
        
        # Get test samples for empirical gamma computation
        test_samples = next(iter(experiment['val_loader']))[0][:5]
        logger.info(f"Using {len(test_samples)} test samples for empirical gamma computation")
        
        # Train with stability analysis
        logger.info("Starting training with stability analysis...")
        logger.info(f"Training for {config_dict['training']['epochs']} epochs")
        
        history = experiment['trainer'].fit_with_stability(
            train_loader=experiment['train_loader'],
            val_loader=experiment['val_loader'],
            perturbed_loader=experiment['perturbed_loader'],
            num_epochs=config_dict['training']['epochs'],
            test_samples=test_samples
        )
        
        logger.info("Training completed successfully")
        
        # Save training history
        torch.save(history, output_dir / 'history.pt')
        logger.info(f"Training history saved to {output_dir / 'history.pt'}")
        
        # Evaluate final model
        logger.info("Evaluating final model...")
        eval_results = evaluate_model(
            experiment['trainer'].model,
            experiment['val_loader'],
            device=experiment['device']
        )
        
        logger.info("Final evaluation results:")
        for metric, value in eval_results.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Save evaluation results
        torch.save(eval_results, output_dir / 'eval_results.pt')
        
        # Generate plots
        logger.info("Generating analysis plots...")
        try:
            plot_stability_analysis(
                {config_dict['model']['name']: history},
                save_path=output_dir / 'stability_analysis.png'
            )
            logger.info(f"Plots saved to {output_dir / 'stability_analysis.png'}")
        except Exception as e:
            logger.warning(f"Failed to generate plots: {e}")
        
        # Save final model with comprehensive metadata
        model_checkpoint = {
            'model_state_dict': experiment['trainer'].model.state_dict(),
            'config': config_dict,
            'eval_results': eval_results,
            'training_history': history,
            'experiment_metadata': {
                'seed': seed,
                'device': experiment['device'],
                'num_parameters': sum(p.numel() for p in experiment['model'].parameters()),
                'pytorch_version': torch.__version__,
                'experiment_timestamp': datetime.now().isoformat()
            }
        }
        
        # Include optimizer state if available
        if hasattr(experiment['trainer'].optimizer, 'state_dict'):
            model_checkpoint['optimizer_state_dict'] = experiment['trainer'].optimizer.state_dict()
        
        torch.save(model_checkpoint, output_dir / 'final_model.pt')
        logger.info(f"Final model saved to {output_dir / 'final_model.pt'}")
        
        # Summary statistics
        final_train_loss = history['train_loss'][-1]
        final_val_loss = history.get('val_loss', [0])[-1]
        final_param_distance = history.get('parameter_distance', [0])[-1]
        final_empirical_gamma = history.get('empirical_gamma', [0])[-1]
        
        logger.info("=== EXPERIMENT SUMMARY ===")
        logger.info(f"Model: {config_dict['model']['name']}")
        logger.info(f"Final Training Loss: {final_train_loss:.6f}")
        logger.info(f"Final Validation Loss: {final_val_loss:.6f}")
        logger.info(f"Final Parameter Distance: {final_param_distance:.6f}")
        logger.info(f"Final Empirical γ(m): {final_empirical_gamma:.6f}")
        logger.info(f"Results saved to: {output_dir}")
        
        print(f"\n✅ Experiment completed successfully!")
        print(f"📁 Results saved to: {output_dir}")
        return history, eval_results
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        logger.exception("Full traceback:")
        raise


def main():
    """Main entry point with improved argument parsing"""
    parser = argparse.ArgumentParser(
        description='Run stability analysis experiment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python experiments/main.py --config configs/baseline.yaml
  python experiments/main.py --config configs/full_system.yaml --seed 123
  python experiments/main.py --config configs/adafm_opt.yaml --verbose
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Validate config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        return 1
    
    try:
        # Run experiment
        run_experiment(str(config_path), args.seed)
        return 0
    except KeyboardInterrupt:
        print("\n⚠️  Experiment interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        return 1


if __name__ == '__main__':
    exit(main())
