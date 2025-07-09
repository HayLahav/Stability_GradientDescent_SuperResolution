# Integration fix for potential import issues in experiments/experiments_main.py

"""
Main experiment runner for stability analysis with improved import handling
"""

import sys
import os
from pathlib import Path

# Add project root to path for absolute imports - improved version
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import torch
import numpy as np
from typing import Dict, Any, Optional
import logging
from datetime import datetime

# Import with error handling
try:
    from src.models.srcnn import SimpleSRCNN
    from src.optimizers.adafm_optimizer import AdaFMOptimizer
    from src.optimizers.adaptive_sgd import AdaptiveSGD, polynomial_decay_lr
    from src.data.synthetic import SyntheticSRDataset
    from src.training.trainer import StabilityTrainer
    from src.stability.analyzer import StabilityAnalyzer
    from src.utils.config import load_config, save_config
    from src.utils.visualization import plot_stability_analysis
    from src.utils.metrics import evaluate_model
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure you have installed the package with 'pip install -e .'")
    print("And that you're running from the project root directory")
    sys.exit(1)

from torch.utils.data import DataLoader, random_split

def setup_logging(log_dir: Path) -> logging.Logger:
    """Setup comprehensive logging with improved error handling"""
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Warning: Could not create log directory {log_dir}: {e}")
        log_dir = Path("./logs")
        log_dir.mkdir(exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('stability_experiment')
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # File handler
    log_file = log_dir / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    try:
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
        
        logger.info(f"Logging initialized. Log file: {log_file}")
    except Exception as e:
        print(f"Warning: Could not setup file logging: {e}")
        # Fall back to console only
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


def validate_config(config: Dict[str, Any]) -> bool:
    """Enhanced configuration validation"""
    required_sections = ['model', 'optimizer', 'training', 'data', 'stability', 'loss', 'logging']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
    
    # Validate model configuration
    model_config = config['model']
    if 'name' not in model_config:
        raise ValueError("Model name is required")
    
    # Validate optimizer configuration
    opt_config = config['optimizer']
    if 'name' not in opt_config:
        raise ValueError("Optimizer name is required")
    
    valid_optimizers = ['SGD', 'AdaFM', 'AdaptiveSGD']
    if opt_config['name'] not in valid_optimizers:
        raise ValueError(f"Invalid optimizer name: {opt_config['name']}. Must be one of {valid_optimizers}")
    
    # Validate training configuration
    training_config = config['training']
    required_training_keys = ['epochs', 'batch_size']
    for key in required_training_keys:
        if key not in training_config:
            raise ValueError(f"Missing required training parameter: {key}")
    
    # Validate data configuration
    data_config = config['data']
    required_data_keys = ['num_samples', 'image_size', 'scale_factor']
    for key in required_data_keys:
        if key not in data_config:
            raise ValueError(f"Missing required data parameter: {key}")
    
    return True


def setup_experiment_safe(config: Dict[str, Any], logger: logging.Logger) -> Optional[Dict[str, Any]]:
    """Setup experiment components with comprehensive error handling"""
    
    try:
        # Validate configuration
        validate_config(config)
        logger.info("Configuration validation passed")
        
        # Set device with fallback
        if torch.cuda.is_available():
            device = 'cuda'
            logger.info(f"CUDA available. Using GPU: {torch.cuda.get_device_name()}")
        else:
            device = 'cpu'
            logger.warning("CUDA not available, using CPU. This may be slow for large experiments.")
        
        # Create datasets with validation
        logger.info("Creating datasets...")
        try:
            dataset = SyntheticSRDataset(
                num_samples=config['data']['num_samples'],
                image_size=config['data']['image_size'],
                scale_factor=config['data']['scale_factor'],
                noise_level=config['data'].get('noise_level', 0.01)
            )
            logger.info(f"Created dataset with {len(dataset)} samples")
        except Exception as e:
            logger.error(f"Failed to create dataset: {e}")
            return None
        
        # Split into train/val with proper validation
        val_split = config['data'].get('val_split', 0.2)
        val_size = max(1, int(len(dataset) * val_split))  # Ensure at least 1 sample
        train_size = len(dataset) - val_size
        
        if train_size <= 0:
            logger.error(f"Invalid data split: train_size={train_size}")
            return None
        
        try:
            train_dataset, val_dataset = random_split(
                dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )
            logger.info(f"Split dataset: {train_size} train, {val_size} validation")
        except Exception as e:
            logger.error(f"Failed to split dataset: {e}")
            return None
        
        # Create perturbed dataset for stability analysis
        perturbed_dataset = None
        if config['stability'].get('compute', False):
            try:
                perturbed_dataset = SyntheticSRDataset(
                    num_samples=config['data']['num_samples'],
                    image_size=config['data']['image_size'],
                    scale_factor=config['data']['scale_factor'],
                    noise_level=config['data'].get('noise_level', 0.01)
                )
                # Add perturbation to specific sample
                perturbation_idx = config['stability'].get('perturbation_idx', 0)
                perturbation_strength = config['stability'].get('perturbation_strength', 0.1)
                perturbed_dataset.add_perturbation(perturbation_idx, perturbation_strength)
                logger.info("Created perturbed dataset for stability analysis")
            except Exception as e:
                logger.error(f"Failed to create perturbed dataset: {e}")
                return None
        
        # Create data loaders with error handling
        try:
            batch_size = config['training']['batch_size']
            num_workers = config['training'].get('num_workers', 0)
            pin_memory = config['training'].get('pin_memory', False) and device == 'cuda'
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=len(train_dataset) > batch_size  # Only drop if we have enough samples
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory
            )
            
            perturbed_loader = None
            if perturbed_dataset:
                perturbed_loader = DataLoader(
                    perturbed_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    drop_last=len(perturbed_dataset) > batch_size
                )
            
            logger.info("Created data loaders successfully")
        except Exception as e:
            logger.error(f"Failed to create data loaders: {e}")
            return None
        
        # Create model with validation
        try:
            model_config = config['model']
            model = SimpleSRCNN(
                use_adafm=model_config.get('use_adafm', False),
                use_correction=model_config.get('use_correction', False),
                num_channels=model_config.get('num_channels', 3),
                num_filters=tuple(model_config.get('num_filters', [64, 32]))
            )
            model = model.to(device)
            
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"Created model with {num_params:,} trainable parameters")
        except Exception as e:
            logger.error(f"Failed to create model: {e}")
            return None
        
        # Create optimizer with comprehensive support
        try:
            opt_config = config['optimizer']
            if opt_config['name'] == 'AdaFM':
                optimizer = AdaFMOptimizer(
                    model.parameters(),
                    gamma=opt_config.get('gamma', 1.0),
                    lam=opt_config.get('lam', 1.0),
                    delta=opt_config.get('delta', 0.001),
                    single_variable=True
                )
            elif opt_config['name'] == 'AdaptiveSGD':
                lr_schedule = polynomial_decay_lr(
                    opt_config.get('lr', 0.01),
                    power=0.5
                )
                optimizer = AdaptiveSGD(
                    model.parameters(),
                    lr=opt_config.get('lr', 0.01),
                    momentum=opt_config.get('momentum', 0.9),
                    weight_decay=opt_config.get('weight_decay', 0.0),
                    lr_schedule=lr_schedule
                )
            else:  # Standard SGD
                optimizer = torch.optim.SGD(
                    model.parameters(),
                    lr=opt_config.get('lr', 0.01),
                    momentum=opt_config.get('momentum', 0.9),
                    weight_decay=opt_config.get('weight_decay', 0.0)
                )
            
            logger.info(f"Created {opt_config['name']} optimizer")
        except Exception as e:
            logger.error(f"Failed to create optimizer: {e}")
            return None
        
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
            return None
        
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
            return None
        
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
        
    except Exception as e:
        logger.error(f"Unexpected error in experiment setup: {e}")
        return None


def main():
    """Main entry point with improved error handling"""
    parser = argparse.ArgumentParser(
        description='Run stability analysis experiment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python experiments/experiments_main.py --config experiments/configs/baseline.yaml
  python experiments/experiments_main.py --config experiments/configs/full_system.yaml --seed 123
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
    
    args = parser.parse_args()
    
    # Validate config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        print(f"Current working directory: {os.getcwd()}")
        print("Available config files:")
        config_dir = Path("experiments/configs")
        if config_dir.exists():
            for config_file in config_dir.glob("*.yaml"):
                print(f"  - {config_file}")
        return 1
    
    # Set random seeds for reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
    
    try:
        # Load configuration
        config = load_config(config_path)
        config_dict = config.to_dict()
        
        # Create output directory
        output_dir = Path(config_dict['logging']['save_dir']) / config_dict['model']['name']
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logger = setup_logging(output_dir)
        logger.info(f"Starting experiment: {config_dict['model']['name']}")
        logger.info(f"Config loaded from: {config_path}")
        logger.info(f"Random seed: {args.seed}")
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        
        # Setup experiment
        logger.info("Setting up experiment components...")
        experiment = setup_experiment_safe(config_dict, logger)
        
        if experiment is None:
            logger.error("Failed to setup experiment. Check logs for details.")
            return 1
        
        # Save configuration
        save_config(config, output_dir / 'config.yaml')
        logger.info(f"Configuration saved to {output_dir / 'config.yaml'}")
        
        # Get test samples for empirical gamma computation
        try:
            test_samples = next(iter(experiment['val_loader']))[0][:5]
            logger.info(f"Using {len(test_samples)} test samples for empirical gamma computation")
        except Exception as e:
            logger.warning(f"Could not get test samples: {e}")
            test_samples = None
        
        # Train with stability analysis
        logger.info("Starting training with stability analysis...")
        logger.info(f"Training for {config_dict['training']['epochs']} epochs")
        
        try:
            history = experiment['trainer'].fit_with_stability(
                train_loader=experiment['train_loader'],
                val_loader=experiment['val_loader'],
                perturbed_loader=experiment['perturbed_loader'],
                num_epochs=config_dict['training']['epochs'],
                test_samples=test_samples
            )
            
            logger.info("Training completed successfully")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return 1
        
        # Save training history
        torch.save(history, output_dir / 'history.pt')
        logger.info(f"Training history saved to {output_dir / 'history.pt'}")
        
        # Evaluate final model
        logger.info("Evaluating final model...")
        try:
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
        except Exception as e:
            logger.warning(f"Model evaluation failed: {e}")
            eval_results = {}
        
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
        
        # Save final model
        try:
            model_checkpoint = {
                'model_state_dict': experiment['trainer'].model.state_dict(),
                'config': config_dict,
                'eval_results': eval_results,
                'training_history': history,
                'experiment_metadata': {
                    'seed': args.seed,
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
        except Exception as e:
            logger.warning(f"Failed to save final model: {e}")
        
        # Summary statistics
        try:
            final_train_loss = history['train_loss'][-1] if history['train_loss'] else 0
            final_val_loss = history.get('val_loss', [0])[-1] if history.get('val_loss') else 0
            final_param_distance = history.get('parameter_distance', [0])[-1] if history.get('parameter_distance') else 0
            final_empirical_gamma = history.get('empirical_gamma', [0])[-1] if history.get('empirical_gamma') else 0
            
            logger.info("=== EXPERIMENT SUMMARY ===")
            logger.info(f"Model: {config_dict['model']['name']}")
            logger.info(f"Final Training Loss: {final_train_loss:.6f}")
            logger.info(f"Final Validation Loss: {final_val_loss:.6f}")
            logger.info(f"Final Parameter Distance: {final_param_distance:.6f}")
            logger.info(f"Final Empirical γ(m): {final_empirical_gamma:.6f}")
            logger.info(f"Results saved to: {output_dir}")
        except Exception as e:
            logger.warning(f"Failed to compute summary statistics: {e}")
        
        print(f"\n✅ Experiment completed successfully!")
        print(f"📁 Results saved to: {output_dir}")
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Experiment interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
