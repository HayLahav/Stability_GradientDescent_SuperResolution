"""
Simple training example
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models import SimpleSRCNN
from src.optimizers import AdaFMOptimizer
from src.data import SyntheticSRDataset
from src.training import Trainer
from src.utils import evaluate_model, plot_convergence_curves


def main():
    """Simple training example"""
    
    print("=== Simple Super-Resolution Training Example ===\n")
    
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create synthetic dataset
    print("\nCreating synthetic dataset...")
    dataset = SyntheticSRDataset(
        num_samples=500,
        image_size=32,
        scale_factor=2,
        noise_level=0.01
    )
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    print(f"Dataset size: {len(dataset)} (train: {train_size}, val: {val_size})")
    
    # Create model
    print("\nCreating model...")
    model = SimpleSRCNN(
        use_correction=True,
        use_adafm=True
    )
    print(f"Number of parameters: {model.get_num_params():,}")
    
    # Create optimizer
    print("\nInitializing AdaFM optimizer...")
    optimizer = AdaFMOptimizer(
        model.parameters(),
        gamma=1.0,
        delta=0.001,
        single_variable=True
    )
    
    # Loss function
    loss_fn = nn.MSELoss()
    
    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device
    )
    
    # Train model
    print("\nTraining model...")
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=20
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    eval_results = evaluate_model(
        model,
        val_loader,
        device=device,
        metrics=('psnr', 'ssim', 'mse')
    )
    
    print("\nEvaluation results:")
    for metric, value in eval_results.items():
        print(f"  {metric}: {value:.4f}")
    
    # Plot results
    plot_convergence_curves(
        {'Simple Training': history['train_loss']},
        title="Training Convergence",
        save_path="simple_training_convergence.png"
    )
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if hasattr(optimizer, 'state_dict') else None,
        'history': history,
        'eval_results': eval_results
    }, 'simple_training_model.pt')
    
    print("\nTraining completed!")
    print("Model saved to: simple_training_model.pt")
    print("Plot saved to: simple_training_convergence.png")


if __name__ == '__main__':
    main()