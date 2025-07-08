"""
Complete integration tests for the stability analysis framework
Tests end-to-end functionality and component integration
"""

import pytest
import torch
import tempfile
import yaml
from pathlib import Path
import shutil
import numpy as np

from src.models import SimpleSRCNN
from src.optimizers import AdaFMOptimizer
from src.data import SyntheticSRDataset
from src.training import StabilityTrainer
from src.stability import StabilityAnalyzer
from src.utils import load_config, save_config, Config, evaluate_model
from torch.utils.data import DataLoader


class TestFullPipelineIntegration:
    """Test complete experimental pipeline integration"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_config(self):
        """Create sample configuration for testing"""
        return {
            'model': {
                'name': 'TestModel',
                'use_correction': False,
                'use_adafm': False,
                'num_channels': 3,
                'num_filters': [16, 8]  # Smaller for faster testing
            },
            'optimizer': {
                'name': 'SGD',
                'lr': 0.01,
                'momentum': 0.9,
                'weight_decay': 0.0
            },
            'training': {
                'epochs': 2,  # Minimal for testing
                'batch_size': 4,
                'num_workers': 0,  # Avoid multiprocessing in tests
                'pin_memory': False
            },
            'data': {
                'num_samples': 20,  # Small dataset for testing
                'image_size': 16,   # Small images for speed
                'scale_factor': 2,
                'noise_level': 0.01,
                'val_split': 0.2
            },
            'stability': {
                'compute': True,
                'perturbation_idx': 0,
                'perturbation_strength': 0.1
            },
            'loss': {
                'name': 'MSE',
                'weights': {
                    'mse': 1.0,
                    'psnr': 0.0,
                    'ssim': 0.0,
                    'perceptual': 0.0
                }
            },
            'logging': {
                'save_dir': 'test_results',
                'log_interval': 1,
                'save_checkpoint': False,
                'tensorboard': False
            }
        }
    
    def test_config_roundtrip(self, temp_dir, sample_config):
        """Test configuration save/load roundtrip"""
        config_path = temp_dir / 'test_config.yaml'
        
        # Create and save config
        config = Config(sample_config)
        save_config(config, config_path)
        
        # Verify file exists
        assert config_path.exists()
        
        # Load and verify
        loaded_config = load_config(config_path)
        assert loaded_config.model.name == sample_config['model']['name']
        assert loaded_config.optimizer.lr == sample_config['optimizer']['lr']
        assert loaded_config.data.num_samples == sample_config['data']['num_samples']
        
        print("✅ Config save/load roundtrip works")
    
    def test_model_creation_variants(self):
        """Test all model variant creation"""
        variants = [
            {'use_correction': False, 'use_adafm': False},  # Baseline
            {'use_correction': True, 'use_adafm': False},   # With correction
            {'use_correction': False, 'use_adafm': True},   # With AdaFM layers
            {'use_correction': True, 'use_adafm': True}     # Full system
        ]
        
        for i, config in enumerate(variants):
            model = SimpleSRCNN(num_filters=(16, 8), **config)
            assert isinstance(model, torch.nn.Module)
            
            # Test forward pass
            x = torch.randn(2, 3, 16, 16)
            y = model(x)
            assert y.shape == x.shape
            
            # Test parameter count
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            assert num_params > 0
        
        print("✅ All model variants work")
    
    def test_data_pipeline_integration(self):
        """Test complete data pipeline"""
        # Create dataset
        dataset = SyntheticSRDataset(
            num_samples=10,
            image_size=16,
            scale_factor=2,
            noise_level=0.01
        )
        
        # Test basic functionality
        assert len(dataset) == 10
        lr, hr = dataset[0]
        assert lr.shape == (3, 8, 8)  # 16/2 = 8
        assert hr.shape == (3, 16, 16)
        assert lr.dtype == torch.float32
        assert hr.dtype == torch.float32
        
        # Test value ranges
        assert torch.all(lr >= 0) and torch.all(lr <= 1)
        assert torch.all(hr >= 0) and torch.all(hr <= 1)
        
        # Test perturbation
        original_lr = lr.clone()
        dataset.add_perturbation(0, 0.1)
        perturbed_lr, _ = dataset[0]
        assert not torch.allclose(original_lr, perturbed_lr, atol=1e-6)
        
        # Test data loader
        loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
        batch_lr, batch_hr = next(iter(loader))
        assert batch_lr.shape == (4, 3, 8, 8)
        assert batch_hr.shape == (4, 3, 16, 16)
        
        print("✅ Data pipeline works")
    
    def test_optimizer_integration(self):
        """Test optimizer creation and basic functionality"""
        model = SimpleSRCNN(num_filters=(16, 8))
        
        # Test SGD
        sgd_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        assert len(list(sgd_optimizer.param_groups)) == 1
        
        # Test AdaFM
        adafm_optimizer = AdaFMOptimizer(
            model.parameters(),
            gamma=1.0,
            delta=0.001,
            single_variable=True
        )
        assert adafm_optimizer.single_variable == True
        assert adafm_optimizer.gamma == 1.0
        
        # Test optimization step
        x = torch.randn(2, 3, 16, 16)
        y_target = torch.randn(2, 3, 16, 16)
        
        adafm_optimizer.zero_grad()
        y_pred = model(x)
        loss = torch.nn.functional.mse_loss(y_pred, y_target)
        loss.backward()
        adafm_optimizer.step()
        
        # Verify optimizer state updated
        assert adafm_optimizer.t == 1
        assert adafm_optimizer.alpha_x > 0
        
        print("✅ Optimizers work")
    
    def test_stability_analyzer_integration(self):
        """Test stability analyzer with models"""
        model1 = SimpleSRCNN(num_filters=(16, 8))
        model2 = SimpleSRCNN(num_filters=(16, 8))
        
        # Create analyzer
        analyzer = StabilityAnalyzer(
            model=model1,
            loss_fn=torch.nn.MSELoss(),
            L=1.0,
            alpha=0.1
        )
        
        # Test parameter distance (should be 0 for identical models)
        distance = analyzer.compute_parameter_distance(model1, model2)
        assert distance == 0.0
        
        # Modify one model and test again
        with torch.no_grad():
            for p in model2.parameters():
                p.add_(torch.randn_like(p) * 0.01)
        
        distance = analyzer.compute_parameter_distance(model1, model2)
        assert distance > 0.0
        
        # Test empirical gamma
        test_data = torch.randn(5, 3, 16, 16)
        gamma = analyzer.compute_empirical_gamma(model1, model2, test_data)
        assert gamma >= 0.0
        
        # Test theoretical bounds
        theo_gamma = analyzer.compute_theoretical_gamma_general(100, 1000, 0.01)
        assert theo_gamma > 0.0
        
        print("✅ Stability analyzer works")
    
    def test_training_integration(self, sample_config):
        """Test training pipeline integration"""
        # Create components
        model = SimpleSRCNN(num_filters=(8, 4))  # Very small for fast testing
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss_fn = torch.nn.MSELoss()
        
        # Create datasets
        dataset = SyntheticSRDataset(
            num_samples=8,
            image_size=16,
            scale_factor=2
        )
        
        perturbed_dataset = SyntheticSRDataset(
            num_samples=8,
            image_size=16,
            scale_factor=2
        )
        perturbed_dataset.add_perturbation(0, 0.1)
        
        # Create data loaders
        train_loader = DataLoader(dataset, batch_size=4, num_workers=0)
        perturbed_loader = DataLoader(perturbed_dataset, batch_size=4, num_workers=0)
        
        # Create stability components
        stability_analyzer = StabilityAnalyzer(
            model=model,
            loss_fn=loss_fn,
            L=1.0,
            alpha=0.1
        )
        
        trainer = StabilityTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            stability_analyzer=stability_analyzer,
            perturbed_dataset=perturbed_dataset,
            device='cpu'
        )
        
        # Test short training run
        test_samples = torch.randn(2, 3, 16, 16)
        history = trainer.fit_with_stability(
            train_loader=train_loader,
            val_loader=None,
            perturbed_loader=perturbed_loader,
            num_epochs=1,
            test_samples=test_samples
        )
        
        # Verify history structure
        required_keys = ['train_loss', 'parameter_distance', 'empirical_gamma', 'theoretical_gamma']
        for key in required_keys:
            assert key in history, f"Missing key: {key}"
        
        assert len(history['train_loss']) == 1
        assert len(history['parameter_distance']) == 1
        assert history['train_loss'][0] > 0
        
        print("✅ Training integration works")
    
    def test_evaluation_integration(self):
        """Test model evaluation pipeline"""
        model = SimpleSRCNN(num_filters=(8, 4))
        
        # Create test dataset
        dataset = SyntheticSRDataset(
            num_samples=8,
            image_size=16,
            scale_factor=2
        )
        test_loader = DataLoader(dataset, batch_size=4, num_workers=0)
        
        # Evaluate model
        results = evaluate_model(
            model=model,
            test_loader=test_loader,
            device='cpu',
            metrics=('psnr', 'ssim', 'mse')
        )
        
        # Verify results structure
        expected_keys = ['psnr_mean', 'psnr_std', 'ssim_mean', 'ssim_std', 'mse_mean', 'mse_std']
        for key in expected_keys:
            assert key in results, f"Missing key: {key}"
            assert isinstance(results[key], (int, float)), f"Invalid type for {key}"
            assert not np.isnan(results[key]), f"NaN value for {key}"
        
        # Sanity check values
        assert results['psnr_mean'] > 0, "PSNR should be positive"
        assert 0 <= results['ssim_mean'] <= 1, "SSIM should be between 0 and 1"
        assert results['mse_mean'] >= 0, "MSE should be non-negative"
        
        print("✅ Evaluation integration works")
    
    def test_end_to_end_pipeline(self, temp_dir, sample_config):
        """Test complete end-to-end pipeline"""
        # Save config
        config_path = temp_dir / 'test_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(sample_config, f)
        
        # Load config
        config = load_config(config_path)
        
        # Create all components as in main experiment
        device = 'cpu'
        
        # Dataset
        dataset = SyntheticSRDataset(
            num_samples=config.data.num_samples,
            image_size=config.data.image_size,
            scale_factor=config.data.scale_factor,
            noise_level=config.data.noise_level
        )
        
        # Data loaders
        train_loader = DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        # Model
        model = SimpleSRCNN(
            use_adafm=config.model.use_adafm,
            use_correction=config.model.use_correction,
            num_channels=config.model.num_channels,
            num_filters=tuple(config.model.num_filters)
        )
        
        # Optimizer
        if config.optimizer.name == 'AdaFM':
            optimizer = AdaFMOptimizer(
                model.parameters(),
                gamma=1.0,
                delta=0.001,
                single_variable=True
            )
        else:
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=config.optimizer.lr,
                momentum=config.optimizer.momentum
            )
        
        # Loss and stability
        loss_fn = torch.nn.MSELoss()
        stability_analyzer = StabilityAnalyzer(
            model=model,
            loss_fn=loss_fn,
            L=1.0,
            alpha=0.1
        )
        
        # Trainer
        trainer = StabilityTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            stability_analyzer=stability_analyzer,
            device=device
        )
        
        # Short training
        history = trainer.fit_with_stability(
            train_loader=train_loader,
            num_epochs=1
        )
        
        # Verify pipeline worked
        assert len(history['train_loss']) == 1
        assert history['train_loss'][0] > 0
        
        # Test evaluation
        results = evaluate_model(model, train_loader, device='cpu')
        assert 'psnr_mean' in results
        assert 'ssim_mean' in results
        assert 'mse_mean' in results
        
        print("✅ End-to-end pipeline works")


class TestComponentCompatibility:
    """Test compatibility between different components"""
    
    def test_adafm_optimizer_with_models(self):
        """Test AdaFM optimizer with different model configurations"""
        model_configs = [
            {'use_correction': False, 'use_adafm': False},
            {'use_correction': True, 'use_adafm': False},
            {'use_correction': False, 'use_adafm': True},
            {'use_correction': True, 'use_adafm': True}
        ]
        
        for config in model_configs:
            model = SimpleSRCNN(num_filters=(8, 4), **config)
            optimizer = AdaFMOptimizer(
                model.parameters(),
                gamma=1.0,
                delta=0.001,
                single_variable=True
            )
            
            # Test optimization step
            x = torch.randn(2, 3, 16, 16)
            y_target = torch.randn(2, 3, 16, 16)
            
            optimizer.zero_grad()
            y_pred = model(x)
            loss = torch.nn.functional.mse_loss(y_pred, y_target)
            loss.backward()
            optimizer.step()
            
            # Verify optimizer state updated
            assert optimizer.t == 1
            assert optimizer.alpha_x > 0
        
        print("✅ AdaFM optimizer compatible with all model configs")
    
    def test_stability_analyzer_with_different_optimizers(self):
        """Test stability analyzer with different optimizers"""
        model = SimpleSRCNN(num_filters=(8, 4))
        loss_fn = torch.nn.MSELoss()
        
        analyzer = StabilityAnalyzer(
            model=model,
            loss_fn=loss_fn,
            L=1.0,
            alpha=0.1
        )
        
        # Test theoretical bounds with different learning rates
        learning_rates = [0.001, 0.01, 0.1]
        for lr in learning_rates:
            gamma = analyzer.compute_theoretical_gamma_general(100, 1000, lr)
            assert gamma > 0, f"Invalid gamma for lr={lr}"
            
            # Higher learning rates should give higher bounds
            if lr > 0.01:
                gamma_low = analyzer.compute_theoretical_gamma_general(100, 1000, 0.01)
                assert gamma > gamma_low, f"Gamma not increasing with lr: {gamma} vs {gamma_low}"
        
        print("✅ Stability analyzer works with different learning rates")
    
    def test_data_compatibility_with_models(self):
        """Test data compatibility with different model configurations"""
        # Test different image sizes and scale factors
        configs = [
            {'image_size': 16, 'scale_factor': 2},
            {'image_size': 32, 'scale_factor': 2},
            {'image_size': 32, 'scale_factor': 4}
        ]
        
        for config in configs:
            dataset = SyntheticSRDataset(
                num_samples=4,
                **config
            )
            
            model = SimpleSRCNN(num_filters=(8, 4))
            
            # Test forward pass
            lr, hr = dataset[0]
            lr_batch = lr.unsqueeze(0)
            sr = model(lr_batch)
            
            # Verify output shape matches input LR shape
            assert sr.shape == lr_batch.shape
            
            # Verify reasonable outputs
            assert torch.isfinite(sr).all(), "Model output contains NaN/Inf"
        
        print("✅ Data compatible with models across different configs")


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_invalid_config_handling(self):
        """Test handling of invalid configurations"""
        # Test missing required section
        invalid_config = {'model': {'name': 'test'}}
        
        # This should work (Config creation doesn't validate)
        config = Config(invalid_config)
        assert config.model.name == 'test'
        
        # But accessing missing sections should fail gracefully
        with pytest.raises(AttributeError):
            _ = config.optimizer.lr
        
        print("✅ Invalid config handled gracefully")
    
    def test_model_input_validation(self):
        """Test model input validation"""
        model = SimpleSRCNN()
        
        # Test with correct input
        valid_input = torch.randn(1, 3, 16, 16)
        output = model(valid_input)
        assert output.shape == valid_input.shape
        
        # Test with different batch sizes
        for batch_size in [1, 2, 4]:
            test_input = torch.randn(batch_size, 3, 16, 16)
            output = model(test_input)
            assert output.shape == test_input.shape
        
        print("✅ Model input validation works")
    
    def test_optimizer_state_persistence(self):
        """Test optimizer state save/load"""
        model = SimpleSRCNN(num_filters=(8, 4))
        optimizer = AdaFMOptimizer(
            model.parameters(),
            gamma=1.0,
            delta=0.001,
            single_variable=True
        )
        
        # Take a step to create state
        x = torch.randn(1, 3, 16, 16)
        y = torch.randn(1, 3, 16, 16)
        
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()
        
        # Save state
        state = optimizer.state_dict()
        assert isinstance(state, dict)
        assert 't' in state
        assert 'alpha_x' in state
        
        # Create new optimizer and load state
        new_optimizer = AdaFMOptimizer(
            model.parameters(),
            gamma=1.0,
            delta=0.001,
            single_variable=True
        )
        new_optimizer.load_state_dict(state)
        
        # Verify state transfer
        assert new_optimizer.t == optimizer.t
        assert new_optimizer.alpha_x == optimizer.alpha_x
        
        print("✅ Optimizer state persistence works")


def test_basic_imports():
    """Test that all basic imports work"""
    # Test model imports
    from src.models import SimpleSRCNN
    assert SimpleSRCNN is not None
    
    # Test optimizer imports
    from src.optimizers import AdaFMOptimizer
    assert AdaFMOptimizer is not None
    
    # Test data imports
    from src.data import SyntheticSRDataset
    assert SyntheticSRDataset is not None
    
    # Test stability imports
    from src.stability import StabilityAnalyzer
    assert StabilityAnalyzer is not None
    
    # Test utils imports
    from src.utils import load_config, evaluate_model
    assert load_config is not None
    assert evaluate_model is not None
    
    print("✅ All imports work correctly")


if __name__ == '__main__':
    # Run tests with verbose output
    pytest.main([__file__, '-v', '--tb=short'])
