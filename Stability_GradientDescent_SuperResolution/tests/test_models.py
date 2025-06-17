"""
Unit tests for model components
"""

import pytest
import torch
import torch.nn as nn

from src.models import SimpleSRCNN, AdaFMLayer, CorrectionFilter, SiameseNetwork


class TestSimpleSRCNN:
    """Test SimpleSRCNN model"""
    
    def test_initialization(self):
        """Test model initialization"""
        model = SimpleSRCNN()
        assert isinstance(model, nn.Module)
        assert model.use_correction == False
        assert model.use_adafm == False
    
    def test_forward_pass(self):
        """Test forward pass"""
        model = SimpleSRCNN()
        x = torch.randn(2, 3, 16, 16)  # LR input
        y = model(x)
        assert y.shape == (2, 3, 16, 16)  # Same size output
    
    def test_with_correction(self):
        """Test model with correction filter"""
        model = SimpleSRCNN(use_correction=True)
        x = torch.randn(2, 3, 16, 16)
        y = model(x)
        assert y.shape == (2, 3, 16, 16)
        assert hasattr(model, 'correction')
    
    def test_with_adafm(self):
        """Test model with AdaFM layers"""
        model = SimpleSRCNN(use_adafm=True)
        x = torch.randn(2, 3, 16, 16)
        y = model(x)
        assert y.shape == (2, 3, 16, 16)
        assert hasattr(model, 'adafm1')
        assert hasattr(model, 'adafm2')
    
    def test_full_system(self):
        """Test full system with all components"""
        model = SimpleSRCNN(use_correction=True, use_adafm=True)
        x = torch.randn(2, 3, 16, 16)
        y = model(x)
        assert y.shape == (2, 3, 16, 16)
    
    def test_num_params(self):
        """Test parameter counting"""
        model = SimpleSRCNN()
        num_params = model.get_num_params()
        assert num_params > 0


class TestAdaFMLayer:
    """Test AdaFM layer"""
    
    def test_initialization(self):
        """Test layer initialization"""
        layer = AdaFMLayer(64)
        assert layer.num_features == 64
        assert layer.gamma.shape == (1, 64, 1, 1)
        assert layer.beta.shape == (1, 64, 1, 1)
    
    def test_forward_pass(self):
        """Test forward pass"""
        layer = AdaFMLayer(32)
        x = torch.randn(4, 32, 16, 16)
        y = layer(x)
        assert y.shape == x.shape
    
    def test_modulation(self):
        """Test that modulation actually changes features"""
        layer = AdaFMLayer(32)
        x = torch.randn(4, 32, 16, 16)
        y = layer(x)
        assert not torch.allclose(x, y)


class TestCorrectionFilter:
    """Test correction filter"""
    
    def test_initialization(self):
        """Test filter initialization"""
        filter = CorrectionFilter(kernel_size=3)
        assert filter.kernel_size == 3
        assert filter.correction_kernel.shape == (3, 1, 3, 3)
    
    def test_identity_init(self):
        """Test identity initialization"""
        filter = CorrectionFilter(kernel_size=3, init_identity=True)
        # Check that center pixel is 1
        kernel = filter.correction_kernel
        assert kernel[0, 0, 1, 1] == 1.0
    
    def test_forward_pass(self):
        """Test forward pass"""
        filter = CorrectionFilter(kernel_size=3)
        x = torch.randn(2, 3, 32, 32)
        y = filter(x)
        assert y.shape == x.shape
    
    def test_spectrum(self):
        """Test frequency spectrum computation"""
        filter = CorrectionFilter(kernel_size=3)
        spectrum = filter.get_kernel_spectrum()
        assert spectrum.shape == (3, 3, 3)


class TestSiameseNetwork:
    """Test Siamese network"""
    
    def test_initialization(self):
        """Test network initialization"""
        net = SiameseNetwork()
        assert isinstance(net, nn.Module)
    
    def test_forward_one(self):
        """Test single branch forward"""
        net = SiameseNetwork()
        x = torch.randn(2, 3, 32, 32)
        feat = net.forward_one(x)
        assert feat.shape == (2, 128 * 4 * 4)
    
    def test_forward_pair(self):
        """Test full forward pass"""
        net = SiameseNetwork()
        x1 = torch.randn(2, 3, 32, 32)
        x2 = torch.randn(2, 3, 32, 32)
        similarity, (feat1, feat2) = net(x1, x2)
        
        assert similarity.shape == (2, 1)
        assert feat1.shape == (2, 128 * 4 * 4)
        assert feat2.shape == (2, 128 * 4 * 4)
        assert torch.all((similarity >= 0) & (similarity <= 1))
    
    def test_distance(self):
        """Test distance computation"""
        net = SiameseNetwork()
        x1 = torch.randn(2, 3, 32, 32)
        x2 = torch.randn(2, 3, 32, 32)
        dist = net.compute_distance(x1, x2)
        assert dist.shape == (2,)
        assert torch.all(dist >= 0)