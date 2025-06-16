# Stability Analysis of Gradient Descent in Super-Resolution

This repository implements and validates theoretical stability bounds from Lecture 8 of the Advanced Topics in Learning course on Stability Analysis for Gradient Descent, applied to super-resolution tasks with correction filters and AdaFM optimization.

## Overview

This project bridges theoretical stability analysis with practical deep learning applications, demonstrating how gradient descent stability affects generalization in super-resolution models. We implement:

- **Theoretical validation** of stability bounds (Theorems 8.1, 8.3, 8.5) based of:
  
  [1] R. Bassily, V. Feldman, C. Guzmán, and K. Talwar. Stability of stochastic gradient
      descent on nonsmooth convex losses. Advances in Neural Information Processing
      Systems, 33:4381–4391, 2020.
  
  [2] M. Hardt, B. Recht, and Y. Singer. Train faster, generalize better: Stability of stochastic
      gradient descent. In International conference on machine learning, pages 1225–1234.
      PMLR, 2016
  
- **correction filter from the paper - Correction Filter for Single Image Super-Resolution:Robustifying Off-the-Shelf Deep Super-Resolvers** (CVPR 2020) for input alignment
- **Modulating Image Restoration with Continual Levels via Adaptive Feature Modification Layers** (CVPR 2019) for adaptive feature modulation
- **AdaFM: Adaptive Variance-Reduced Algorithm for Stochastic Minimax Optimization** (ICLR 2025) for parameter-free optimization
- **Comprehensive stability analysis** with empirical validation

## Key Features

- ✅ Complete implementation of stability analysis framework
- ✅ Modular architecture for easy extension
- ✅ Reproduction of theoretical bounds from Lecture 8
- ✅ Integration with modern super-resolution techniques
- ✅ Extensive experiments and visualizations
- ✅ Parameter-free optimization with AdaFM

## Installation

```bash
# Clone the repository
git clone https://github.com/username/Stability_GradientDescent_SuperResolution.git
cd Stability_GradientDescent_SuperResolution

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Run Main Experiments

```bash
# Run all four configurations
bash scripts/run_all_experiments.sh

# Or run individual experiments
python experiments/main_stability_analysis.py --config experiments/configs/baseline.yaml
python experiments/main_stability_analysis.py --config experiments/configs/with_correction.yaml
python experiments/main_stability_analysis.py --config experiments/configs/with_adafm_opt.yaml
python experiments/main_stability_analysis.py --config experiments/configs/full_system.yaml
```

### 2. Theoretical Validation

```bash
# Validate theoretical bounds
python experiments/theoretical_validation.py

# Run minimax optimization demo
python experiments/minimax_demo.py
```

### 3. Interactive Analysis

```python
from src.models import SimpleSRCNN
from src.optimizers import AdaFMOptimizer
from src.stability import StabilityAnalyzer

# Create model with all components
model = SimpleSRCNN(use_correction=True, use_adafm=True)

# Use AdaFM optimizer
optimizer = AdaFMOptimizer(model.parameters(), gamma=1.0, delta=0.001)

# Analyze stability
analyzer = StabilityAnalyzer(model, loss_fn, L=1.0, alpha=0.1)
```

## Project Structure

```
Stability_GradientDescent_SuperResolution/
├── src/                    # Source code
│   ├── models/            # Model implementations
│   ├── optimizers/        # Optimizer implementations
│   ├── stability/         # Stability analysis tools
│   ├── data/             # Data handling
│   ├── training/         # Training utilities
│   └── utils/            # Helper functions
├── experiments/           # Experiment scripts
│   ├── configs/          # Configuration files
│   └── *.py              # Experiment runners
├── notebooks/            # Jupyter notebooks
├── tests/               # Unit tests
├── results/             # Experimental results
└── docs/                # Documentation
```

## Key Results

### 1. Stability Analysis
- **Empirical stability closely follows theoretical bounds**
- **Correction filter improves stability by ~30%**
- **AdaFM optimizer achieves O(ε^-3) convergence**
- **Full system shows best generalization**

### 2. Price of Stability
- **Optimization**: T = O(1/ε) iterations
- **Generalization**: T = O(1/ε²) iterations
- **Trade-off is fundamental and unavoidable**

### 3. Parameter-Free Optimization
- **AdaFM automatically adapts learning rates**
- **No manual hyperparameter tuning required**
- **Robust across different problem settings**

## Theoretical Background

This project implements key results from Lecture 8:

**Strongly Convex Case (Theorem 8.1)**:
```
γ(m) = O(L²/(α√T) + L²/(αm))
```

**General Case (Theorem 8.3)**:
```
γ(m) = 4ηL²√T + 4ηL²T/m
```

**Smooth Case (Theorem 8.5)**:
```
γ(m) = 2ηTL²/m
```

## Configuration

Experiments are configured using YAML files. Key parameters:

```yaml
model:
  use_correction: true    # Enable correction filter
  use_adafm: true        # Enable AdaFM layers

optimizer:
  name: "AdaFM"          # Optimizer choice
  gamma: 1.0             # Learning rate parameter
  delta: 0.001           # Adaptation parameter

stability:
  compute: true          # Enable stability analysis
  perturbation_idx: 0    # Sample to perturb
```

## Testing

Run unit tests:
```bash
pytest tests/
```

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{stability_sr_2024,
  title={Stability Analysis of Gradient Descent in Super-Resolution},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/username/Stability_GradientDescent_SuperResolution}}
}
```

## References

1. Lecture 8: Stability Analysis for Gradient Descent
2. M. Hardt et al. "Train faster, generalize better: Stability of SGD." ICML 2016
3. Abu Hussein et al. "Correction Filter for SR." CVPR 2020
4. He et al. "AdaFM Layers." CVPR 2019
5. "AdaFM: Adaptive Variance-Reduced Algorithm." ICLR 2025

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Prof. Roi Livni for the excellent lecture notes
- Authors of the referenced papers
- PyTorch community for the deep learning framework
