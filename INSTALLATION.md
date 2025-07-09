# Installation Guide

This guide provides step-by-step instructions for setting up the Stability Analysis project.

## Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **Memory**: At least 8GB RAM recommended
- **Storage**: 2GB free space for datasets and results
- **GPU**: Optional but recommended (CUDA-compatible)

### Platform Support
- ✅ Linux (Ubuntu 18.04+, CentOS 7+)
- ✅ macOS (10.15+)
- ✅ Windows 10/11 (with WSL2 recommended)

## Installation Methods

### Method 1: Standard Installation (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/Stability_GradientDescent_SuperResolution.git
cd Stability_GradientDescent_SuperResolution

# 2. Create and activate virtual environment
python -m venv stability_env

# Activate (Linux/macOS)
source stability_env/bin/activate

# Activate (Windows)
stability_env\Scripts\activate

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install the package
pip install -e .

# 5. Verify installation
python -c "from src.models import SimpleSRCNN; print('✅ Installation successful!')"
```

### Method 2: Development Installation

For contributors and developers:

```bash
# Follow steps 1-3 from Method 1, then:

# 4. Install with development dependencies
pip install -e ".[dev]"

# 5. Install pre-commit hooks
pre-commit install

# 6. Run tests to verify
pytest tests/ -v
```

### Method 3: Docker Installation

```bash
# 1. Clone repository
git clone https://github.com/yourusername/Stability_GradientDescent_SuperResolution.git
cd Stability_GradientDescent_SuperResolution

# 2. Build Docker image
docker build -t stability-analysis .

# 3. Run container
docker run -it --gpus all -v $(pwd)/results:/app/results stability-analysis
```

## Dependency Details

### Core Dependencies
- **PyTorch** (≥2.0.0): Deep learning framework
- **NumPy** (≥1.21.0): Numerical computing
- **Matplotlib** (≥3.5.0): Plotting and visualization
- **PyYAML** (≥6.0): Configuration file handling
- **tqdm** (≥4.64.0): Progress bars

### Optional Dependencies

#### Visualization & Analysis
```bash
pip install -e ".[viz]"
```
- **Plotly**: Interactive plots
- **TensorBoard**: Training monitoring
- **Seaborn**: Statistical visualizations

#### Full Feature Set
```bash
pip install -e ".[full]"
```
- **OpenCV**: Advanced image processing
- **Scikit-image**: Scientific image analysis

## Environment Setup

### CUDA Setup (GPU Support)

#### Check CUDA Availability
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
```

#### Install CUDA-compatible PyTorch
```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1  
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Jupyter Setup (Optional)

```bash
# Install Jupyter
pip install jupyter ipywidgets

# Install kernel
python -m ipykernel install --user --name=stability_env

# Start Jupyter
jupyter lab
```

## Verification

### Quick Test
```bash
python -c "
from src.models import SimpleSRCNN
from src.data import SyntheticSRDataset
from src.optimizers import AdaFMOptimizer
print('✅ All core components imported successfully!')
"
```

### Run Sample Experiment
```bash
# Create minimal test config
cat > test_config.yaml << EOF
model:
  name: "Test"
  use_correction
