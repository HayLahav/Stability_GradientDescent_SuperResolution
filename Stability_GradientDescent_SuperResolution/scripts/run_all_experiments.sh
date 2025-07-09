#!/bin/bash
# Run all stability analysis experiments

echo "=== Running Stability Analysis Experiments ==="
echo

# Set PYTHONPATH to ensure imports work
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Create results directory
mkdir -p results/figures

# Function to check if previous command succeeded
check_status() {
    if [ $? -eq 0 ]; then
        echo "✅ $1 completed successfully"
    else
        echo "❌ $1 failed"
        exit 1
    fi
}

# Run baseline experiment
echo "1. Running baseline experiment..."
python experiments/experiments_main.py --config experiments/configs/baseline.yaml
check_status "Baseline experiment"
echo

# Run correction filter experiment  
echo "2. Running correction filter experiment..."
python experiments/experiments_main.py --config experiments/configs/correction.yaml
check_status "Correction filter experiment"
echo

# Run AdaFM optimizer experiment
echo "3. Running AdaFM optimizer experiment..."
python experiments/experiments_main.py --config experiments/configs/adafm_opt.yaml
check_status "AdaFM optimizer experiment"
echo

# Run full system experiment
echo "4. Running full system experiment..."
python experiments/experiments_main.py --config experiments/configs/full_system.yaml
check_status "Full system experiment"
echo

# Run theoretical validation
echo "5. Running theoretical validation..."
python experiments/experiments_theoretical.py
check_status "Theoretical validation"
echo

# Run minimax demonstration
echo "6. Running minimax optimization demo..."
python experiments/experiments_minimax_demo.py
check_status "Minimax demonstration"
echo

# Generate comparison plots
echo "7. Generating comparison plots..."
python scripts/generate_report_figures.py
check_status "Plot generation"
echo

echo "=== All experiments completed successfully! ==="
echo "📁 Results saved to: results/"
echo "📊 Figures saved to: results/figures/"
echo ""
echo "To view results:"
echo "  - Open notebooks/experiment-visualization-notebook.ipynb"
echo "  - Check results/ directory for detailed outputs"
echo "  - View figures in results/figures/"
