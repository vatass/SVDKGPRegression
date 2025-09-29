# SVDK Regression with Monotonicity Constraints

Stochastic Variational Deep Kernel (SVDK) Regression for brain imaging data with monotonicity constraints to ensure biologically plausible temporal trajectories.

## 🚀 Quick Start

### Automated Setup (Recommended)
```bash
# Clone the repository
git clone https://github.com/vatass/SVDKGPRegression.git
cd SVDKGPRegression

# Make setup script executable
chmod +x setup_environment.sh

# Run automated setup
./setup_environment.sh

# Activate the environment
conda activate svdk-regression

# Test the installation
python test_core_functionality.py
```

### Manual Setup

**Using Conda (Recommended):**
```bash
# Clone the repository
git clone https://github.com/vatass/SVDKGPRegression.git
cd SVDKGPRegression

# Create environment with exact specifications
conda env create -f environment.yml

# Activate environment
conda activate svdk-regression

# Test installation
python test_core_functionality.py
```

**Using pip (Alternative):**
```bash
# Clone the repository
git clone https://github.com/vatass/SVDKGPRegression.git
cd SVDKGPRegression

# Create virtual environment
python3 -m venv svdk-regression-env

# Activate environment
source svdk-regression-env/bin/activate  # On Windows: svdk-regression-env\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Test installation
python test_core_functionality.py
```

## 📋 System Requirements

### Minimum Requirements
- **Python**: 3.8+ (tested with Python 3.8.20)
- **Operating System**: Linux, macOS, or Windows
- **Memory**: 8GB RAM minimum, 16GB recommended
- **Storage**: 2GB free space

### Required Software
- **Conda** (recommended) or **Python 3.8+**
- **Git** (for cloning the repository)

## 📦 Package Versions

The environment is configured with exact versions tested and working:

### Core Dependencies
- **Python**: 3.8.20
- **PyTorch**: 1.12.1 (CPU version)
- **GPyTorch**: 1.10.0
- **NumPy**: 1.22.3
- **Pandas**: 1.2.3
- **Scikit-learn**: 1.3.0
- **SciPy**: 1.9.3

### Optional Dependencies (for visualization)
- **Matplotlib**: ≥3.4.0
- **Seaborn**: ≥0.11.0
- **Plotly**: ≥5.0.0
- **TQDM**: ≥4.62.0

## 🏃‍♂️ Usage

### Test Environment
```bash
# Test core functionality (works without visualization packages)
python test_core_functionality.py

# Test full environment (requires visualization packages)
python test_environment.py
```

### Run Monotonicity Experiments
```bash
# Run with different penalty weights
python svdkgpregressionmonotonicity.py --lambda_penalty 0.0  # Baseline
python svdkgpregressionmonotonicity.py --lambda_penalty 5.0  # Moderate constraint
python svdkgpregressionmonotonicity.py --lambda_penalty 10.0 # Strong constraint
python svdkgpregressionmonotonicity.py --lambda_penalty 100.0 # Very strong constraint

# Or use the batch script
./monotonicity_experiments.sh
```

### Run Other Model Variants
```bash
# Base SVDK regression
python svdkregressiono1.py --help

# Multitask learning
python multitasksvdkregressionclassification.py --help

# Progression-informed learning
python svdkprogressionoinformed.py --help
```

### Evaluate Results
```bash
python evaluate_monotonicity.py
```

## 📁 Project Structure

```
SVDKRegression/
├── svdkgpregressionmonotonicity.py    # Main monotonic SVDK implementation
├── svdkregression.py                  # Base SVDK regression
├── svdkregressiono1.py               # Original SVDK version
├── multitasksvdkregressionclassification.py  # Multitask learning
├── svdkprogressionoinformed.py       # Progression-informed learning
├── evaluate_monotonicity.py          # Evaluation and visualization
├── models.py                         # Neural network models
├── functions.py                      # Utility functions
├── datasets.py                       # Dataset handling
├── utils.py                          # Utilities
├── monotonicity_experiments.sh       # Batch experiment script
├── test_core_functionality.py        # Core functionality test
├── test_environment.py               # Full environment test
├── requirements.txt                  # Python dependencies (pip)
├── environment.yml                   # Conda environment specification
├── setup_environment.sh             # Automated setup script
└── README.md                        # This file
```

## 🧠 Key Features

- **Monotonicity Constraints**: Enforces biologically plausible temporal trajectories
- **Gaussian Process Integration**: Uncertainty quantification with GPyTorch
- **Deep Feature Extraction**: Neural networks for imaging data
- **Longitudinal Analysis**: Handles temporal brain imaging data
- **Cross-validation**: Robust evaluation with fold-based testing
- **Visualization**: Comprehensive plotting and analysis tools (optional)

## 📊 Brain Regions Analyzed

- Amygdala (Left & Right)
- Hippocampus (Left & Right)
- Lateral Ventricle (Right)
- Parahippocampal Gyrus (Right)
- Thalamus Proper (Right)

## 🔬 Model Variants

1. **`svdkgpregressionmonotonicity.py`** - Main implementation with monotonicity constraints
2. **`svdkregression.py`** - Base SVDK regression model
3. **`svdkregressiono1.py`** - Original SVDK implementation
4. **`multitasksvdkregressionclassification.py`** - Multitask learning (regression + classification)
5. **`svdkprogressionoinformed.py`** - Progression-informed metric learning

## 📈 Performance Results

Monotonicity constraints consistently improve performance:
- **λ = 0**: MSE = 0.102, MAE = 0.233, R² = 0.902 (baseline)
- **λ = 5**: MSE = 0.089, MAE = 0.223, R² = 0.914 (improved)
- **λ = 10**: MSE = 0.087, MAE = 0.221, R² = 0.916 (best)

## 🛠️ Development

### Environment Status
- **✅ Core ML functionality**: Fully working
- **✅ SVDK regression algorithms**: Ready to use
- **✅ Monotonicity constraints**: Supported
- **⚠️ Visualization features**: Optional (can be added with `pip install matplotlib seaborn plotly tqdm`)

### Code Style
```bash
black *.py  # Format code
flake8 *.py # Lint code
```

### Testing
```bash
# Test core functionality
python test_core_functionality.py

# Test full environment (requires visualization packages)
python test_environment.py
```

## 🔧 Troubleshooting

### Common Issues

1. **ImportError: No module named 'matplotlib'**
   - This is expected if visualization packages aren't installed
   - Core functionality works without them
   - Install with: `pip install matplotlib seaborn plotly tqdm`

2. **Environment not found**
   - Make sure you're using `conda activate svdk-regression`
   - Check with: `conda env list`

3. **Permission errors**
   - Make scripts executable: `chmod +x *.sh`

4. **Conda environment creation fails**
   - Try: `conda clean --all` then retry
   - Or use pip installation as alternative

5. **PyTorch installation issues**
   - The environment uses CPU-only PyTorch
   - For GPU support, modify `environment.yml` to use GPU PyTorch

### Getting Help

If you encounter issues:
1. Check the troubleshooting section above
2. Run `python test_core_functionality.py` to diagnose problems
3. Open an issue on GitHub with error details
4. Include your system information (OS, Python version, etc.)

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{svdk_regression_monotonic,
  title={SVDK Regression with Monotonicity Constraints for Brain Imaging},
  author={Your Name},
  year={2024},
  url={https://github.com/vatass/SVDKGPRegression}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For questions or issues, please open an issue on GitHub or contact the maintainers.

---

## 📋 Quick Reference

**Complete setup for new users:**
```bash
git clone https://github.com/vatass/SVDKGPRegression.git
cd SVDKGPRegression
chmod +x setup_environment.sh
./setup_environment.sh
conda activate svdk-regression
python test_core_functionality.py
python svdkgpregressionmonotonicity.py --help
```

**Add visualization (optional):**
```bash
pip install matplotlib seaborn plotly tqdm
```
