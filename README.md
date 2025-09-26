# SVDK Regression with Monotonicity Constraints

Stochastic Variational Deep Kernel (SVDK) Regression for brain imaging data with monotonicity constraints to ensure biologically plausible temporal trajectories.

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)
```bash
./setup_environment.sh
```

### Option 2: Manual Setup

#### Using Conda
```bash
conda env create -f environment.yml
conda activate svdk-regression
```

#### Using pip
```bash
python3 -m venv svdk-regression-env
source svdk-regression-env/bin/activate  # On Windows: svdk-regression-env\Scripts\activate
pip install -r requirements.txt
```

## 📋 Requirements

- Python 3.9+
- PyTorch 1.9+
- GPyTorch 1.6+
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn, Plotly (for visualization)

## 🏃‍♂️ Usage

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
├── monotonicity_experiments.sh       # Batch experiment script
├── requirements.txt                  # Python dependencies
├── environment.yml                   # Conda environment
└── setup_environment.sh             # Automated setup script
```

## 🧠 Key Features

- **Monotonicity Constraints**: Enforces biologically plausible temporal trajectories
- **Gaussian Process Integration**: Uncertainty quantification with GPyTorch
- **Deep Feature Extraction**: Neural networks for imaging data
- **Longitudinal Analysis**: Handles temporal brain imaging data
- **Cross-validation**: Robust evaluation with fold-based testing
- **Visualization**: Comprehensive plotting and analysis tools

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

### Code Style
```bash
black *.py  # Format code
flake8 *.py # Lint code
```

### Testing
```bash
pytest tests/  # Run tests (when available)
```

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
