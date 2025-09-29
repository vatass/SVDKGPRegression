# SVDK Regression with Monotonicity Constraints

Stochastic Variational Deep Kernel (SVDK) Regression for brain imaging data with monotonicity constraints to ensure biologically plausible temporal trajectories.

## 🚀 Quick Start

### ✅ Recommended: Use Existing Environment

The project is **ready to use** with the existing `dkgp_env` conda environment:

```bash
# Activate the existing environment
conda activate dkgp_env

# Navigate to project directory
cd /home/cbica/Desktop/SVDKRegression

# Test the environment
python test_core_functionality.py

# Run the main SVDK code
python svdkgpregressionmonotonicity.py --help
```

### 🔧 Alternative: Fresh Installation

If you need to set up a new environment:

#### Option 1: Automated Setup
```bash
./setup_environment.sh
```

#### Option 2: Manual Setup

**Using Conda:**
```bash
conda env create -f environment.yml
conda activate svdk-regression
```

**Using pip:**
```bash
python3 -m venv svdk-regression-env
source svdk-regression-env/bin/activate  # On Windows: svdk-regression-env\Scripts\activate
pip install -r requirements.txt
```

## 📋 Requirements

### ✅ Core Requirements (Available in dkgp_env)
- **Python 3.8+** ✅
- **PyTorch 1.12.1** ✅
- **GPyTorch 1.10** ✅
- **NumPy 1.22.3** ✅
- **Pandas 1.2.3** ✅
- **Scikit-learn 1.3.0** ✅
- **SciPy 1.9.3** ✅

### ⚠️ Optional Requirements (for visualization)
- **Matplotlib** - For plotting (can be installed with `pip install matplotlib`)
- **Seaborn** - For statistical plots (can be installed with `pip install seaborn`)
- **Plotly** - For interactive plots (can be installed with `pip install plotly`)
- **TQDM** - For progress bars (can be installed with `pip install tqdm`)

## 🏃‍♂️ Usage

### Test Environment
```bash
# Test core functionality
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
├── monotonicity_experiments.sh       # Batch experiment script
├── test_core_functionality.py        # Core functionality test
├── test_environment.py               # Full environment test
├── requirements.txt                  # Python dependencies
├── environment.yml                   # Conda environment
├── setup_environment.sh             # Automated setup script
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
   - Make sure you're using `conda activate dkgp_env`
   - Check with: `conda env list`

3. **Permission errors**
   - Make scripts executable: `chmod +x *.sh`

## 📋 Quick Reference

**Start working immediately:**
```bash
conda activate dkgp_env
cd /home/cbica/Desktop/SVDKRegression
python test_core_functionality.py
python svdkgpregressionmonotonicity.py --help
```

**Add visualization (optional):**
```bash
pip install matplotlib seaborn plotly tqdm
```
