#!/bin/bash

# SVDK Regression Environment Setup Script
# Based on actual code analysis

echo "🚀 Setting up SVDK Regression environment..."
echo "📋 Required packages: torch, gpytorch, numpy, pandas, scipy, scikit-learn, matplotlib, seaborn, plotly, tqdm, torchvision"

# Function to check if a package is installed
check_package() {
    python -c "import $1" 2>/dev/null && echo "✅ $1" || echo "❌ $1"
}

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "✅ Conda found. Creating conda environment..."
    
    # Create conda environment
    conda env create -f environment.yml --force
    
    # Activate environment
    echo "📦 Activating environment..."
    conda activate svdk-regression
    
    echo "✅ Environment 'svdk-regression' created and activated!"
    echo "💡 To activate in the future, run: conda activate svdk-regression"
    
elif command -v python3 &> /dev/null; then
    echo "⚠️  Conda not found. Using pip with virtual environment..."
    
    # Create virtual environment
    python3 -m venv svdk-regression-env
    
    # Activate environment
    source svdk-regression-env/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install requirements
    pip install -r requirements.txt
    
    echo "✅ Virtual environment 'svdk-regression-env' created and activated!"
    echo "💡 To activate in the future, run: source svdk-regression-env/bin/activate"
    
else
    echo "❌ Neither conda nor python3 found. Please install Python first."
    exit 1
fi

echo ""
echo "🔍 Verifying package installation..."

# Check all required packages
check_package torch
check_package gpytorch
check_package numpy
check_package pandas
check_package scipy
check_package sklearn
check_package matplotlib
check_package seaborn
check_package plotly
check_package tqdm
check_package torchvision

echo ""
echo "🎉 Environment setup complete!"
echo "📋 Next steps:"
echo "   1. Activate the environment (see above)"
echo "   2. Test with: python -c \"import torch, gpytorch; print('Setup successful!')\""
echo "   3. Run experiments: python svdkgpregressionmonotonicity.py --help"
echo "   4. Check the README.md for usage instructions"
