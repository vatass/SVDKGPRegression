#!/bin/bash

# SVDK Regression Environment Setup Script
echo "🚀 Setting up SVDK Regression environment..."

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "✅ Conda found. Creating conda environment..."
    
    # Create conda environment
    conda env create -f environment.yml
    
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
echo "🎉 Environment setup complete!"
echo "📋 Next steps:"
echo "   1. Activate the environment (see above)"
echo "   2. Run: python svdkgpregressionmonotonicity.py --help"
echo "   3. Check the README.md for usage instructions"
