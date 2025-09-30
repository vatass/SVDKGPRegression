#!/usr/bin/env python3
"""
Test script to verify SVDK Regression environment setup
"""

def test_imports():
    """Test all required imports"""
    print("🔍 Testing package imports...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch: {e}")
        return False
    
    try:
        import gpytorch
        print(f"✅ GPyTorch {gpytorch.__version__}")
    except ImportError as e:
        print(f"❌ GPyTorch: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"✅ Pandas {pd.__version__}")
    except ImportError as e:
        print(f"❌ Pandas: {e}")
        return False
    
    try:
        import scipy
        print(f"✅ SciPy {scipy.__version__}")
    except ImportError as e:
        print(f"❌ SciPy: {e}")
        return False
    
    try:
        import sklearn
        print(f"✅ Scikit-learn {sklearn.__version__}")
    except ImportError as e:
        print(f"❌ Scikit-learn: {e}")
        return False
    
    try:
        import matplotlib
        print(f"✅ Matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"❌ Matplotlib: {e}")
        return False
    
    try:
        import seaborn as sns
        print(f"✅ Seaborn {sns.__version__}")
    except ImportError as e:
        print(f"❌ Seaborn: {e}")
        return False
    
    try:
        import plotly
        print(f"✅ Plotly {plotly.__version__}")
    except ImportError as e:
        print(f"❌ Plotly: {e}")
        return False
    
    try:
        import tqdm
        print(f"✅ TQDM {tqdm.__version__}")
    except ImportError as e:
        print(f"❌ TQDM: {e}")
        return False
    
    try:
        import torchvision
        print(f"✅ TorchVision {torchvision.__version__}")
    except ImportError as e:
        print(f"❌ TorchVision: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality of key packages"""
    print("\n🧪 Testing basic functionality...")
    
    try:
        import torch
        import numpy as np
        
        # Test PyTorch tensor creation
        x = torch.randn(10, 5)
        print(f"✅ PyTorch tensor creation: {x.shape}")
        
        # Test NumPy array creation
        y = np.random.randn(10, 5)
        print(f"✅ NumPy array creation: {y.shape}")
        
        # Test basic GPyTorch import
        import gpytorch
        print("✅ GPyTorch basic import successful")
        
        return True
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def main():
    print("🚀 SVDK Regression Environment Test")
    print("=" * 50)
    
    # Test imports
    imports_ok = test_imports()
    
    if imports_ok:
        # Test basic functionality
        functionality_ok = test_basic_functionality()
        
        if functionality_ok:
            print("\n🎉 All tests passed! Environment is ready.")
            print("📋 You can now run:")
            print("   python svdkgpregressionmonotonicity.py --help")
            return True
        else:
            print("\n⚠️  Imports successful but functionality test failed.")
            return False
    else:
        print("\n❌ Import tests failed. Please check your environment setup.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
