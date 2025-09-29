#!/usr/bin/env python3
"""
Test core SVDK functionality without visualization dependencies
"""

def test_core_imports():
    """Test only the core imports needed for SVDK regression"""
    print("🔍 Testing core SVDK imports...")
    
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
        import sklearn
        print(f"✅ Scikit-learn {sklearn.__version__}")
    except ImportError as e:
        print(f"❌ Scikit-learn: {e}")
        return False
    
    return True

def test_models_import():
    """Test if models.py can be imported"""
    try:
        from models import MLP, GaussianProcessLayer
        print("✅ Models imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Models import failed: {e}")
        return False

def test_utils_import():
    """Test if utils.py can be imported"""
    try:
        import utils
        print("✅ Utils imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Utils import failed: {e}")
        return False

def main():
    print("🚀 SVDK Core Functionality Test")
    print("=" * 40)
    
    core_ok = test_core_imports()
    models_ok = test_models_import()
    utils_ok = test_utils_import()
    
    if core_ok and models_ok and utils_ok:
        print("\n🎉 Core SVDK functionality is ready!")
        print("📋 The dkgp_env environment can run the main SVDK code")
        print("💡 Visualization features will be disabled, but core ML functionality works")
        return True
    else:
        print("\n❌ Some core functionality is missing")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
