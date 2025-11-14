"""
STEP 1: Environment Setup & Dependency Check
============================================
This script checks and installs all required packages for Physics-Informed GNN
"""

import sys
import subprocess
import importlib

print("="*80)
print("STEP 1: ENVIRONMENT SETUP")
print("="*80)

# Required packages
REQUIRED_PACKAGES = {
    'pandas': 'pandas',
    'numpy': 'numpy',
    'matplotlib': 'matplotlib',
    'seaborn': 'seaborn',
    'scipy': 'scipy',
    'sklearn': 'scikit-learn',
    'torch': 'torch',
    'torch_geometric': 'torch-geometric'
}

def check_package(package_name, import_name):
    """Check if package is installed"""
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False

def install_package(package_name):
    """Install package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return True
    except:
        return False

print("\n[1/3] Checking installed packages...")
print("-" * 80)

missing_packages = []
installed_packages = []

for import_name, package_name in REQUIRED_PACKAGES.items():
    print(f"\nChecking {package_name}...", end=" ")
    if check_package(package_name, import_name):
        print("✓ INSTALLED")
        installed_packages.append(package_name)
    else:
        print("✗ MISSING")
        missing_packages.append(package_name)

print("\n" + "-" * 80)
print(f"\nInstalled: {len(installed_packages)}/{len(REQUIRED_PACKAGES)}")
print(f"Missing: {len(missing_packages)}")

if missing_packages:
    print("\n[2/3] Installing missing packages...")
    print("-" * 80)
    
    for package in missing_packages:
        print(f"\nInstalling {package}...")
        
        # Special handling for PyTorch Geometric
        if package == 'torch-geometric':
            print("\n⚠ PyTorch Geometric requires special installation!")
            print("\nPlease run these commands manually:")
            print("\n  # For CPU:")
            print("  pip install torch torchvision torchaudio")
            print("  pip install torch-geometric")
            print("\n  # For GPU (CUDA 11.8):")
            print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
            print("  pip install torch-geometric")
            print("\nAfter installation, re-run this script.")
            continue
        
        success = install_package(package)
        if success:
            print(f"  ✓ {package} installed successfully")
        else:
            print(f"  ✗ Failed to install {package}")
            print(f"    Try manually: pip install {package}")

print("\n[3/3] Final verification...")
print("-" * 80)

# Verify all packages
all_installed = True
for import_name, package_name in REQUIRED_PACKAGES.items():
    installed = check_package(package_name, import_name)
    status = "✓" if installed else "✗"
    print(f"{status} {package_name}")
    if not installed:
        all_installed = False

print("\n" + "="*80)

if all_installed:
    print("SUCCESS! All packages installed correctly")
    print("="*80)
    print("\n✓ Environment is ready!")
    print("\nNext step: Type 'next' to proceed to Step 2 (Data Loading)")
else:
    print("WARNING! Some packages are missing")
    print("="*80)
    print("\n⚠ Please install missing packages before proceeding")
    print("\nFor PyTorch Geometric:")
    print("  Visit: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html")
    print("\nFor other packages:")
    print("  pip install <package-name>")

# Test imports
print("\n" + "="*80)
print("TESTING IMPORTS")
print("="*80)

try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    print("\n✓ Basic packages working")
    
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        
        try:
            import torch_geometric
            print(f"✓ PyTorch Geometric version: {torch_geometric.__version__}")
            print("\n🎉 ALL SYSTEMS GO! Ready for GNN implementation!")
        except ImportError:
            print("\n⚠ PyTorch Geometric not found")
            print("  This is required for GNN implementation")
            print("  Install with: pip install torch-geometric")
    except ImportError:
        print("\n⚠ PyTorch not found")
        print("  Install with: pip install torch")
        
except ImportError as e:
    print(f"\n✗ Import error: {e}")
    print("  Please install missing packages")

print("\n" + "="*80)
print("STEP 1 COMPLETE")
print("="*80)
print("\nWhen all packages are installed, type 'next' to continue to Step 2")
print("="*80)

# Save installation log
import os
OUTPUT_DIR = r'F:\concrete data\test 3\gnn_implementation'
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(os.path.join(OUTPUT_DIR, 'step1_installation_log.txt'), 'w') as f:
    f.write("STEP 1: Environment Setup Log\n")
    f.write("="*80 + "\n\n")
    f.write("Required Packages:\n")
    for import_name, package_name in REQUIRED_PACKAGES.items():
        installed = check_package(package_name, import_name)
        status = "INSTALLED" if installed else "MISSING"
        f.write(f"  {package_name}: {status}\n")
    f.write("\n")
    
    try:
        import torch
        f.write(f"PyTorch Version: {torch.__version__}\n")
        f.write(f"CUDA Available: {torch.cuda.is_available()}\n")
    except:
        f.write("PyTorch: Not installed\n")
    
    try:
        import torch_geometric
        f.write(f"PyTorch Geometric Version: {torch_geometric.__version__}\n")
    except:
        f.write("PyTorch Geometric: Not installed\n")

print(f"\n✓ Installation log saved to: {OUTPUT_DIR}/step1_installation_log.txt")