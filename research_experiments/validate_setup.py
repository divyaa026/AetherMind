"""
Quick validation script to test the research experiment suite setup.

This script performs basic checks to ensure all modules can be imported
and the configuration is valid.
"""

import sys
from pathlib import Path

def check_imports():
    """Check if all required modules can be imported."""
    print("Checking module imports...")
    
    try:
        import yaml
        print("✓ yaml")
    except ImportError:
        print("✗ yaml - Install with: pip install pyyaml")
        return False
    
    try:
        import numpy as np
        print("✓ numpy")
    except ImportError:
        print("✗ numpy - Install with: pip install numpy")
        return False
    
    try:
        import pandas as pd
        print("✓ pandas")
    except ImportError:
        print("✗ pandas - Install with: pip install pandas")
        return False
    
    try:
        import torch
        print(f"✓ torch (version {torch.__version__})")
    except ImportError:
        print("✗ torch - Install with: pip install torch")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✓ matplotlib")
    except ImportError:
        print("✗ matplotlib - Install with: pip install matplotlib")
        return False
    
    try:
        import sklearn
        print("✓ scikit-learn")
    except ImportError:
        print("✗ scikit-learn - Install with: pip install scikit-learn")
        return False
    
    return True

def check_config():
    """Check if config.yaml is valid."""
    print("\nChecking configuration file...")
    
    try:
        import yaml
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required sections
        required_sections = ['paths', 'data', 'federated', 'model', 'training', 
                           'privacy', 'experiments']
        
        for section in required_sections:
            if section in config:
                print(f"✓ {section} section found")
            else:
                print(f"✗ {section} section missing")
                return False
        
        return True
        
    except FileNotFoundError:
        print("✗ config.yaml not found")
        return False
    except yaml.YAMLError as e:
        print(f"✗ Invalid YAML: {e}")
        return False

def check_experiment_modules():
    """Check if all experiment modules exist."""
    print("\nChecking experiment modules...")
    
    modules = [
        'experiment_data_quality.py',
        'experiment_convergence.py',
        'experiment_privacy_tradeoff.py',
        'experiment_robustness.py'
    ]
    
    all_exist = True
    for module in modules:
        if Path(module).exists():
            print(f"✓ {module}")
        else:
            print(f"✗ {module} not found")
            all_exist = False
    
    return all_exist

def check_directory_structure():
    """Check if directory structure is correct."""
    print("\nChecking directory structure...")
    
    dirs = ['results', 'results/figures']
    
    all_exist = True
    for dir_path in dirs:
        path = Path(dir_path)
        if path.exists() and path.is_dir():
            print(f"✓ {dir_path}/")
        else:
            print(f"⚠ {dir_path}/ not found (will be created automatically)")
    
    return True

def main():
    """Run all validation checks."""
    print("="*60)
    print("Research Experiment Suite - Setup Validation")
    print("="*60)
    print()
    
    checks = [
        ("Dependencies", check_imports),
        ("Configuration", check_config),
        ("Experiment Modules", check_experiment_modules),
        ("Directory Structure", check_directory_structure)
    ]
    
    results = []
    
    for check_name, check_func in checks:
        result = check_func()
        results.append(result)
        print()
    
    print("="*60)
    if all(results):
        print("✓ ALL CHECKS PASSED")
        print("="*60)
        print("\nYou can now run the experiments:")
        print("  python run_research_experiments.py")
        print()
        return 0
    else:
        print("✗ SOME CHECKS FAILED")
        print("="*60)
        print("\nPlease fix the issues above before running experiments.")
        print("Install missing dependencies:")
        print("  pip install -r requirements.txt")
        print()
        return 1

if __name__ == "__main__":
    sys.exit(main())
