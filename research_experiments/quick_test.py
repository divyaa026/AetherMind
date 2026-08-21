#!/usr/bin/env python3
"""
Quick Test Script

Run experiments with minimal parameters for fast validation.
Use this to verify the experiment suite works before running full experiments.

Usage:
    python quick_test.py
"""

import sys
import time
from pathlib import Path

def main():
    print("="*70)
    print("QUICK TEST - Research Experiment Suite")
    print("="*70)
    print()
    print("This will run all experiments with REDUCED parameters for testing.")
    print("Expected runtime: 5-15 minutes (depending on hardware)")
    print()
    print("Configuration:")
    print("  - 3 clients (instead of 10)")
    print("  - 10 rounds (instead of 50)")
    print("  - Smaller model (32 hidden units)")
    print("  - 3 epsilon values (instead of 5)")
    print("  - CPU only")
    print()
    
    response = input("Continue with quick test? (y/n): ")
    
    if response.lower() != 'y':
        print("Test cancelled.")
        return 0
    
    print("\nStarting quick test...")
    print("-"*70)
    
    start_time = time.time()
    
    # Import and run
    try:
        from run_research_experiments import ExperimentRunner
        
        runner = ExperimentRunner(config_path='config_quick_test.yaml')
        results = runner.run_all_experiments()
        
        elapsed = time.time() - start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        
        print("\n" + "="*70)
        print("[OK] QUICK TEST COMPLETED SUCCESSFULLY")
        print("="*70)
        print(f"\nTotal time: {minutes}m {seconds}s")
        print(f"\nResults saved to: {runner.results_dir}")
        print(f"Summary report: {runner.results_dir / 'research_summary.md'}")
        print()
        print("The full experiment suite is working correctly!")
        print("For production runs, use:")
        print("  python run_research_experiments.py")
        print()
        
        return 0
        
    except Exception as e:
        elapsed = time.time() - start_time
        print("\n" + "="*70)
        print("[FAILED] QUICK TEST FAILED")
        print("="*70)
        print(f"\nError after {elapsed:.1f} seconds:")
        print(f"{str(e)}")
        print()
        print("Please check:")
        print("1. All dependencies installed: pip install -r requirements.txt")
        print("2. Configuration file is valid: config_quick_test.yaml")
        print("3. Check the log file: results/experiment.log")
        print()
        
        import traceback
        print("Full traceback:")
        print(traceback.format_exc())
        
        return 1

if __name__ == "__main__":
    sys.exit(main())
