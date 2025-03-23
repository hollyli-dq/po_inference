#!/usr/bin/env python3
"""
Convenience script for running the Bayesian Partial Order Inference CLI.
This script adds the project root to the Python path and calls the CLI module.
"""

import os
import sys
import argparse
from pathlib import Path

# Add the project root to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)


def print_example_usage():
    """Print example usage commands."""
    print("\nExample Usage:")
    print("--------------")
    print("# Run complete pipeline (data generation + inference + plots)")
    print("python scripts/run_po_inference.py")
    print("\n# Generate data only")
    print("python scripts/run_po_inference.py --generate-data-only")
    print("\n# Run inference on existing data")
    print("python scripts/run_po_inference.py --inference-only --data-file data/sample_data.json")
    print("\n# Generate plots only from existing results")
    print("python scripts/run_po_inference.py --plot-only --data-file data/sample_data.json")
    print("\n# Override MCMC parameters")
    print("python scripts/run_po_inference.py --iterations 5000 --dimensions 3 --burn-in 1000")
    print("\n# Use custom configuration files")
    print("python scripts/run_po_inference.py --data-config config/custom_data_config.yaml --mcmc-config config/custom_mcmc_config.yaml")
    print("\n# Specify output directory")
    print("python scripts/run_po_inference.py --output-dir experiments/run1")
    print("\n# Generate verbose output for debugging")
    print("python scripts/run_po_inference.py --verbose --debug")


if __name__ == "__main__":
    # Check if the user wants to see example usage
    if len(sys.argv) > 1 and sys.argv[1] in ["--examples", "--help-examples"]:
        print_example_usage()
        sys.exit(0)
        
    # Import CLI module
    from src.cli import main
    
    # Run the CLI
    main() 