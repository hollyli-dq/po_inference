#!/usr/bin/env python3
"""
Main script for running the partial order inference pipeline.
This script handles both data generation and inference.
"""

import os
import json
import yaml
import numpy as np
import argparse
from typing import Dict, Any
from src.data.data_generator import generate_data
from src.inference.po_inference import run_inference, save_results, generate_plots
from src.utils.basic_utils import load_config, BasicUtils

def get_project_root() -> str:
    """Get the absolute path to the project root directory."""
    # Get the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # The script is in the project root, so return script_dir directly
    return script_dir

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run partial order inference pipeline')
    
    # MCMC parameters
    parser.add_argument('--iterations', type=int, help='Number of MCMC iterations')
    parser.add_argument('--burn-in', type=int, help='Number of burn-in iterations')
    parser.add_argument('--thinning', type=int, help='Thinning interval for MCMC samples')
    parser.add_argument('--dimension', type=int, help='Dimension of latent space (K)')
    
    # Model parameters
    parser.add_argument('--noise-model', choices=['queue_jump', 'mallows_noise'], 
                       help='Noise model to use')
    
    # Data generation parameters
    parser.add_argument('--n-items', type=int, help='Number of items to generate')
    parser.add_argument('--n-observations', type=int, help='Number of observations to generate')
    
    # Prior parameters
    parser.add_argument('--rho-prior', type=float, help='Prior parameter for correlation')
    parser.add_argument('--noise-beta-prior', type=float, help='Beta prior parameter for noise')
    parser.add_argument('--K-prior', type=float, help='Prior parameter for dimension K')
    
    # Output parameters
    parser.add_argument('--output-dir', type=str, help='Output directory for results')
    
    return parser.parse_args()

def update_config_with_args(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Update configuration with command line arguments."""
    if args.iterations is not None:
        if 'mcmc' not in config:
            config['mcmc'] = {}
        config['mcmc']['num_iterations'] = args.iterations
    if args.burn_in is not None:
        if 'mcmc' not in config:
            config['mcmc'] = {}
        config['mcmc']['burn_in'] = args.burn_in
    if args.thinning is not None:
        if 'mcmc' not in config:
            config['mcmc'] = {}
        config['mcmc']['thinning'] = args.thinning
    if args.dimension is not None:
        if 'mcmc' not in config:
            config['mcmc'] = {}
        config['mcmc']['K'] = args.dimension
    if args.noise_model is not None:
        if 'noise' not in config:
            config['noise'] = {}
        config['noise']['noise_option'] = args.noise_model
    if args.output_dir is not None:
        if 'data' not in config:
            config['data'] = {}
        config['data']['output_dir'] = args.output_dir
    return config

def save_generated_data(data: Dict[str, Any], output_dir: str, data_name: str):
    """Save generated data to JSON file."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        data_path = os.path.join(output_dir, f"{data_name}.json")
        with open(data_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\nGenerated data saved to {data_path}")
        return data_path
    except Exception as e:
        print(f"Error in save_generated_data: {str(e)}")
        raise

def main():
    """Main function to run the data generation and inference pipeline."""
    try:
        # Parse command line arguments
        args = parse_args()
        
        # Get project root directory
        project_root = get_project_root()
        print(f"Project root directory: {project_root}")
        
        # Load configurations
        config_dir = os.path.join(project_root, 'config')
        print(f"Looking for config files in: {config_dir}")
        
        mcmc_config_path = os.path.join(config_dir, 'mcmc_config.yaml')
        
        if not os.path.exists(mcmc_config_path):
            raise FileNotFoundError(f"MCMC config not found at: {mcmc_config_path}")
            
        # Load configuration
        mcmc_config = load_config(mcmc_config_path)
        
        # Update configuration with command line arguments
        mcmc_config = update_config_with_args(mcmc_config, args)
        
        # Set up output directory
        output_dir = os.path.join(project_root, mcmc_config['data']['output_dir'])
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate or load data
        if mcmc_config['data']['generate_data']:
            print("\nGenerating synthetic data...")
            data = generate_data(mcmc_config)
            data_path = save_generated_data(data, output_dir, mcmc_config['data']['data_name'])
        else:
            print("\nLoading existing data...")
            data_path = os.path.join(project_root, mcmc_config['data']['path'])
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data file not found at: {data_path}")
            with open(data_path, 'r') as f:
                data = json.load(f)
        
        # Run inference
        print("\nRunning MCMC inference...")
        results = run_inference(data, mcmc_config)
        
        # Save results
        print("\nSaving results...")
        save_results(results, output_dir, mcmc_config['data']['data_name'])
        
        # Generate plots
        print("\nGenerating plots...")
        generate_plots(results, data, mcmc_config, output_dir, mcmc_config['data']['data_name'])
        
        print("\nAnalysis completed successfully!")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

if __name__ == "__main__":
    main() 