#!/usr/bin/env python3
"""
Main script for running the partial order inference pipeline.
This script handles both data generation and inference.
"""

import os
import logging
import json
import yaml
import numpy as np
from typing import Dict, Any
from pathlib import Path

from src.utils.basic_utils import BasicUtils
from src.data.data_generator import generate_data
from src.mcmc.mcmc_simulation import mcmc_partial_order
from src.visualization.po_plot import POPlot
from src.inference.po_inference import run_inference, save_results, generate_plots, load_config

def setup_directories(config: Dict[str, Any]) -> Dict[str, Path]:
    """Set up project directories and return paths."""
    base_dir = Path(config['data']['output_dir'])
    dirs = {
        'data': base_dir / 'data',
        'figures': base_dir / 'figures',
        'results': base_dir / 'results',
        'partial_orders': base_dir / 'partial_orders'
    }
    
    # Create directories
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs

def get_project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def save_generated_data(data: Dict[str, Any], output_dir: str, data_name: str):
    """Save generated data to JSON file."""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save data to JSON file
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
        project_root = get_project_root()
        print(f"Project root: {project_root}")
        
        # Load configuration using absolute path

        
        # Load configurations
        mcmc_config = load_config(os.path.join(project_root, 'config', 'mcmc_config.yaml'))
        
        # Set up output directory
        output_dir = os.path.join(project_root, mcmc_config['data']['output_dir'])
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate data if specified
        if mcmc_config['data'].get('generate_data', True):
            print("\nGenerating synthetic data...")
            data = generate_data(mcmc_config)
            # Save generated data
            data_name = "synthetic_data"
            data_path = save_generated_data(data, output_dir, data_name)
            
            # Update mcmc config with the path to generated data
            mcmc_config['data']['path'] = os.path.join(mcmc_config['data']['output_dir'], f"{data_name}.json")
        else:
            # Load existing data
            data_path = os.path.join(project_root, mcmc_config['data']['path'])
            with open(data_path, 'r') as f:
                data = json.load(f)
            data_name = os.path.splitext(os.path.basename(data_path))[0]
        
        # Run inference
        print("\nRunning MCMC inference...")
        results = run_inference(data, mcmc_config)
        
        # Save results
        save_results(results, output_dir, data_name)
        
        # Generate plots
        generate_plots(results, data, mcmc_config, output_dir, data_name)
        
        print("\nPipeline completed successfully!")
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 