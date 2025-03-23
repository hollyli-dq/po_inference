#!/usr/bin/env python3
"""
Command-line interface for Bayesian Partial Order Inference.
"""

import os
import sys
import argparse
import json
from typing import Dict, Any

from src.data.data_generator import generate_data
from src.inference.po_inference import run_inference, save_results, generate_plots
from src.utils.basic_utils import load_config


def get_project_root() -> str:
    """Get the absolute path to the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Bayesian Partial Order Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Config file paths
    parser.add_argument(
        "--data-config", 
        type=str, 
        default="config/data_generator_config.yaml",
        help="Path to data generation config file"
    )
    parser.add_argument(
        "--mcmc-config", 
        type=str, 
        default="config/mcmc_config.yaml",
        help="Path to MCMC config file"
    )
    
    # Operation modes
    parser.add_argument(
        "--generate-data", 
        action="store_true",
        help="Generate synthetic data only"
    )
    parser.add_argument(
        "--inference-only", 
        action="store_true",
        help="Run inference only with existing data"
    )
    
    # Output settings
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default=None,
        help="Output directory for results (overrides config setting)"
    )
    
    # MCMC settings override
    parser.add_argument(
        "--iterations", 
        type=int, 
        default=None,
        help="Number of MCMC iterations (overrides config setting)"
    )
    parser.add_argument(
        "--burn-in", 
        type=int, 
        default=None,
        help="Burn-in period for MCMC (overrides config setting)"
    )
    parser.add_argument(
        "--latent-dim", 
        type=int, 
        default=None,
        help="Latent dimension K (overrides config setting)"
    )
    
    return parser.parse_args()


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
    """Main function to run the command-line interface."""
    # Parse command-line arguments
    args = parse_arguments()
    
    try:
        # Get project root directory
        project_root = get_project_root()
        
        # Load configurations
        data_gen_config_path = os.path.join(project_root, args.data_config)
        mcmc_config_path = os.path.join(project_root, args.mcmc_config)
        
        if not os.path.exists(data_gen_config_path):
            raise FileNotFoundError(f"Data generator config not found at: {data_gen_config_path}")
        if not os.path.exists(mcmc_config_path):
            raise FileNotFoundError(f"MCMC config not found at: {mcmc_config_path}")
            
        data_gen_config = load_config(data_gen_config_path)
        mcmc_config = load_config(mcmc_config_path)
        
        # Override config settings with command-line arguments
        if args.output_dir:
            mcmc_config['data']['output_dir'] = args.output_dir
        
        if args.iterations:
            mcmc_config['mcmc']['num_iterations'] = args.iterations
            
        if args.burn_in:
            mcmc_config['visualization']['burn_in'] = args.burn_in
            
        if args.latent_dim:
            mcmc_config['mcmc']['K'] = args.latent_dim
        
        # Set up output directory
        output_dir = os.path.join(project_root, mcmc_config['data']['output_dir'])
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")
        
        # Generate data if specified
        if args.generate_data or (not args.inference_only and mcmc_config['data'].get('generate_data', True)):
            print("\nGenerating synthetic data...")
            data = generate_data(data_gen_config)
            
            # Save generated data
            data_name = "synthetic_data"
            data_path = save_generated_data(data, output_dir, data_name)
            
            # Update mcmc config with the path to generated data
            mcmc_config['data']['path'] = os.path.relpath(
                os.path.join(output_dir, f"{data_name}.json"),
                project_root
            )
            
            if args.generate_data:
                print("Data generation completed successfully.")
                return
        
        # Run inference if specified
        if not args.generate_data:
            # Load existing data
            data_path = os.path.join(project_root, mcmc_config['data']['path'])
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data file not found at: {data_path}")
                
            print(f"Loading data from: {data_path}")
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
            
            print("\nInference completed successfully!")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main()) 