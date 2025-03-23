#!/usr/bin/env python3
"""
Command-line interface for the Bayesian Partial Order Inference package.
This module provides a robust CLI using argparse to run the pipeline with
different parameters without editing config files.
"""

import os
import sys
import argparse
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

from src.data.data_generator import generate_data
from src.inference.po_inference import run_inference, save_results, generate_plots
from src.utils.basic_utils import load_config


def get_project_root() -> Path:
    """Get the absolute path to the project root directory."""
    return Path(__file__).parent.parent.absolute()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Bayesian Partial Order Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # General options
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose output")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode (more detailed logging)")
    
    # Configuration files
    config_group = parser.add_argument_group("Configuration")
    config_group.add_argument("--data-config", type=str, 
                              default="config/data_generator_config.yaml",
                              help="Path to data generation configuration file")
    config_group.add_argument("--mcmc-config", type=str,
                              default="config/mcmc_config.yaml",
                              help="Path to MCMC configuration file")
    config_group.add_argument("--output-dir", type=str, 
                              help="Output directory for results (overrides config)")
    
    # Pipeline control
    pipeline_group = parser.add_argument_group("Pipeline Control")
    pipeline_group.add_argument("--generate-data-only", action="store_true",
                               help="Only generate data, skip inference")
    pipeline_group.add_argument("--inference-only", action="store_true",
                               help="Only run inference on existing data, skip generation")
    pipeline_group.add_argument("--plot-only", action="store_true",
                               help="Only generate plots from existing results")
    
    # MCMC parameters (override config)
    mcmc_group = parser.add_argument_group("MCMC Parameters")
    mcmc_group.add_argument("--iterations", type=int, 
                           help="Number of MCMC iterations (overrides config)")
    mcmc_group.add_argument("--burn-in", type=int,
                           help="Number of burn-in iterations (overrides config)")
    mcmc_group.add_argument("--dimensions", type=int,
                           help="Number of latent dimensions (K) (overrides config)")
    mcmc_group.add_argument("--rho", type=float,
                           help="Initial rho value (overrides config)")
    
    # Data generation parameters (override config)
    data_group = parser.add_argument_group("Data Parameters")
    data_group.add_argument("--num-items", type=int,
                           help="Number of items (overrides config)")
    data_group.add_argument("--num-observations", type=int,
                           help="Number of observations (overrides config)")
    data_group.add_argument("--noise-option", type=str, 
                           choices=["queue_jump", "mallows_noise"],
                           help="Noise model to use (overrides config)")
    
    # Input/output files
    io_group = parser.add_argument_group("Input/Output")
    io_group.add_argument("--data-file", type=str,
                         help="Path to input data file (for inference only)")
    io_group.add_argument("--results-prefix", type=str, default="result",
                         help="Prefix for output files")
    
    return parser.parse_args()


def update_config_with_args(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Update configuration with command-line arguments."""
    # Make a deep copy of the config to avoid modifying the original
    updated_config = json.loads(json.dumps(config))
    
    # Update MCMC config
    if args.iterations:
        updated_config["mcmc"]["num_iterations"] = args.iterations
    
    if args.burn_in and "visualization" in updated_config:
        updated_config["visualization"]["burn_in"] = args.burn_in
    
    if args.dimensions:
        updated_config["mcmc"]["K"] = args.dimensions
    
    if args.rho and "rho" in updated_config:
        updated_config["rho"]["initial"] = args.rho
    
    # Update data generation config
    if args.num_items and "generation" in updated_config:
        updated_config["generation"]["n"] = args.num_items
    
    if args.num_observations and "generation" in updated_config:
        updated_config["generation"]["N"] = args.num_observations
    
    if args.noise_option and "noise" in updated_config:
        updated_config["noise"]["noise_option"] = args.noise_option
    
    # Update output directory
    if args.output_dir and "data" in updated_config:
        updated_config["data"]["output_dir"] = args.output_dir
    
    # Update data file path
    if args.data_file and "data" in updated_config:
        updated_config["data"]["path"] = args.data_file
    
    return updated_config


def setup_logging(verbose: bool, debug: bool) -> None:
    """Set up logging configuration."""
    import logging
    
    log_level = logging.INFO
    if debug:
        log_level = logging.DEBUG
    elif not verbose:
        log_level = logging.WARNING
    
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def main() -> None:
    """Main entry point for the CLI."""
    # Parse command-line arguments
    args = parse_args()
    
    # Set up logging
    setup_logging(args.verbose, args.debug)
    
    try:
        # Get project root directory
        project_root = get_project_root()
        
        # Resolve config paths (allow both absolute and relative paths)
        data_config_path = Path(args.data_config)
        if not data_config_path.is_absolute():
            data_config_path = project_root / data_config_path
            
        mcmc_config_path = Path(args.mcmc_config)
        if not mcmc_config_path.is_absolute():
            mcmc_config_path = project_root / mcmc_config_path
        
        # Load configurations
        data_gen_config = load_config(str(data_config_path))
        mcmc_config = load_config(str(mcmc_config_path))
        
        # Update configurations with command-line arguments
        data_gen_config = update_config_with_args(data_gen_config, args)
        mcmc_config = update_config_with_args(mcmc_config, args)
        
        # Set up output directory
        output_dir = args.output_dir or mcmc_config["data"]["output_dir"]
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(project_root, output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine data file and name
        if args.data_file:
            data_path = args.data_file
            if not os.path.isabs(data_path):
                data_path = os.path.join(project_root, data_path)
            data_name = os.path.splitext(os.path.basename(data_path))[0]
        else:
            data_name = args.results_prefix or "synthetic_data"
            data_path = os.path.join(output_dir, f"{data_name}.json")
        
        # Generate data if needed
        if not args.inference_only and not args.plot_only:
            if not os.path.exists(data_path) or not args.generate_data_only:
                print(f"\nGenerating synthetic data...")
                data = generate_data(data_gen_config)
                
                # Save generated data
                os.makedirs(os.path.dirname(data_path), exist_ok=True)
                with open(data_path, 'w') as f:
                    json.dump(data, f, indent=2)
                print(f"\nGenerated data saved to {data_path}")
                
                if args.generate_data_only:
                    print("\nData generation completed successfully!")
                    return
        
        # Load data for inference
        if not args.generate_data_only:
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data file not found at: {data_path}")
                
            with open(data_path, 'r') as f:
                data = json.load(f)
            
            # Run inference if not plot-only
            if not args.plot_only:
                print("\nRunning MCMC inference...")
                results = run_inference(data, mcmc_config)
                
                # Save results
                results_dir = os.path.join(output_dir, "results")
                os.makedirs(results_dir, exist_ok=True)
                save_results(results, output_dir, data_name)
            else:
                # Load existing results for plot-only mode
                results_path = os.path.join(output_dir, "results", f"{data_name}_results.json")
                if not os.path.exists(results_path):
                    raise FileNotFoundError(f"Results file not found at: {results_path}")
                    
                with open(results_path, 'r') as f:
                    results = json.load(f)
            
            # Generate plots
            print("\nGenerating plots...")
            generate_plots(results, data, mcmc_config, output_dir, data_name)
        
        print("\nPipeline completed successfully!")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 