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
from main import main as run_pipeline


def get_project_root() -> Path:
    """Get the absolute path to the project root directory."""
    return Path(__file__).parent.parent.absolute()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Run partial order inference pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration files
    parser.add_argument(
        '--data-gen-config',
        default='config/data_generator_config.yaml',
        help='Path to data generator configuration file'
    )
    parser.add_argument(
        '--mcmc-config',
        default='config/mcmc_config.yaml',
        help='Path to MCMC configuration file'
    )
    
    # Data generation parameters
    parser.add_argument(
        '--n-items',
        type=int,
        help='Number of items to generate (overrides config)'
    )
    parser.add_argument(
        '--n-observations',
        type=int,
        help='Number of observations to generate (overrides config)'
    )
    parser.add_argument(
        '--dimension',
        type=int,
        help='Latent dimension K (overrides config)'
    )
    
    # MCMC parameters
    parser.add_argument(
        '--iterations',
        type=int,
        help='Number of MCMC iterations (overrides config)'
    )
    parser.add_argument(
        '--burn-in',
        type=int,
        help='Number of burn-in iterations (overrides config)'
    )
    parser.add_argument(
        '--thinning',
        type=int,
        help='Thinning interval (overrides config)'
    )
    
    # Output options
    parser.add_argument(
        '--output-dir',
        default='output',
        help='Output directory for results'
    )
    parser.add_argument(
        '--data-name',
        default='synthetic_data',
        help='Name for the generated/loaded data'
    )
    
    # Logging options
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    # Other options
    parser.add_argument(
        '--generate-data',
        action='store_true',
        help='Force data generation even if data exists'
    )
    parser.add_argument(
        '--no-generate-data',
        action='store_true',
        help='Skip data generation and use existing data'
    )
    
    return parser.parse_args()


def update_config(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Update configuration with command line arguments."""
    if args.n_items is not None:
        config['generation']['n'] = args.n_items
    if args.n_observations is not None:
        config['generation']['N'] = args.n_observations
    if args.dimension is not None:
        config['generation']['K'] = args.dimension
    
    if args.iterations is not None:
        config['mcmc']['num_iterations'] = args.iterations
    if args.burn_in is not None:
        config['visualization']['burn_in'] = args.burn_in
    if args.thinning is not None:
        config['mcmc']['thinning'] = args.thinning
    
    if args.output_dir is not None:
        config['data']['output_dir'] = args.output_dir
    
    if args.generate_data:
        config['data']['generate_data'] = True
    if args.no_generate_data:
        config['data']['generate_data'] = False
    
    return config


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


def validate_data_file(data_file: str, project_root: Path) -> None:
    """Validate that the data file exists and is readable."""
    data_path = project_root / data_file
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at: {data_path}")
    try:
        with open(data_path, 'r') as f:
            json.load(f)
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in data file: {data_path}")


def setup_directories(output_dir: str):
    """Create necessary output directories."""
    dirs = {
        'figures': os.path.join(output_dir, 'figures'),
        'mcmc_traces': os.path.join(output_dir, 'figures', 'mcmc_traces'),
        'partial_orders': os.path.join(output_dir, 'figures', 'partial_orders'),
        'results': os.path.join(output_dir, 'results'),
        'mcmc_samples': os.path.join(output_dir, 'results', 'mcmc_samples'),
        'summary_stats': os.path.join(output_dir, 'results', 'summary_stats'),
        'logs': os.path.join(output_dir, 'logs')
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs


def main() -> None:
    """Main CLI function."""
    try:
        # Parse arguments
        args = parse_args()
        
        # Get project root directory
        project_root = get_project_root()
        
        # Set up logging
        setup_logging(args.verbose, args.debug)
        
        # Load configurations
        data_gen_config = load_config(args.data_gen_config)
        mcmc_config = load_config(args.mcmc_config)
        
        # Update configurations with command line arguments
        data_gen_config = update_config(data_gen_config, args)
        mcmc_config = update_config(mcmc_config, args)
        
        # Run pipeline
        run_pipeline(data_gen_config, mcmc_config, args.data_name)
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main() 