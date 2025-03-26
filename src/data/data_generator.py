"""
Data generator module for partial order inference.
"""

import os
import json
import yaml
import numpy as np
from scipy.stats import beta
from typing import Dict, List, Any
from src.utils.basic_utils import BasicUtils
from src.utils.statistical_utils import StatisticalUtils
from src.utils.generation_utils import GenerationUtils
from src.visualization.po_plot import POPlot
import matplotlib.pyplot as plt

def get_project_root() -> str:
    """Get the absolute path to the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file (relative to project root or absolute)
        
    Returns:
        Dictionary containing configuration
    """
    if not os.path.isabs(config_path):
        # If relative path, make it relative to project root
        config_path = os.path.join(get_project_root(), config_path)
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def generate_data(config: Dict[str, Any]) -> Dict[str, Any]:
    """Generate synthetic data for partial order inference."""
    try:
        # 1. Set up parameters
        n = config['generation']['n']  # Number of nodes
        N = config['generation']['N']  # Number of total orders
        K = config['generation']['K']  # Number of dimensions
        p = config.get('covariates', {}).get('p', 2)  # Number of covariates
        
        # 2. Generate latent positions and correlation parameter
        rho_prior = config['prior']['rho_prior']
        rho_true = beta.rvs(1, rho_prior)
        print(f"True correlation parameter (rho): {rho_true:.4f}")
        
        # 3. Generate latent positions
        U = GenerationUtils.generate_U(n, K, rho_true)
        print("\nLatent positions (U):")
        print(U)
        
        # 4. Generate covariates and effects
        X = np.random.randn(p, n)  # Generate random covariates
        beta_true = np.random.randn(p)  # Generate random effects
        alpha = X.T @ beta_true  # Compute covariate effects
        print("\nCovariate effects (alpha):")
        print(alpha)
        
        # 5. Transform latent positions
        eta = StatisticalUtils.transform_U_to_eta(U, alpha)
        print("\nTransformed latent positions (eta):")
        print(eta)
        
        # 6. Generate partial order from transformed latent positions
        h = BasicUtils.generate_partial_order(eta)
        h_true = BasicUtils.transitive_reduction(h.copy())
        print("\nPartial Order (adjacency matrix):")
        print(h_true)
        
        # 7. Generate subsets for sampling total orders
        subsets = GenerationUtils.generate_subsets(N, n)
        
        # 8. Generate total orders
        h_tc = BasicUtils.transitive_closure(h)
        total_orders = []
        
        for subset in subsets:
            total_order = GenerationUtils.sample_total_order(h_tc, subset)
            total_orders.append(total_order)
        
        # 9. Prepare output data in the format expected by the inference module
        output_data = {
            'total_orders': total_orders,  # List of total orders
            'subsets': subsets,  # List of subsets
            'parameters': {
                'n': n,
                'N': N,
                'K': K,
                'rho_true': float(rho_true)
            },
            'true_partial_order': h_true.tolist(),  # True partial order matrix
            'beta_true': beta_true.tolist(),  # True covariate effects
            'X': X.tolist()  # Covariate matrix
        }
        
        return output_data
        
    except Exception as e:
        print(f"Error in generate_data: {str(e)}")
        raise

def main():
    try:
        # Load configuration
        config = load_config('config/mcmc_config.yaml')
        
        # Get project root and create output directory
        project_root = get_project_root()
        output_dir = os.path.join(project_root, config['output']['dir'])
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate data
        data = generate_data(config)
        
        # Save data
        output_path = os.path.join(output_dir, config['output']['filename'])
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\nData saved to {output_path}")
        
        # Convert partial order back to numpy array
        h_true = np.array(data['true_partial_order'], dtype=int)
        
        # Create figure for initial partial order
        plt.figure(figsize=(10, 8))
        POPlot.visualize_partial_order(
            h_true,
            Ma_list=data['items']['names'],
            title='Initial Partial Order'
        )
        save_path = os.path.join(output_dir, 'partial_order_init.png')
        plt.savefig(save_path)
        plt.close()
        print(f"\nInitial partial order plot saved to {save_path}")
        
        # Print statistics
        print("\nSampling Statistics:")
        total_orders = data['total_orders']
        unique_orders = set(map(tuple, total_orders))
        print(f"Number of unique total orders: {len(unique_orders)}")
        
        # Count occurrences of each ordering
        order_counts = {}
        for order in total_orders:
            order_tuple = tuple(order)
            order_counts[order_tuple] = order_counts.get(order_tuple, 0) + 1
        
        # Find most common ordering
        most_common = max(order_counts.items(), key=lambda x: x[1])
        print(f"Most common ordering: {most_common[0]} (appears {most_common[1]} times)")
        
    except Exception as e:
        print(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main() 