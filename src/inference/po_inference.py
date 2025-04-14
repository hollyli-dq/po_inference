import os
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any
from src.utils.basic_utils import BasicUtils
from src.utils.statistical_utils import StatisticalUtils
from src.utils.generation_utils import GenerationUtils
from src.visualization.po_plot import POPlot
from src.mcmc.mcmc_simulation import mcmc_partial_order


def get_project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Config file not found at: {config_path}")
        raise
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {str(e)}")
        raise


def load_data(data_path: str) -> Dict[str, Any]:
    """Load data from JSON file."""
    try:
        with open(data_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Data file not found at: {data_path}")
        raise
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file: {str(e)}")
        raise

def run_inference(data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Run MCMC inference on the data."""
    try:
        # Extract data
        total_orders = data.get('total_orders', [])
        subsets = data.get('subsets', [])
        parameters = data.get('parameters', {})

        # Get MCMC parameters from config
        num_iterations = config["mcmc"]["num_iterations"]
        K=config["generation"]["K"]
        mcmc_pt= [
            config["mcmc"]["update_probabilities"]["rho"],
            config["mcmc"]["update_probabilities"]["noise"],
            config["mcmc"]["update_probabilities"]["U"]
]
        dr = config["rho"]["dr"]
        noise_option = config["noise"]["noise_option"]
        sigma_mallow = config["noise"]["sigma_mallow"]

        # Get prior parameters
        rho_prior = config["prior"]["rho_prior"]
        noise_beta_prior = config["prior"]["noise_beta_prior"]
        mallow_ua = config["prior"]["mallow_ua"]

        # Get covariate effects if available
        beta_true = parameters.get('beta_true', np.zeros(config['covariates']['p']))
        X = parameters.get('X', np.zeros((len(total_orders), config['covariates']['p'])))
        alpha = X @ beta_true

        # Run MCMC simulation
        mcmc_results = mcmc_partial_order(
            total_orders,
            subsets,
            num_iterations,
            K,
            dr,
            sigma_mallow,
            noise_option,
            mcmc_pt,
            rho_prior,
            noise_beta_prior,
            mallow_ua,
            alpha
        )

        # Compute the final inferred partial order 'h' from the MCMC trace.
        if 'h_trace' in mcmc_results:
            burn_in = int(config['visualization']['burn_in'])
            h_trace = np.array(mcmc_results['h_trace'])

            if burn_in >= len(h_trace):
                print(f"Warning: burn_in ({burn_in}) is larger than trace length ({len(h_trace)}). Using last 1000 iterations.")
                burn_in = max(0, len(h_trace) - 1000)

            post_burn_in_trace = h_trace[burn_in:]
            if len(post_burn_in_trace) == 0:
                raise ValueError("No valid data after burn-in period")

            # Compute the mean over the trace (ignoring NaNs)
            h_final = np.nanmean(post_burn_in_trace, axis=0)
            if np.any(np.isnan(h_final)):
                print("Warning: NaN values detected in h_final. Using last valid state.")
                h_final = h_trace[-1]

            # Apply a threshold (e.g. 0.5) and perform transitive reduction
            threshold = 0.5
            h_final_inferred = BasicUtils.transitive_reduction(h_final >= threshold).astype(int)

            # Add the final inferred partial order to the results
            mcmc_results['h'] = h_final_inferred
        else:
            print("Warning: h_trace not found in MCMC results; setting 'h' to an empty array.")
            mcmc_results['h'] = np.array([])

        # Also add final states for Z, beta, rho, and prob_noise.
        if "Z_trace" in mcmc_results and mcmc_results["Z_trace"]:
            mcmc_results["Z"] = mcmc_results["Z_trace"][-1]
        else:
            mcmc_results["Z"] = np.array([])

        mcmc_results['beta'] = beta_true

        if "rho_trace" in mcmc_results and mcmc_results["rho_trace"]:
            mcmc_results["rho"] = mcmc_results["rho_trace"][-1]
        else:
            mcmc_results["rho"] = 0.0

        if "prob_noise_trace" in mcmc_results and mcmc_results["prob_noise_trace"]:
            mcmc_results["prob_noise"] = mcmc_results["prob_noise_trace"][-1]
        else:
            mcmc_results["prob_noise"] = 0.0

        # Package trace information into a 'trace' key for saving.
        trace_keys = ['Z_trace', 'h_trace', 'rho_trace', 'prob_noise_trace', 'mallow_theta_trace']
        trace_info = {key: mcmc_results.get(key, []) for key in trace_keys}
        mcmc_results['trace'] = trace_info

        return mcmc_results

    except Exception as e:
        print(f"Error in run_inference: {str(e)}")
        raise


def save_results(results: Dict[str, Any], output_dir: str, data_name: str):
    """Save inference results to JSON file."""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        def convert_to_serializable(obj):
            """Convert numpy arrays and other non-serializable objects to serializable format."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_to_serializable(item) for item in obj)
            else:
                return obj
        
        # Convert all numpy arrays to lists for JSON serialization
        results_dict = convert_to_serializable(results)
        
        # Save results to JSON file
        results_path = os.path.join(output_dir, f"{data_name}_results.json")
        with open(results_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        print(f"\nResults saved to {results_path}")
        
        # Save partial order matrix separately as numpy array
        if 'h' in results:
            h_path = os.path.join(output_dir, f"{data_name}_partial_order.npy")
            np.save(h_path, results['h'])
            print(f"Partial order matrix saved to {h_path}")
            
    except Exception as e:
        print(f"Error in save_results: {str(e)}")
        raise

def generate_plots(results: Dict[str, Any], data: Dict[str, Any], config: Dict[str, Any], output_dir: str, data_name: str):
    """Generate and save plots."""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Get true parameters if they exist
        true_param = {}
        if 'parameters' in data:
            if 'rho_true' in data['parameters']:
                true_param['rho_true'] = data['parameters']['rho_true']
            if 'prob_noise_true' in data['parameters']:
                true_param['prob_noise_true'] = data['parameters']['prob_noise_true']

        # Get burn-in from config
        burn_in = config['visualization']['burn_in']

        # Plot MCMC inferred variables
        plot_path = os.path.join(output_dir, f"{data_name}_mcmc_plots.pdf")
        POPlot.plot_mcmc_inferred_variables(
            results['trace'],
            true_param,
            config,
            burn_in=burn_in,
            output_filename=plot_path
        )
        print(f"Plots saved to {plot_path}")

        # Plot partial orders
        items = data.get('items', {}).get('names', [f"Item {i}" for i in range(results['h'].shape[0])])

        # Plot inferred partial order
        inferred_plot_path = os.path.join(output_dir, f"{data_name}_inferred_po.pdf")
        plt.figure(figsize=(10, 8))
        POPlot.visualize_partial_order(
            final_h=results['h'],
            Ma_list=items,
            title='Inferred Partial Order'
        )
        plt.savefig(inferred_plot_path)
        plt.close()
        print(f"Inferred partial order plot saved to {inferred_plot_path}")

        # If true partial order exists in data, convert it to a numpy array before processing.
        if 'true_partial_order' in data:
            true_po = data['true_partial_order']
            if isinstance(true_po, list):
                true_po = np.array(true_po)
            true_plot_path = os.path.join(output_dir, f"{data_name}_true_po.pdf")
            plt.figure(figsize=(10, 8))
            POPlot.visualize_partial_order(
                final_h=BasicUtils.transitive_reduction(true_po),
                Ma_list=items,
                title='True Partial Order'
            )
            plt.savefig(true_plot_path)
            plt.close()
            print(f"True partial order plot saved to {true_plot_path}")

            # Compare relationships
            missing_relationships = BasicUtils.compute_missing_relationships(
                true_po, 
                results['h'], 
                items
            )
            redundant_relationships = BasicUtils.compute_redundant_relationships(
                true_po, 
                results['h'], 
                items
            )

            # Print relationship comparisons
            if missing_relationships:
                print("\nMissing Relationships (edges present in true PO but absent in inferred PO):")
                for i, j in missing_relationships:
                    print(f"{i} < {j}")
            else:
                print("\nNo missing relationships. The inferred partial order matches the true partial order.")

            if redundant_relationships:
                print("\nRedundant Relationships (edges present in inferred PO but absent in true PO):")
                for i, j in redundant_relationships:
                    print(f"{i} < {j}")
            else:
                print("\nNo redundant relationships. The inferred partial order is a subset of the true partial order.")

    except Exception as e:
        print(f"Error in generate_plots: {str(e)}")
        raise


def main():
    """Main function to run the inference pipeline."""
    try:
        project_root = get_project_root()
        print(f"Project root: {project_root}")

        # Load configuration using absolute path
        config_path = os.path.join(project_root, 'config', 'mcmc_config.yaml')
        print(f"Config path: {config_path}")
        config = load_config(config_path)

        # Load data using absolute path
        data_path = os.path.join(project_root, config['data']['path'])
        data = load_data(data_path)

        # Create output directory if it doesn't exist
        output_dir = os.path.join(project_root, config['data']['output_dir'])
        os.makedirs(output_dir, exist_ok=True)

        # Run inference
        results = run_inference(data, config)

        # Save results
        data_name = os.path.splitext(os.path.basename(data_path))[0]
        save_results(results, output_dir, data_name)

        # Generate plots
        generate_plots(results, data, config, output_dir, data_name)

    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise


if __name__ == "__main__":
    main()
