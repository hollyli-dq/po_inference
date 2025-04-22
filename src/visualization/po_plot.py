"""
Visualization module for partial order inference.
"""

import os
import math
from collections import Counter
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd  
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy.stats import beta, expon, kstest, probplot
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

class POPlot:
    """Class for visualizing partial orders and MCMC results."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the visualization module.
        
        Args:
            config: Configuration dictionary containing visualization settings
        """
        self.config = config
        self.output_dir = Path(config['data']['output_dir']) / 'figures'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_results(self, results: Dict[str, Any]):
        """
        Plot MCMC results and partial orders.
        
        Args:
            results: Dictionary containing MCMC results
        """
        # Plot MCMC traces
        self.plot_traces(results)
        
        # Plot partial orders
        self.plot_partial_orders(results)
        
    def plot_traces(self, results: Dict[str, Any]):
        """Plot MCMC parameter traces."""
        trace_dir = self.output_dir / 'mcmc_traces'
        trace_dir.mkdir(exist_ok=True)
        
        for param, trace in results['traces'].items():
            plt.figure(figsize=(10, 6))
            plt.plot(trace)
            plt.title(f'MCMC Trace: {param}')
            plt.xlabel('Iteration')
            plt.ylabel('Value')
            plt.savefig(trace_dir / f'{param}_trace.png')
            plt.close()
            
    def plot_partial_orders(self, results: Dict[str, Any]):
        """Plot partial orders using networkx."""
        po_dir = self.output_dir / 'partial_orders'
        po_dir.mkdir(exist_ok=True)
        
        # Plot true partial order
        self._plot_single_po(
            results['true_partial_order'],
            po_dir / 'true_partial_order.png',
            'True Partial Order'
        )
        
        # Plot inferred partial order
        self._plot_single_po(
            results['inferred_partial_order'],
            po_dir / 'inferred_partial_order.png',
            'Inferred Partial Order'
        )
        
    def _plot_single_po(self, matrix: np.ndarray, output_path: Path, title: str):
        """
        Plot a single partial order matrix.
        
        Args:
            matrix: Partial order matrix
            output_path: Path to save the plot
            title: Plot title
        """
        G = nx.DiGraph()
        n = matrix.shape[0]
        
        # Add nodes
        for i in range(n):
            G.add_node(i)
            
        # Add edges
        for i in range(n):
            for j in range(n):
                if matrix[i, j] == 1:
                    G.add_edge(i, j)
                    
        # Plot
        plt.figure(figsize=(10, 10))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightblue',
                node_size=500, font_size=16, font_weight='bold')
        plt.title(title)
        plt.savefig(output_path)
        plt.close()

    @staticmethod
    def plot_Z_trace(Z_trace: List[np.ndarray], index_to_item: Dict[int, str], burn_in: int = 100) -> None:
        """Plot the trace of multidimensional latent variables Z."""
        Z_array = np.array(Z_trace)
        iterations = Z_array.shape[0]
        
        if Z_array.ndim != 3:
            raise ValueError("Z_trace should be a list of Z matrices with shape (n, K).")

        if len(Z_array) > burn_in:
            Z_array = Z_array[burn_in:]
            iterations = Z_array.shape[0]
            print(f"Excluding {burn_in} burn-in iterations")
        else:
            burn_in = 0
            print("No burn-in period applied")
        
        _, n_items, K = Z_array.shape
        
        fig, axes = plt.subplots(K, 1, figsize=(12, 4 * K), sharex=True)
        if K == 1:
            axes = [axes]

        for k in range(K):
            ax = axes[k]
            for idx in range(n_items):
                ax.plot(range(iterations), Z_array[:, idx, k], label=f"{index_to_item[idx]}")
            ax.set_ylabel(f'Latent Variable Z (Dimension {k + 1})')
            ax.legend(loc='best', fontsize='small')
            ax.grid(True)
        
        axes[-1].set_xlabel('Iteration')
        plt.suptitle(f'Trace Plot of Multidimensional Latent Variables Z (Post Burn-in)', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    @staticmethod
    def plot_acceptance_rates(accepted_iterations: List[int], acceptance_rates: List[float]) -> None:
        """Plot the acceptance rates over iterations."""
        plt.figure(figsize=(10, 6))
        plt.plot(accepted_iterations, acceptance_rates, marker='o', linestyle='-', color='blue')
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Acceptance Rate', fontsize=12)
        plt.title('Acceptance Rate Over Time', fontsize=14)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_top_partial_orders(top_percentages: List[Tuple[np.ndarray, int, float]], 
                              top_n: int = 5, 
                              item_labels: Optional[List[str]] = None) -> None:
        """Plot the top N partial orders as heatmaps."""
        n_cols = 3
        n_rows = (top_n + n_cols - 1) // n_cols
        
        plt.figure(figsize=(5 * n_cols, 4 * n_rows))
        
        for idx, (order, count, percentage) in enumerate(top_percentages[:top_n], 1):
            plt.subplot(n_rows, n_cols, idx)
            sns.heatmap(order, annot=True, fmt="d", cmap="Blues", cbar=False, 
                       linewidths=.5, linecolor='gray',
                        xticklabels=item_labels, yticklabels=item_labels)
            plt.title(f"Top {idx}: {percentage:.2f}%\nCount: {count}")
            plt.xlabel("Items")
            plt.ylabel("Items")
        
        total_plots = n_rows * n_cols
        if top_n < total_plots:
            for empty_idx in range(top_n + 1, total_plots + 1):
                plt.subplot(n_rows, n_cols, empty_idx)
                plt.axis('off')
        
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_log_likelihood(log_likelihood_data: Union[Dict[str, Any], List[float]], 
                          burn_in: int = 100,
                            title: str = 'Log Likelihood Over MCMC Iterations') -> None:
        """Plot the total log likelihood over MCMC iterations."""
        if isinstance(log_likelihood_data, dict):
            log_likelihood_currents = log_likelihood_data.get('log_likelihood_currents', [])
        else:
            log_likelihood_currents = log_likelihood_data
        
        if len(log_likelihood_currents) > burn_in:
            burned_ll = log_likelihood_currents[burn_in:]
            iterations = np.arange(burn_in + 1, len(log_likelihood_currents) + 1)
            print(f"Excluding {burn_in} burn-in iterations")
        else:
            burned_ll = log_likelihood_currents
            iterations = np.arange(1, len(log_likelihood_currents) + 1)
            burn_in = 0
            print("No burn-in period applied")
        
        sns.set(style="whitegrid")
        plt.figure(figsize=(14, 7))
        
        sns.lineplot(x=iterations, y=burned_ll, label='Current State', color='blue')
        
        plt.title(f'{title} (Post Burn-in)', fontsize=16)
        plt.xlabel('Iteration', fontsize=14)
        plt.ylabel('Total Log Likelihood', fontsize=14)
        
        plt.legend(title='State')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def visualize_partial_order(
        final_h: np.ndarray,
        Ma_list: list,
        title: str = None
    ) -> None:
        """
        Visualizes the partial order for a single assessor using NetworkX and PyGraphviz for layout.
        
        Parameters:
        - final_h (np.ndarray): An n x n numpy array representing the adjacency matrix of the partial order.
        - Ma_list (list): A list of item labels corresponding to the nodes in the partial order.
        - assessor (int, optional): The assessor ID. If provided and title is not specified, it will be used in the default title.
        - title (str, optional): The title of the plot. If not provided, a default title is generated.
        """
        import networkx as nx
        import matplotlib.pyplot as plt
        
        # Set default title if not provided.
        if title is None:
            if assessor is not None:
                title = f"Partial Order Graph for Assessor {assessor}"
            else:
                title = "Partial Order Graph"
        
        # Create a directed graph from the adjacency matrix.
        G = nx.DiGraph(final_h)
        
        # Build node labels from Ma_list.
        # We assume final_h is an n x n matrix with n equal to len(Ma_list).
        labels = {i: str(Ma_list[i]) for i in range(len(Ma_list))}
        
        # Try to use PyGraphviz for layout; if not available, fall back to a spring layout.
        try:
            A = nx.nx_agraph.to_agraph(G)
            # Set node labels using the labels dictionary.
            for node in A.nodes():
                try:
                    node_int = int(node)
                except ValueError:
                    node_int = node
                node.attr['label'] = labels.get(node_int, str(node))
            A.layout('dot')
            A.draw('graph.png')
            img = plt.imread('graph.png')
            plt.imshow(img)
            plt.axis('off')
            plt.title(title)
            plt.show()
        except (ImportError, nx.NetworkXException):
            pos = nx.spring_layout(G)
            nx.draw(G, pos, labels=labels, with_labels=True, arrows=True)
            plt.title(title)
            plt.show()

    @staticmethod
    def plot_mcmc_inferred_variables(mcmc_results: Dict[str, Any],
                                   true_param: Dict[str, Any],
                                   config: Dict[str, Any],
                                   burn_in: int = 100,
                                   output_filename: str = "mcmc_inferred_result.pdf",
                                   output_filepath: str = ".",
                                   assessors: Optional[List[int]] = None,
                                   M_a_dict: Optional[Dict[int, Any]] = None) -> None:
        """Create plots showing MCMC traces and histograms."""
        sns.set_style("whitegrid")
        
        # Extract main MCMC traces and apply burn-in
        traces = {}
        true_values = {}
        
        # Define variables to plot with their properties
        var_configs = {
            'rho': {'color': '#1f77b4', 'prior': 'beta', 'prior_params': {'a': 1.0, 'b': config["prior"].get("rho_prior", 1.0)}, 'truncated': True},
            'tau': {'color': 'brown', 'prior': 'uniform'},
            'K': {'color': 'darkcyan', 'prior': 'poisson', 'prior_params': {'lambda': config["prior"].get("K_prior", 1.0)}}
        }
        
        # Add noise parameters based on noise model type
        noise_model = config.get("noise", {}).get("noise_option", "").lower()
        if noise_model == "queue_jump":
            var_configs['prob_noise'] = {'color': 'orange', 'prior': 'beta', 'prior_params': {'a': 1.0, 'b': config["prior"].get("noise_beta_prior", 1.0)}}
        elif noise_model == "mallows_noise":
            var_configs['mallow_theta'] = {'color': 'purple', 'prior': None}
        
        # Extract traces and true values
        for var_name, var_config in var_configs.items():
            trace_key = f"{var_name}_trace"
            if trace_key in mcmc_results and mcmc_results[trace_key] is not None:
                traces[var_name] = np.array(mcmc_results[trace_key])[burn_in:]
                true_values[var_name] = true_param.get(f"{var_name}_true", None)
        
        # Create subplots
        n_vars = len(traces)
        fig, axes = plt.subplots(n_vars, 2, figsize=(12, 4 * n_vars))
        if n_vars == 1:
            axes = axes.reshape(1, -1)
        
        # Plot each variable
        for idx, (var_name, trace) in enumerate(traces.items()):
            var_config = var_configs[var_name]
            true_val = true_values[var_name]
            
            # Trace plot
            ax_trace = axes[idx, 0]
            iterations = np.arange(burn_in + 1, burn_in + len(trace) + 1)
            if isinstance(trace[0], np.ndarray):
                mean_trace = np.mean(trace, axis=(1, 2))
                ax_trace.plot(iterations, mean_trace, color=var_config['color'], lw=1.2, alpha=0.8)
                ax_trace.set_ylabel(f'Mean {var_name}')
            else:
                ax_trace.plot(iterations, trace, color=var_config['color'], lw=1.2, alpha=0.8)
                ax_trace.set_ylabel(var_name)
            
            ax_trace.set_title(f"Trace: {var_name}")
            ax_trace.set_xlabel("Iteration")
            ax_trace.grid(True, alpha=0.3)
            
            # Density plot
            ax_hist = axes[idx, 1]
            if var_name == 'rho' and var_config.get('truncated', False):
                tol = 1e-4
                trunc_point = 1 - tol
                bin_edges = np.linspace(0, trunc_point, 101)
                bin_edges[-1] += 1e-6
                sns.histplot(trace, kde=False, ax=ax_hist, color=var_config['color'],
                           bins=bin_edges, edgecolor='black', linewidth=0)
                ax_hist.set_xlim(0.5, trunc_point)
                
                x_vals = np.linspace(0.5, trunc_point, 1000)
                norm_const = beta.cdf(trunc_point, **var_config['prior_params'])
                norm_const = max(norm_const, 1e-15)
                prior_pdf = beta.pdf(x_vals, **var_config['prior_params']) / norm_const
                ax_hist.plot(x_vals, prior_pdf, 'k-', lw=2, label='Theoretical PDF')
            else:
                if isinstance(trace[0], np.ndarray):
                    mean_values = np.mean(trace, axis=(1, 2))
                    sns.histplot(mean_values, kde=False, ax=ax_hist, color=var_config['color'])
                else:
                    sns.histplot(trace, kde=False, ax=ax_hist, color=var_config['color'])
            
            ax_hist.set_title(f"Density: {var_name}")
            ax_hist.set_xlabel(var_name)
            ax_hist.set_ylabel("Density")
            
            # Add prior distribution if specified
            if var_config['prior'] == 'beta' and var_name != 'rho':
                x_vals = np.linspace(0, 1, 1000)
                prior_pdf = beta.pdf(x_vals, **var_config['prior_params'])
                scale_factor = len(trace) * (ax_hist.get_xlim()[1] / 30.0)
                ax_hist.plot(x_vals, prior_pdf * scale_factor, 'k--', label='Prior')
            elif var_config['prior'] == 'uniform':
                x_vals = np.linspace(0, max(trace), 1000)
                prior_pdf = np.ones_like(x_vals)
                scale_factor = len(trace) * (ax_hist.get_xlim()[1] / 30.0)
                ax_hist.plot(x_vals, prior_pdf * scale_factor, 'k--', label='Prior')
            elif var_config['prior'] == 'poisson':
                k_range = np.arange(1, max(trace) + 3)
                lambda_param = var_config['prior_params']['lambda']
                norm_const = 1.0 - math.exp(-lambda_param)
                pmf_vals = np.array([math.exp(-lambda_param + k * math.log(lambda_param) - math.lgamma(k + 1)) / norm_const 
                                   for k in k_range])
                scale = len(trace)
                ax_hist.plot(k_range, pmf_vals * scale, 'k--', label='Prior')
            
            # Add true value if available
            if true_val is not None:
                if isinstance(true_val, np.ndarray):
                    true_val = np.mean(true_val)
                ax_hist.axvline(true_val, color='red', linestyle='--', label='True')
            
            # Add sample mean
            sample_mean = np.mean(trace) if not isinstance(trace[0], np.ndarray) else np.mean(np.mean(trace, axis=(1, 2)))
            ax_hist.axvline(sample_mean, color='green', linestyle='--', label='Sample Mean')
            
            ax_hist.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_filepath, output_filename))
        print(f"[INFO] Saved MCMC parameter plots to '{output_filename}'")
        plt.show()

    @staticmethod
    def create_mcmc_trace_plot(
        rho_trace: List[float],
        prob_noise_trace: List[float],
        mallow_theta_trace: List[float],
        burn_in: int,
        true_param: Dict[str, float] = None
    ) -> plt.Figure:
        """Create MCMC trace plots for parameters."""
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        
        # Plot rho trace
        axes[0].plot(rho_trace, label='MCMC')
        if true_param and 'rho_true' in true_param:
            axes[0].axhline(y=true_param['rho_true'], color='r', linestyle='--', label='True')
        axes[0].axvline(x=burn_in, color='k', linestyle='--', label='Burn-in')
        axes[0].set_title('Rho Trace')
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Value')
        axes[0].legend()
        
        # Plot prob_noise trace
        axes[1].plot(prob_noise_trace, label='MCMC')
        if true_param and 'prob_noise_true' in true_param:
            axes[1].axhline(y=true_param['prob_noise_true'], color='r', linestyle='--', label='True')
        axes[1].axvline(x=burn_in, color='k', linestyle='--', label='Burn-in')
        axes[1].set_title('Noise Probability Trace')
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Value')
        axes[1].legend()
        
        # Plot mallow_theta trace
        axes[2].plot(mallow_theta_trace, label='MCMC')
        axes[2].axvline(x=burn_in, color='k', linestyle='--', label='Burn-in')
        axes[2].set_title('Mallows Theta Trace')
        axes[2].set_xlabel('Iteration')
        axes[2].set_ylabel('Value')
        axes[2].legend()
        
        plt.tight_layout()
        return fig

    @staticmethod
    def create_partial_order_plot(
        h: np.ndarray,
        index_to_item: Dict[int, str],
        title: str = "Partial Order"
    ) -> plt.Figure:
        """Create a visualization of a partial order."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Add nodes
        for idx, item in index_to_item.items():
            G.add_node(item)
        
        # Add edges based on partial order matrix
        n = h.shape[0]
        for i in range(n):
            for j in range(n):
                if h[i, j] == 1:
                    G.add_edge(index_to_item[i], index_to_item[j])
        
        # Use spring layout for node positioning
        pos = nx.spring_layout(G)
        
        # Draw the graph
        nx.draw(G, pos, ax=ax, with_labels=True, node_color='lightblue',
                node_size=2000, font_size=10, font_weight='bold',
                arrows=True, edge_color='gray')
        
        ax.set_title(title)
        plt.tight_layout()
        return fig
    
    @staticmethod

    def plot_beta_parameters(mcmc_results: Dict[str, Any],
                            true_param: Dict[str, Any],
                            config: Dict[str, Any],
                            burn_in: int = 100,
                            output_filepath: str = ".") -> None:
        """
        Plot each component of beta in a separate figure. Each figure has two subplots:
        one for the trace and one for the density. Font sizes for beta labels are set very small.
        
        Assumes that mcmc_results["beta_trace"] is a 2D array with shape (n_samples, p),
        and that true_param["beta_true"] is a NumPy array of length p.
        """
        sns.set_style("whitegrid")
        # Use a small font for beta plots.
        beta_font = {
            'title': 8,
            'label': 7,
            'legend': 6,
            'ticks': 6
        }
        
        # Extract beta trace and true beta
        beta_trace = np.array(mcmc_results.get("beta_trace", []))
        if beta_trace.size == 0:
            print("No beta trace available.")
            return
        beta_trace = beta_trace[burn_in:]
        true_beta = true_param.get("beta_true", None)
        
        # Determine dimensions
        n_samples, p_dim = beta_trace.shape
        
        # Create one separate figure per beta coefficient.
        for d in range(p_dim):
            fig, (ax_trace, ax_hist) = plt.subplots(1, 2, figsize=(8, 3))
            iterations = np.arange(burn_in + 1, burn_in + 1 + n_samples)
            
            # TRACE subplot for beta_d
            ax_trace.plot(iterations, beta_trace[:, d], color=plt.cm.tab10(d), lw=1.2, alpha=0.8)
            ax_trace.set_title(f"β{d} Trace", fontsize=beta_font['title'])
            ax_trace.set_xlabel("Iteration", fontsize=beta_font['label'])
            ax_trace.set_ylabel("β value", fontsize=beta_font['label'])
            ax_trace.tick_params(axis='both', labelsize=beta_font['ticks'])
            ax_trace.grid(True, alpha=0.3)
            
            # DENSITY subplot for beta_d
            sns.histplot(beta_trace[:, d], kde=True, ax=ax_hist, color=plt.cm.tab10(d), alpha=0.5)
            ax_hist.set_title(f"β{d} Density", fontsize=beta_font['title'])
            ax_hist.set_xlabel("β value", fontsize=beta_font['label'])
            ax_hist.set_ylabel("Count", fontsize=beta_font['label'])
            ax_hist.tick_params(axis='both', labelsize=beta_font['ticks'])
            if true_beta is not None and d < len(true_beta):
                ax_hist.axvline(true_beta[d], color=plt.cm.tab10(d), linestyle='--', lw=1,
                                label="True β")
            sample_mean = np.mean(beta_trace[:, d])
            ax_hist.axvline(sample_mean, color='green', linestyle='--', lw=1, label="Sample Mean")
            ax_hist.legend(fontsize=beta_font['legend'])
            
            plt.tight_layout()
            outname = os.path.join(output_filepath, f"beta_{d}_plot.pdf")
            plt.savefig(outname, dpi=300, bbox_inches='tight')
            print(f"[INFO] Saved beta coefficient {d} plot to '{outname}'")
            plt.show()