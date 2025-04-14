"""
Basic utility functions for partial order operations.
"""

import os
import yaml
import numpy as np
import networkx as nx
import math
from typing import List, Dict, Set, Any, Optional
import itertools
from functools import lru_cache

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing the configuration
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class BasicUtils:
    """
    Utility class for basic operations on partial orders.
    """    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            Dictionary containing configuration parameters
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found at: {config_path}")
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file: {str(e)}")
            
    @staticmethod
    def apply_transitive_reduction_hpo(h_U: dict) -> None:
        """
        For each key in h_U, if the value is a NumPy array, replace it with its transitive closure.
        If the value is a dictionary (e.g. assessor-level partial orders by task), then apply the
        operation to each matrix in that dictionary.
        
        This function modifies h_U in place.
        """
        for key, value in h_U.items():
            if isinstance(value, dict):
                # If value is a dictionary, iterate over its keys
                for subkey, subval in value.items():
                    if isinstance(subval, np.ndarray):
                        value[subkey] = BasicUtils.transitive_reduction(subval)
            elif isinstance(value, np.ndarray):
                h_U[key] = BasicUtils.transitive_reduction(value)
    @staticmethod
    def generate_all_linear_extensions(h: np.ndarray, items: Optional[List[Any]] = None) -> List[List[Any]]:
        """
        Generate all linear extensions (i.e. valid total orders) of a partial order
        represented by the adjacency matrix h. Here, h is an n x n matrix where h[i, j] == 1
        means that index i must precede index j. The items are by default the indices [0,1,...,n-1],
        but if a list 'items' is provided, it will be used to map indices to actual items.

        Parameters:
            h: n x n numpy array representing the partial order.
            items: Optional list of items corresponding to the indices of h.
                If None, items are assumed to be [0, 1, ..., n-1].

        Returns:
            A list of linear extensions, each represented as a list of items (or indices if items is None).
        """
        n = h.shape[0]
        if items is None:
            items = list(range(n))        
        def _recursive_extensions(h_sub: np.ndarray, remaining: List[int]) -> List[List[int]]:
            # Base case: if no elements remain, return an empty extension.
            if not remaining:
                return [[]]
            
            m = len(remaining)
            # Compute in-degrees for the current submatrix.
            in_degree = [0] * m
            for i in range(m):
                for j in range(m):
                    if h_sub[i, j]:
                        in_degree[j] += 1
            
            # Minimal elements are those with in-degree zero.
            minimal_indices = [i for i, d in enumerate(in_degree) if d == 0]
            
            extensions = []
            for idx in minimal_indices:
                # 'current' is the actual index from the original set.
                current = remaining[idx]
                # Remove the minimal element from the remaining list.
                new_remaining = remaining[:idx] + remaining[idx+1:]
                # Remove the corresponding row and column from the matrix.
                new_h = np.delete(np.delete(h_sub, idx, axis=0), idx, axis=1)
                # Recursively generate extensions for the reduced poset.
                for ext in _recursive_extensions(new_h, new_remaining):
                    extensions.append([current] + ext)
            return extensions

        # Start the recursion with all indices [0, 1, ..., n-1].
        index_extensions = _recursive_extensions(h, list(range(n)))
        # Map the indices to actual items if provided.
        extensions = [[items[i] for i in extension] for extension in index_extensions]
        return extensions

    @staticmethod
    def build_Sigma_rho(K: int, rho_val: float) -> np.ndarray:
        """Build correlation matrix with given rho value."""
        mat = np.full((K, K), rho_val, dtype=float)
        np.fill_diagonal(mat, 1.0)
        return mat

    @staticmethod
    def generate_partial_order(eta: np.ndarray) -> np.ndarray:
        """
        Generate a partial order matrix from transformed latent positions.
        
        Parameters:
        -----------
        eta : np.ndarray
            Matrix of transformed latent positions (n × K)
            
        Returns:
        --------
        np.ndarray
            Binary matrix representing the partial order (n × n)
        """
        n = eta.shape[0]
        h = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(n):
                if i != j:
                    if np.all(eta[i] >= eta[j]) and np.any(eta[i] > eta[j]):
                        h[i, j] = 1
        return h

    @staticmethod
    def is_total_order(adj_matrix: np.ndarray) -> bool:
        """Check if adjacency matrix represents a total order using vectorized operations."""
        n = adj_matrix.shape[0]
        # Compute transitive closure
        closure = BasicUtils.transitive_closure(adj_matrix)
        # Check if all off-diagonal elements are 1
        return np.all(closure + closure.T == 1) and np.all(np.diag(closure) == 0)

    @staticmethod
    def restrict_partial_order(h: np.ndarray, subset: List[int]) -> np.ndarray:
        """
        Restrict the partial order matrix `h` to the given `subset` using numpy indexing.
        """
        return h[np.ix_(subset, subset)]

    @staticmethod
    def transitive_reduction(h: np.ndarray) -> np.ndarray:
        """
        Compute the transitive reduction of a partial order matrix.

        Parameters:
        -----------
        h : np.ndarray
            Binary matrix representing the partial order

        Returns:
        --------
        np.ndarray
            Transitive reduction of the input matrix
        """
        n = h.shape[0]
        tr = h.copy()

        # Floyd-Warshall algorithm for transitive closure
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if tr[i, k] and tr[k, j]:
                        tr[i, j] = 0  # Remove direct edge if there's an indirect path

        return tr

    @staticmethod
    def transitive_closure(adj_matrix: np.ndarray) -> np.ndarray:
        """
        Computes the transitive closure of a relation represented by an adjacency matrix.

        Parameters:
        - adj_matrix: An n x n numpy array representing the adjacency matrix of the relation.

        Returns:
        - closure: An n x n numpy array representing the adjacency matrix of the transitive closure.
        """
        n = adj_matrix.shape[0]
        closure = adj_matrix.copy()
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    closure[i, j] = closure[i, j] or (closure[i, k] and closure[k, j])
        return closure

    @staticmethod
    @lru_cache(maxsize=1000)
    def _nle_cached(h_tuple: tuple) -> int:
        """
        Cached version of nle that works with tuples.
        """
        # Convert tuple back to numpy array
        n = int(np.sqrt(len(h_tuple)))
        h = np.array(h_tuple).reshape(n, n)
        
        if n <= 1:
            return 1

        # Find minimal elements (those with no incoming edges)
        in_degree = np.sum(h, axis=0)
        minimal_elements = np.where(in_degree == 0)[0]
        
        if len(minimal_elements) == 0:
            return 0  # Not a valid partial order (has cycles)

        total = 0
        for min_elem in minimal_elements:
            mask = np.ones(n, dtype=bool)
            mask[min_elem] = False
            sub_h = h[mask][:, mask]
            # Convert submatrix to tuple for caching
            sub_h_tuple = tuple(sub_h.flatten())
            total += BasicUtils._nle_cached(sub_h_tuple)

        return total

    @staticmethod
    def nle(h: np.ndarray) -> int:
        """
        Count the number of linear extensions of a partial order with caching.
        Converts numpy array to tuple for caching purposes.
        
        Parameters:
        -----------
        h : np.ndarray
            Binary matrix representing the partial order
            
        Returns:
        --------
        int
            Number of linear extensions
        """
        # Convert numpy array to tuple for caching
        h_tuple = tuple(h.flatten())
        return BasicUtils._nle_cached(h_tuple)

    @staticmethod
    def find_tops(tr: np.ndarray) -> List[int]:
        """
        Identify all top elements using vectorized operations.
        """
        return np.where(np.sum(tr, axis=0) == 0)[0].tolist()

    @staticmethod
    def num_extensions_with_first(h: np.ndarray, first_idx: int) -> int:
        """
        Count the number of linear extensions where a specific element appears first.

        Parameters:
        -----------
        h : np.ndarray
            Binary matrix representing the partial order
        first_idx : int
            Index of the element that should appear first

        Returns:
        --------
        int
            Number of linear extensions with the specified element first
        """
        n = h.shape[0]
        
        # Check if the element can be first (no incoming edges)
        if np.any(h[:, first_idx] == 1):
            return 0

        # Remove the element and count extensions of the remaining elements
        mask = np.ones(n, dtype=bool)
        mask[first_idx] = False
        sub_h = h[mask][:, mask]
        
        return BasicUtils.nle(sub_h)

    @staticmethod
    def is_consistent(h: np.ndarray, observed_orders: List[List[int]]) -> bool:
        """
        Check if all observed orders are consistent with the partial order using vectorized operations.
        """
        # Create a directed graph from the partial order matrix h
        G_PO = nx.DiGraph(h)
        # Compute the transitive closure to capture all implied precedence relations
        tc_PO = BasicUtils.transitive_closure(h)

        # Convert observed orders to numpy arrays for vectorized operations
        for order in observed_orders:
            positions = np.array([order.index(i) if i in order else float('inf') for i in range(h.shape[0])])
            # Check all edges in the transitive closure
            conflicts = tc_PO & (positions[:, np.newaxis] > positions[np.newaxis, :])
            if np.any(conflicts):
                return False

        return True

    @staticmethod
    def is_valid_partial_order(h: np.ndarray) -> bool:
        """
        Check if a matrix represents a valid partial order.

        Parameters:
        -----------
        h : np.ndarray
            Binary matrix to check

        Returns:
        --------
        bool
            True if the matrix represents a valid partial order
        """
        n = h.shape[0]
        
        # Check antisymmetry
        for i in range(n):
            for j in range(i + 1, n):
                if h[i, j] and h[j, i]:
                    return False

        # Check transitivity
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if h[i, j] and h[j, k]:
                        if not h[i, k]:
                            return False

        return True

    @staticmethod
    def compute_missing_relationships(h_true: np.ndarray, h_final: np.ndarray, index_to_item: Dict[int, int]) -> List[tuple]:
        """
        Compute the missing relationships between the true partial order and the inferred partial order.
        Both input matrices are converted to their transitive closure before comparison.
        
        Parameters:
        -----------
        h_true : np.ndarray
            Adjacency matrix of the true partial order
        h_final : np.ndarray
            Adjacency matrix of the inferred partial order
        index_to_item : Dict[int, int]
            Mapping from index to item
            
        Returns:
        --------
        List[tuple]
            List of tuples representing the missing relationships (i, j)
        """
        # Convert both matrices to their transitive closure
        h_true_closed = BasicUtils.transitive_closure(h_true)
        h_final_closed = BasicUtils.transitive_closure(h_final)
        
        missing = []
        n = h_true.shape[0]
        for i in range(n):
            for j in range(n):
                if h_true_closed[i, j] == 1 and h_final_closed[i, j] == 0:
                    missing.append((index_to_item[i], index_to_item[j]))
        return missing

    @staticmethod
    def compute_redundant_relationships(h_true: np.ndarray, h_final: np.ndarray, index_to_item: Dict[int, int]) -> List[tuple]:
        """
        Compute the redundant relationships in the inferred partial order not present in the true partial order.
        Both input matrices are converted to their transitive closure before comparison.
        
        Parameters:
        -----------
        h_true : np.ndarray
            Adjacency matrix of the true partial order
        h_final : np.ndarray
            Adjacency matrix of the inferred partial order
        index_to_item : Dict[int, int]
            Mapping from index to item
            
        Returns:
        --------
        List[tuple]
            List of tuples representing the redundant relationships (i, j)
        """
        # Convert both matrices to their transitive closure
        h_true_closed = BasicUtils.transitive_closure(h_true)
        h_final_closed = BasicUtils.transitive_closure(h_final)
        
        redundant = []
        n = h_true.shape[0]
        for i in range(n):
            for j in range(n):
                if h_true_closed[i, j] == 0 and h_final_closed[i, j] == 1:
                    redundant.append((index_to_item[i], index_to_item[j]))
        return redundant 