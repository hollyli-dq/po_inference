"""
Utility functions for handling k-dimensional partial orders and their properties.
"""

import numpy as np
import itertools
from typing import List, Dict, Tuple, Any
from .basic_utils import BasicUtils
from .statistical_utils import StatisticalUtils

class KDimensionUtils:
    """
    Utility class for handling k-dimensional partial orders and their properties.
    """
    
    @staticmethod
    def find_critical_pairs(items: List[Any], h: np.ndarray) -> List[Tuple[Any, Any]]:
        """
        Find critical pairs in a partial order.
        
        Parameters:
        -----------
        items : List[Any]
            List of items in the partial order
        h : np.ndarray
            Partial order matrix
            
        Returns:
        --------
        List[Tuple[Any, Any]]
            List of critical pairs (pairs of incomparable elements)
        """
        n = len(items)
        critical_pairs = []
        
        for i in range(n):
            for j in range(i + 1, n):
                if h[i, j] == 0 and h[j, i] == 0:
                    critical_pairs.append((items[i], items[j]))
                    
        return critical_pairs

    @staticmethod
    def find_min_realizer(h: np.ndarray, items: List[Any]) -> Tuple[List[List[Any]], int]:
        """
        Find the minimal realizer of a partial order.
        
        Parameters:
        -----------
        h : np.ndarray
            Partial order matrix
        items : List[Any]
            List of items in the partial order
        
        Returns:
        --------
        Tuple[List[List[Any]], int]
            The minimal realizer and its size
        """
        all_exts = BasicUtils.generate_all_linear_extensions(h, items)
        best_size = float('inf')
        best_subset = None

        # For each combination of linear extensions (starting from size 1)
        for size in range(1, len(all_exts) + 1):
            for combo in itertools.combinations(all_exts, size):
                inter_matrix = KDimensionUtils.realizer_to_partial_order_matrix(combo, items)
                # Convert inter_dict back to a matrix for comparison:
                if np.array_equal(BasicUtils.transitive_closure(inter_matrix),
                                BasicUtils.transitive_closure(h)):
                    best_size = size
                    best_subset = combo
                    break
            if best_subset is not None:
                break
    
        return best_subset, best_size
    
    @staticmethod
    def realizer_to_partial_order_matrix(realizer: List[List[Any]], items: List[Any] = None) -> np.ndarray:
        """
        Generate a partial order matrix from a collection of linear extensions.
        
        Parameters:
        -----------
        realizer : List[List[Any]]
            A collection of linear extensions (each is a list/tuple of items)
        items : List[Any], optional
            List of items. If not provided, items are extracted from the realizer
            
        Returns:
        --------
        np.ndarray
            An n x n numpy array such that H[i,j] = 1 if every linear extension 
            has items[i] before items[j]
        """
        if items is None:
            items = sorted(set(item for ext in realizer for item in ext))
        item_index = {item: idx for idx, item in enumerate(items)}
        n = len(items)
        H = np.zeros((n, n), dtype=int)
        
        # For each pair (i,j) check if every extension orders items[i] before items[j]
        for i in range(n):
            for j in range(n):
                if i != j:
                    H[i, j] = 1 if all(ext.index(items[i]) < ext.index(items[j]) for ext in realizer) else 0

        return H

    @staticmethod
    def generate_crown_poset(k: int) -> Tuple[List[str], Dict[str, set], np.ndarray]:
        """
        Generate a crown poset with n=2k elements.
        
        Parameters:
        -----------
        k : int
            Number of a-items (and b-items), so total elements = 2k
            
        Returns:
        --------
        Tuple[List[str], Dict[str, set], np.ndarray]
            items: List of items (strings) in the order [a1,...,aK, b1,...,bK]
            adj: A dictionary where each key is an item and the value is a set of items it points to
            adj_matrix: A numpy array representing the adjacency matrix
        """
        # Create items lists
        A = [f"a{i}" for i in range(1, k+1)]
        B = [f"b{i}" for i in range(1, k+1)]
        items = A + B
        
        # Build adjacency dictionary
        adj = {x: set() for x in items}
        for i, a_item in enumerate(A, start=1):
            for j, b_item in enumerate(B, start=1):
                if i != j:
                    adj[a_item].add(b_item)
        
        # Build a mapping from item to its index
        item_to_index = {item: idx for idx, item in enumerate(items)}
        
        # Create adjacency matrix
        n = len(items)
        adj_matrix = np.zeros((n, n), dtype=int)
        for x, neighbors in adj.items():
            for y in neighbors:
                i = item_to_index[x]
                j = item_to_index[y]
                adj_matrix[i, j] = 1
                
        return items, adj, adj_matrix