"""
Utility functions for converting between different representations of partial orders.
"""

import numpy as np
from typing import List, Optional
from .basic_utils import BasicUtils

class ConversionUtils:
    """
    Utility class for converting sequences and orders to different representations.
    """

    @staticmethod
    def seq2dag(seq: List[int], n: int) -> np.ndarray:
        """
        Converts a sequence to a directed acyclic graph (DAG) represented as an adjacency matrix.

        Parameters:
        - seq: A sequence (list) of integers representing a total order.
        - n: Total number of elements.

        Returns:
        - adj_matrix: An n x n numpy array representing the adjacency matrix of the DAG.
        """
        adj_matrix = np.zeros((n, n), dtype=int)
        for i in range(len(seq)):
            u = seq[i] - 1  # Convert to 0-based index
            for j in range(i + 1, len(seq)):
                v = seq[j] - 1  # Convert to 0-based index
                adj_matrix[u, v] = 1
        return adj_matrix

    @staticmethod
    def order2partial(v: List[List[int]], n: Optional[int] = None) -> np.ndarray:
        """
        Computes the intersection of the transitive closures of a list of total orders.

        Parameters:
        - v: List of sequences, where each sequence is a list of integers representing a total order.
        - n: Total number of elements (optional).

        Returns:
        - result_matrix: An n x n numpy array representing the adjacency matrix of the partial order.
        """
        if n is None:
            n = max(max(seq) for seq in v)
        z = np.zeros((n, n), dtype=int)
        for seq in v:
            dag_matrix = ConversionUtils.seq2dag(seq, n)
            closure_matrix = BasicUtils.transitive_closure(dag_matrix)
            z += closure_matrix
        result_matrix = (z == len(v)).astype(int)
        return result_matrix 