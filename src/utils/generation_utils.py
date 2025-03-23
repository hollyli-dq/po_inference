"""
Utility functions for generating various components of partial orders.
"""

import numpy as np
import networkx as nx
import random
from typing import List, Dict, Optional
from scipy.stats import multivariate_normal
import itertools
from .basic_utils import BasicUtils

class GenerationUtils:
    """
    Utility class for generating latent positions, partial orders, random partial orders, 
    linear extensions, total orders, and subsets.
    """
    @staticmethod
    def generate_choice_sets_for_assessors(
        M_a_dict: Dict[int, List[int]],
        min_tasks: int = 1,
        min_size: int = 2
    ) -> Dict[int, List[List[int]]]:
        """
        Generate a dictionary of choice sets O_{a,i} for each assessor.
        
        Each assessor a is assigned a random number of tasks (choice sets) between
        min_tasks and max_tasks. For each task, a random subset of items (of size at least
        min_size and at most the total number of items in M_a) is selected from the assessor's M_a.
        
        Parameters:
            M_a_dict : Dict[int, List[int]]
                Dictionary mapping assessor IDs to their overall list of item IDs.
            min_tasks : int, optional
                Minimum number of tasks per assessor (default is 1).
            min_size : int, optional
                Minimum number of items in a choice set (default is 2).
        
        Returns:
            Dict[int, List[List[int]]]: Dictionary where each key is an assessor ID and each value is a list
                                        of choice sets (each choice set is a list of item IDs).
        """
        O_a_i_dict = {}
        for assessor, items in M_a_dict.items():
            num_items = len(items)
            max_tasks = 10*num_items
            # Determine the number of tasks for this assessor.
            num_tasks = random.randint(min_tasks, max_tasks)
            tasks = []
            for _ in range(num_tasks):
                # Choose a task size: at least min_size, at most all available items.
                task_size = random.randint(min_size, num_items)
                task = sorted(random.sample(items, task_size))
                tasks.append(task)
            O_a_i_dict[assessor] = tasks
        return O_a_i_dict

    @staticmethod
    def generate_latent_positions(n: int, K: int, rho: float) -> np.ndarray:
        """
        Generates latent positions Z for n items in K dimensions with correlation rho.

        Parameters:
        - n: Number of items.
        - K: Number of dimensions.
        - rho: Correlation coefficient between dimensions.

        Returns:
        - Z: An n x K numpy array of latent positions.
        """
        Sigma = BasicUtils.build_Sigma_rho(K, rho)
        mu = np.zeros(K)
        rv = multivariate_normal(mean=mu, cov=Sigma)
        Z = rv.rvs(size=n)
        if K == 1:
            Z = Z.reshape(n, 1)
        return Z

    @staticmethod
    def generate_random_PO(n: int) -> nx.DiGraph:
        """
        Generates a random partial order (directed acyclic graph) with `n` nodes.
        Ensures there are no cycles in the generated graph.

        Parameters:
        - n: Number of nodes in the partial order.

        Returns:
        - h: A NetworkX DiGraph representing the partial order.
        """
        h = nx.DiGraph()
        h.add_nodes_from(range(n))
        possible_edges = list(itertools.combinations(range(n), 2))
        random.shuffle(possible_edges)
        for u, v in possible_edges:
            if random.choice([True, False]):
                h.add_edge(u, v)
                if not nx.is_directed_acyclic_graph(h):
                    h.remove_edge(u, v)
        return h

    @staticmethod
    def generate_U(n: int, K: int, rho_val: float) -> np.ndarray:
        """
        Generate a latent variable matrix U of size n x K from a multivariate normal distribution
        with zero mean and a covariance matrix based on the given correlation rho_val.

        Parameters:
        - n: Number of observations.
        - K: Number of features.
        - rho_val: Correlation value for constructing the covariance matrix.

        Returns:
        - U: An n x K numpy array of latent positions.
        """
        K = int(K)
        cov = BasicUtils.build_Sigma_rho(K, rho_val)
        mean = np.zeros(K)
        U = np.random.multivariate_normal(mean, cov, size=n)
        return U

    @staticmethod
    def unifLE(tc: np.ndarray, elements: List[int], le: Optional[List[int]] = None) -> List[int]:
        """
        Sample a linear extension uniformly at random from the given partial order matrix `tc`.

        Parameters:
        - tc: Transitive closure matrix representing the partial order (numpy 2D array).
        - elements: List of elements corresponding to the current `tc` matrix.
        - le: List to build the linear extension (default: None).

        Returns:
        - le: A linear extension (list of elements in the original subset).
        """
        if le is None:
            le = []

        if len(elements) == 0:
            return le

        # Find the set of minimal elements (no incoming edges)
        indegrees = np.sum(tc, axis=0)
        minimal_elements_indices = np.where(indegrees == 0)[0]

        if len(minimal_elements_indices) == 0:
            raise ValueError("No minimal elements found. The partial order might contain cycles.")

        # Randomly select one of the minimal elements
        idx_in_tc = random.choice(minimal_elements_indices)
        element = elements[idx_in_tc]
        le.append(element)

        # Remove the selected element from the matrix and elements list
        tc_new = np.delete(np.delete(tc, idx_in_tc, axis=0), idx_in_tc, axis=1)
        elements_new = [e for i, e in enumerate(elements) if i != idx_in_tc]

        # Recursive call
        return GenerationUtils.unifLE(tc_new, elements_new, le)

    @staticmethod
    def sample_total_order(h: np.ndarray, subset: List[int]) -> List[int]:
        """
        Sample a total order (linear extension) for a restricted partial order.

        Parameters:
        - h: The original partial order adjacency matrix.
        - subset: List of node indices to sample a linear extension for.

        Returns:
        - sampled_order: A list representing the sampled linear extension.
        """
        # Restrict the matrix to the given subset
        restricted_matrix = BasicUtils.restrict_partial_order(h, subset)

        # Initialize elements as the elements in the subset
        elements = subset.copy()
        restricted_matrix_tc = BasicUtils.transitive_closure(restricted_matrix)

        # Sample one linear extension using the `unifLE` function
        sampled_order = GenerationUtils.unifLE(restricted_matrix_tc, elements)

        return sampled_order

    @staticmethod
    def topological_sort(adj_matrix: np.ndarray) -> List[int]:
        """
        Returns one valid topological ordering of nodes in a DAG
        represented by an adjacency matrix.

        Parameters:
        - adj_matrix: n x n adjacency matrix (0/1),
                    where edge i->j means adj_matrix[i, j] == 1.

        Returns:
        - ordering: A list of node indices in topological order.
        
        Raises:
        - ValueError if the graph has a cycle or is not a DAG.
        """
        n = adj_matrix.shape[0]
        # in_degree[i] = number of incoming edges for node i
        in_degree = np.sum(adj_matrix, axis=0)

        # start with nodes that have no incoming edges
        queue = [i for i in range(n) if in_degree[i] == 0]
        ordering = []

        while queue:
            node = queue.pop()
            ordering.append(node)

            # "Remove" node from the graph => 
            # For each edge node->v, reduce in_degree[v] by 1
            for v in range(n):
                if adj_matrix[node, v] == 1:
                    in_degree[v] -= 1
                    # If v becomes a node with no incoming edges => add to queue
                    if in_degree[v] == 0:
                        queue.append(v)

        if len(ordering) != n:
            # A cycle must exist, or something prevented us from ordering all nodes
            raise ValueError("The adjacency matrix contains a cycle (not a DAG).")

        return ordering

    @staticmethod
    def generate_subsets(N: int, n: int) -> List[List[int]]:
        """
        Generate N subsets O1, O2, ..., ON where:
        - N is the number of subsets.
        - n is the size of the universal set {0, 1, ..., n-1}.
        
        Each subset Oi is created by:
        - Determining the subset size ni by uniformly sampling from [2, n].
        - Randomly selecting ni distinct elements from the set {0, 1, ..., n-1}.

        Parameters:
        - N: Number of subsets to generate.
        - n: Size of the universal set.

        Returns:
        - subsets: A list of subsets, each subset is a list of distinct integers.
        """
        subsets = []
        universal_set = list(range(n))  # Universal set from 0 to n-1

        for _ in range(N):
            # Randomly sample the subset size ni from [2, n]
            ni = random.randint(2, n)
            # Randomly select ni distinct elements from the universal set
            subset = random.sample(universal_set, ni)
            subset = sorted(subset)
            subsets.append(subset)

        return subsets

    @staticmethod
    def generate_total_orders_for_assessor(
        h_dict: Dict[int, np.ndarray],
        M_a_dict: Dict[int, List[int]],
        O_a_i_dict: Dict[int, List[List[int]]],
        prob_noise: float
    ) -> Dict[int, List[List[int]]]:
        """
        For each assessor, generate total orders (linear extensions) from their local partial order.
        
        Parameters:
        h_dict: Dictionary mapping assessor IDs to local partial order matrices (each of shape (|Mₐ|,|Mₐ|)).
        M_a_dict: Dictionary mapping assessor IDs to their ordered list of global item IDs.
                The order corresponds to the rows/columns in the local partial order matrix.
        O_a_i_dict: Dictionary mapping assessor IDs to a list of choice sets.
                    Each choice set is a list of global item IDs.
        prob_noise: The noise (jump) probability.
        
        Returns:
        Dict[int, List[List[int]]]: Mapping from assessor IDs to a list of total orders.
                                    Each total order is expressed as a list of global item IDs.
        """
        total_orders_dict = {}
        
        for a, choice_sets in O_a_i_dict.items():
            # Retrieve local partial order matrix.
            h_local = h_dict.get(a)
            if h_local is None:
                print(f"Warning: No partial order matrix found for assessor {a}. Skipping.")
                continue
            # Retrieve assessor's ordered global items.
            M_a = M_a_dict.get(a)
            if M_a is None:
                print(f"Warning: No item set found for assessor {a}. Skipping.")
                continue
            
            assessor_orders = []
            for subset in choice_sets:
                # Generate total order for this choice set.
                total_order = StatisticalUtils.generate_total_order_for_choice_set_with_queue_jump(subset, M_a, h_local, prob_noise)
                if total_order:
                    assessor_orders.append(total_order)
            total_orders_dict[a] = assessor_orders
        
        return total_orders_dict 