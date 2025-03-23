"""
Likelihood cache implementation for efficient MCMC sampling.
"""

import sys
import math
import numpy as np
from scipy.stats import beta
import copy
from typing import Dict, List, Optional, Union, Tuple, Any
from functools import lru_cache

from ..utils.basic_utils import BasicUtils
from ..utils.mallows import Mallows

class LogLikelihoodCache:
    # Class-level dictionaries for caching
    nle_cache = {}
    nle_first_cache = {}

    @staticmethod
    def _matrix_key(adj_matrix: np.ndarray) -> bytes:
        """Convert adjacency matrix to a bytes object as a cache key."""
        return adj_matrix.tobytes()

    @classmethod
    def _get_nle(cls, adj_matrix: np.ndarray) -> int:
        """Retrieve or compute the number of linear extensions with caching."""
        key = cls._matrix_key(adj_matrix)
        if key in cls.nle_cache:
            return cls.nle_cache[key]
        val = BasicUtils.nle(adj_matrix)
        cls.nle_cache[key] = val
        return val

    @classmethod
    def _get_nle_first(cls, adj_matrix: np.ndarray, local_idx: int) -> int:
        """Retrieve or compute the number of extensions with a specific first item."""
        matrix_key = cls._matrix_key(adj_matrix)
        cache_key = (matrix_key, local_idx)
        if cache_key in cls.nle_first_cache:
            return cls.nle_first_cache[cache_key]
        val = BasicUtils.num_extensions_with_first(adj_matrix, local_idx)
        cls.nle_first_cache[cache_key] = val
        return val

    @classmethod
    def calculate_log_likelihood(
        cls,
        Z, 
        h_Z, 
        observed_orders_idx, 
        choice_sets, 
        item_to_index,
        prob_noise, 
        mallow_theta, 
        noise_option
    ):
        if noise_option not in ["queue_jump", "mallows_noise"]:
            raise ValueError(f"Invalid noise_option: {noise_option}. Valid options are ['queue_jump', 'mallows_noise'].")

        log_likelihood = 0.0

        for idx, y_i in enumerate(observed_orders_idx):
            O_i = choice_sets[idx]
            O_i_indices = sorted([item_to_index[item] for item in O_i])
            m = len(y_i)

            if noise_option == "queue_jump":
                for j, y_j in enumerate(y_i):
                    remaining_indices = y_i[j:]
                    h_Z_remaining = h_Z[np.ix_(remaining_indices, remaining_indices)]
                    tr_remaining = BasicUtils.transitive_reduction(h_Z_remaining)
                    num_le = cls._get_nle(tr_remaining)
                    local_idx = remaining_indices.index(y_j)
                    num_first_item = cls._get_nle_first(tr_remaining, local_idx)

                    prob_no_jump = (1 - prob_noise) * (num_first_item / num_le)
                    prob_jump = prob_noise * (1 / (m - j))
                    prob_observed = prob_no_jump + prob_jump
                    log_likelihood += math.log(prob_observed)

            elif noise_option == "mallows_noise":
                h_Z_Oi = h_Z[np.ix_(O_i_indices, O_i_indices)]
                mallows_prob = Mallows.compute_mallows_likelihood(
                    y=y_i,
                    h=h_Z_Oi,
                    theta=mallow_theta,
                    O_i_indice=O_i_indices
                )
                log_likelihood += math.log(mallows_prob if mallows_prob > 0 else 1e-20)

        return log_likelihood
