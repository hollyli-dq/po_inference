"""
Mallows model implementation for ranking data.
"""

import math
import numpy as np
import random
from typing import List, Optional, Dict, Any
from .basic_utils import BasicUtils

class Mallows:
    """
    Implementation of the Mallows model for ranking data.
    """

    @staticmethod
    def mallows_local_factor(i: int, theta: float, y: List[int]) -> float:
        """
        Compute the local factor for Mallows model.

        Parameters:
        -----------
        i : int
            Current item
        theta : float
            Mallows model dispersion parameter
        y : List[int]
            Current ordering

        Returns:
        --------
        float
            Local factor value
        """
        if len(y) <= 1:
            return 1.0  # If there's only one item, the local factor is 1

        # Count how many items in `y` are positioned before `i` in `y` 
        # even though `i` has a smaller index in `y`.
        idx_i = y.index(i)
        d = 0
        for a in y:
            if a != i:
                if idx_i > y.index(a):
                    d += 1

        # The denominator is sum_{k=0..(|y|-1)} e^{-theta * k}, for length(y) items
        n = len(y)
        denom = 0.0
        for k in range(n):
            denom += math.exp(-theta * k)

        numerator = math.exp(-theta * d)
        return numerator / denom

    @staticmethod
    def p_mallows_of_l_given_y(l: List[int], y: List[int], theta: float) -> float:
        """
        Compute p^{(M)}(l | y, theta) using local factors approach.
        p^{(M)}(l|y,theta) = ∏_{i=1..n} q^{(M)}(l_i | y_{i..n}, theta)

        Parameters:
        -----------
        l : List[int]
            Linear extension
        y : List[int]
            Reference ordering
        theta : float
            Mallows model parameter

        Returns:
        --------
        float
            Probability value
        """
        if len(l) != len(y):
            # if we interpret partial, or mismatch => return 0
            return 0.0

        prob_val = 1.0
        remain = list(y)  # items that haven't been "used" yet
        for item in l:
            if item not in remain:
                return 0.0
            # local factor
            gf = Mallows.mallows_local_factor(item, theta, remain)
            prob_val *= gf
            # remove 'item'
            remain.remove(item)

        return prob_val

    @staticmethod
    def f_mallows(
        y: List[int],
        h: np.ndarray,
        theta: float,
        O_i_indice: List[int]
    ) -> float:
        """
        Compute f value for Mallows model.
        O_i_indice is the list of item labels corresponding to rows/columns in h.

        Parameters:
        -----------
        y : List[int]
            Current ordering
        h : np.ndarray
            Partial order matrix
        theta : float
            Mallows model parameter
        O_i_indice : List[int]
            Item labels for matrix indices

        Returns:
        --------
        float
            f value
        """
        n = h.shape[0]
        # 1) If total => exactly 1 extension => p(l|y,theta)
        if BasicUtils.is_total_order(h):
            # single extension
            l = BasicUtils.topological_sort(h)
            # note that 'l' is in 0..n-1, but we interpret them as O_i_indice[l[i]]
            # let's convert that to actual labels
            real_l = [O_i_indice[x] for x in l]
            p_val = Mallows.p_mallows_of_l_given_y(real_l, y, theta)
            return p_val

        # 2) If h is empty => sum=1
        if np.sum(h) == 0:
            return 1.0

        # 3) Sum over 'tops'
        tops = BasicUtils.find_tops(h)
        f_val = 0.0
        for t in tops:
            k_label = O_i_indice[t]    # the actual item label
            # local factor
            gk = Mallows.mallows_local_factor(k_label, theta, y)

            # remove row t, col t
            h_sub = np.delete(np.delete(h, t, axis=0), t, axis=1)
            # remove item from O_i_indice
            O_i_sub = O_i_indice[:t] + O_i_indice[t+1:]

            # remove k_label from y
            y_sub = [itm for itm in y if itm != k_label]

            f_k = Mallows.f_mallows(y_sub, h_sub, theta, O_i_sub)
            f_val += gk * f_k

        return f_val

    @staticmethod
    def compute_mallows_likelihood(
        y: List[int],
        h: np.ndarray,
        theta: float,
        O_i_indice: Optional[List[int]] = None
    ) -> float:
        """
        Compute p^{(M)}(y | h, theta) = f / count

        Parameters:
        -----------
        y : List[int]
            Observed ranking
        h : np.ndarray
            Partial order matrix
        theta : float
            Mallows model parameter
        O_i_indice : Optional[List[int]]
            Item labels for matrix indices

        Returns:
        --------
        float
            Likelihood value
        """
        if O_i_indice is None:
            O_i_indice = list(range(len(y)))

        f_val = Mallows.f_mallows(y, h, theta, O_i_indice)
        tr_h = BasicUtils.transitive_reduction(h)
        c_val = BasicUtils.nle(tr_h)
        if c_val == 0:
            return 0.0
        return f_val / c_val

    @staticmethod
    def generate_total_order_noise_mallow(
        y: List[int],
        h: np.ndarray,
        theta: float,
        O_indices: List[int]
    ) -> List[int]:
        """
        Recursively generate a total order using the pure Mallows model.

        Parameters:
        -----------
        y : List[int]
            Items to be ordered
        h : np.ndarray
            Partial order matrix
        theta : float
            Mallows model parameter
        O_indices : List[int]
            Item labels for matrix indices

        Returns:
        --------
        List[int]
            Generated total order
        """
        # Base cases
        if len(y) == 0:
            return []
        if len(y) == 1:
            return y

        # Find valid candidates (tops)
        tops = BasicUtils.find_tops(h)

        # Compute probabilities for each candidate
        candidate_probs = []
        for local_idx in tops:
            candidate = O_indices[local_idx]
            gf = Mallows.mallows_local_factor(candidate, theta, y)
            candidate_probs.append(gf)

        # Normalize probabilities
        total_prob = sum(candidate_probs)
        candidate_probs = [p / total_prob for p in candidate_probs]

        # Sample one candidate
        chosen_top_idx = random.choices(tops, weights=candidate_probs, k=1)[0]
        chosen_item = O_indices[chosen_top_idx]

        # Update remaining items and structures
        y_new = [item for item in y if item != chosen_item]
        h_new = np.delete(np.delete(h, chosen_top_idx, axis=0), chosen_top_idx, axis=1)
        O_new = O_indices[:chosen_top_idx] + O_indices[chosen_top_idx+1:]

        # Recursive call
        return [chosen_item] + BasicUtils.generate_total_order_mallows_no_jump(y_new, h_new, theta, O_new) 