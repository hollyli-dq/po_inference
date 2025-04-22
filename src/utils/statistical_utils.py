import numpy as np
import networkx as nx
import random
import math
from typing import List, Dict, Any
from collections import defaultdict
from scipy.stats import multivariate_normal, beta, gamma, expon, norm
from typing import List, Dict, Tuple, Union

from .basic_utils import BasicUtils

class StatisticalUtils:
    """
    Utility class for statistical computations related to partial orders.
    """   
    @staticmethod
    def rBetaPrior(sigma_beta: Union[float, np.ndarray], p: int) -> np.ndarray:
        """
        Sample a new beta from a Normal(0, Sigma) distribution, where Sigma is a diagonal matrix.
        
        Parameters:
        -----------
        sigma_beta: float or np.ndarray
            If scalar: the same prior std dev for each coefficient (diagonal elements will be sigma_beta^2)
            If array: different prior std dev for each coefficient (must have length p)
        p: integer
            Dimension of beta.
        
        Returns:
        --------
        np.ndarray
            Sampled beta vector of shape (p,)
        """
        if np.isscalar(sigma_beta):
            return np.random.normal(loc=0.0, scale=sigma_beta, size=(p,))
        else:
            if len(sigma_beta) != p:
                raise ValueError(f"If sigma_beta is an array, it must have length {p}")
            return np.random.normal(loc=0.0, scale=sigma_beta, size=(p,))

    @staticmethod
    def dBetaprior(beta: np.ndarray, sigma_beta: Union[float, np.ndarray]) -> float:
        """
        Log-pdf of a multivariate Normal(0, Sigma) at point 'beta', where Sigma is a diagonal matrix.
        
        Parameters:
        -----------
        beta: shape (p,)
            Vector of coefficients
        sigma_beta: float or np.ndarray of shape (p,)
            The prior standard deviation(s) for each coefficient.
            Can be either a scalar (same std dev for all coefficients) or an array (different std dev per coefficient)
        
        Returns:
        --------
        float
            The log-density value
            
        Notes:
        ------
        When sigma_beta is a scalar, formula is:
          - (p/2) * log(2*pi) 
          - p*log(sigma_beta)
          - (1 / (2*sigma_beta^2)) * sum(beta^2)
          
        When sigma_beta is an array, formula is:
          - (p/2) * log(2*pi) 
          - sum(log(sigma_beta))  # sum of logs instead of p times log of one value
          - sum(beta^2 / (2*sigma_beta^2))  # element-wise division by the variances
        """
        p = len(beta)
        
        if np.isscalar(sigma_beta):
            # Original implementation for scalar sigma_beta
            log_det_part = -0.5 * p * math.log(2.0 * math.pi) - p * math.log(sigma_beta)
            quad_part = -0.5 * np.sum(beta**2) / (sigma_beta**2)
        else:
            # Handle array case
            if len(sigma_beta) != p:
                raise ValueError(f"sigma_beta must be a scalar or have length {p} to match beta")
            
            log_det_part = -0.5 * p * math.log(2.0 * math.pi) - np.sum(np.log(sigma_beta))
            quad_part = -0.5 * np.sum(beta**2 / (sigma_beta**2))
            
        return log_det_part + quad_part   
    def count_unique_partial_orders(h_trace):
        """
        Count the frequency of each unique partial order in h_trace.
        
        Parameters:
        - h_trace: List of NumPy arrays representing partial orders.
        
        Returns:
        - Dictionary with partial order representations as keys and their counts as values.
        """
        unique_orders = defaultdict(int)
        
        for h_Z in h_trace:
            # Convert the matrix to a tuple of tuples for immutability
            h_tuple = tuple(map(tuple, h_Z))
            unique_orders[h_tuple] += 1
    

        sorted_unique_orders = sorted(unique_orders.items(), key=lambda x: x[1], reverse=True)
        
        # Convert the sorted tuples back to NumPy arrays for readability
        sorted_unique_orders = [(np.array(order), count) for order, count in sorted_unique_orders]
        return sorted_unique_orders
    @staticmethod
    def log_U_prior(Z: np.ndarray, rho: float, K: int, debug: bool = False) -> float:
        """
        Compute the log prior probability of Z.

        Parameters:
        - Z: Current latent variable matrix (numpy.ndarray).
        - rho: Step size for proposal (used here to scale covariance).
        - K: Number of dimensions.
        - debug: If True, prints the covariance matrix.

        Returns:
        - log_prior: Scalar log prior probability.
        """
        # Covariance matrix is scaled identity matrix
        Sigma =BasicUtils.build_Sigma_rho(K,rho)

        if debug:
            print(f"Covariance matrix Sigma:\n{Sigma}")

        # Compute log prior for each row in Z assuming independent MVN
        try:
            mvn = multivariate_normal(mean=np.zeros(K), cov=Sigma)
            log_prob = mvn.logpdf(Z)
            log_prior = np.sum(log_prob)
        except np.linalg.LinAlgError as e:
            print(f"LinAlgError in log_prior: {e}")
            print(f"Covariance matrix Sigma:\n{Sigma}")
            raise e

        return log_prior

    @staticmethod
    def description_partial_order(h: np.ndarray) -> Dict[str, Any]:
        """
        Provides a detailed description of the partial order represented by the adjacency matrix h.

        Parameters:
        - h: An n x n numpy array representing the adjacency matrix of the partial order.

        Returns:
        - description: A dictionary containing descriptive statistics of the partial order.
        """
        G = nx.DiGraph(h)
        n = h.shape[0]
        node_num= G.number_of_nodes()

        # Number of relationships (edges)
        num_relationships = G.number_of_edges()

        # Number of alone nodes (no incoming or outgoing edges)
        alone_nodes = [node for node in G.nodes() if G.in_degree(node) == 0 and G.out_degree(node) == 0]
        num_alone_nodes = len(alone_nodes)

        # Maximum number of relationships a node can have with other nodes
        # Considering both in-degree and out-degree
        in_degrees = dict(G.in_degree())
        out_degrees = dict(G.out_degree())
        max_in_degree = max(in_degrees.values()) if in_degrees else 0
        max_out_degree = max(out_degrees.values()) if out_degrees else 0
        max_relationships = max(max_in_degree, max_out_degree)

        # Number of linear extensions
        tr = BasicUtils.transitive_reduction(h)
        num_linear_extensions = BasicUtils.nle(tr)

        # Depth of the partial order (length of the longest chain)
        try:
            depth = nx.dag_longest_path_length(G)
        except nx.NetworkXUnfeasible:
            depth = None  # If the graph is not a DAG

        description = {
            "Number of Nodes": node_num,
            "Number of Relationships": num_relationships,
            "Number of Alone Nodes": num_alone_nodes,
            "Alone Nodes": alone_nodes,
            "Maximum In-Degree": max_in_degree,
            "Maximum Out-Degree": max_out_degree,
            "Maximum Relationships per Node": max_relationships,
            "Number of Linear Extensions": num_linear_extensions,
            "Depth of Partial Order": depth
        }

        # Print the description
        print("\n--- Partial Order Description ---")
        for key, value in description.items():
            print(f"{key}: {value}")
        print("---------------------------------")



    @staticmethod
    def sample_conditional_z(Z, rZ, cZ, rho):
        K = Z.shape[1]

        # Build correlation matrix
        Sigma = np.full((K, K), rho)
        np.fill_diagonal(Sigma, 1.0)

        dependent_ind = cZ
        given_inds = [i for i in range(K) if i != cZ]

        Sigma_dd = Sigma[dependent_ind, dependent_ind]  # scalar
        Sigma_dg = Sigma[dependent_ind, given_inds]     # shape (K-1,)  <-- FIXED HERE
        Sigma_gg = Sigma[np.ix_(given_inds, given_inds)]

        # X_g is also shape (K-1,)
        X_g = Z[rZ, given_inds]

        # Means are 0
        mu_d = 0.0
        mu_g = 0.0

        # Invert Sigma_gg
        try:
            Sigma_gg_inv = np.linalg.inv(Sigma_gg)
        except np.linalg.LinAlgError:
            Sigma_gg += np.eye(Sigma_gg.shape[0]) * 1e-8
            Sigma_gg_inv = np.linalg.inv(Sigma_gg)

        # Conditional mean
        mu_cond = mu_d + Sigma_dg @ Sigma_gg_inv @ (X_g - mu_g)
        # Conditional variance
        var_cond = Sigma_dd - Sigma_dg @ Sigma_gg_inv @ Sigma_dg

        var_cond = max(var_cond, 1e-8)
        Z_new = np.random.normal(loc=mu_cond, scale=np.sqrt(var_cond))

        return Z_new

#############################################Hyperparameter prior for HPO#############################################

#   ### rho 
    @staticmethod
    def rRprior(fac=1/6, tol=1e-4):
        """
        Draw a sample for ρ from a Beta(1, fac) distribution, but reject any sample
        for which 1 - ρ < tol, to avoid numerical instability when ρ is extremely close to 1.
        
        Parameters:
        fac: Second parameter of the Beta distribution (default 1/6).
        tol: Tolerance such that we require 1 - ρ >= tol (default 1e-4).
        
        Returns:
        A single float value for ρ.
        """
        while True:
            rho = beta.rvs(1, fac)
            if 1 - rho >= tol:
                return rho
    @staticmethod
    def dRprior(rho: float, fac=1/6, tol=1e-4) -> float:
        """
        Compute the log prior for ρ from a Beta(1, fac) distribution, with truncation
        at 1 - tol. If ρ > 1 - tol, return -Inf. Otherwise, adjust the log density
        by subtracting the log cumulative probability at 1-tol.
        
        Parameters:
        rho: the value of ρ.
        fac: the Beta distribution second parameter (default 1/6).
        tol: tolerance for the upper bound (default 1e-4).
        
        Returns:
        The log density (a float).
        """
        if rho > 1 - tol:
            return -np.inf
        # Compute the log PDF at rho.
        log_pdf = beta.logpdf(rho, 1, fac)
        # Subtract the log of the cumulative probability up to 1-tol, effectively renormalizing.
        log_cdf_trunc = beta.logcdf(1 - tol, 1, fac)
        return log_pdf - log_cdf_trunc
####Prob 

    @staticmethod
    def rPprior(noise_beta_prior):
        return beta.rvs(1, noise_beta_prior)
    
    @staticmethod
    def dPprior(p, beta_param):
        """
        Log-prior for p ~ Beta(1, beta_param).

        Returns -inf if p is out of (0,1).
        Otherwise, logpdf of Beta(1, beta_param).
        """
        if p <= 0.0 or p >= 1.0:
            return -math.inf
        
        return beta.logpdf(p, 1.0, beta_param)

### Tau 
    @staticmethod
    def rTauprior():
        return random.uniform(0, 1) 


    @staticmethod
    def dTauprior(tau):
        return 1

####Theta  

    @staticmethod
    def rTprior(mallow_ua):
        return gamma.rvs(a=1, scale=1.0/mallow_ua)
    @staticmethod
    def dTprior(mallow_theta, ua):
        """
        Log-prior for mallow_theta under an Exponential(ua) distribution.
        i.e. p(mallow_theta) = Exponential(ua) with pdf:
            p(mallow_theta) = ua * exp(-ua * mallow_theta), for mallow_theta > 0.
        """
        if mallow_theta <= 0:
            return -np.inf

        return expon.logpdf(mallow_theta, scale=1/ua)
    
####K  
    @staticmethod
    def dKprior(k: int, lam: float) -> float:
        """Log PMF of Poisson(λ) truncated at k ≥ 1."""
        if k < 1:
            return -np.inf
        # log(k!) using gammaln(k+1)
        log_k_fact = math.lgamma(k+1)
        # normalizing constant for truncation
        norm_const = -np.log(1 - np.exp(-lam))
        val = -lam + k * np.log(lam) - log_k_fact + norm_const
        return val
    
    @staticmethod
    def rKprior(current_K: int, lam: float = 3.0) -> int:
        """Propose new K with 50% chance to increase/decrease."""
        if current_K == 1:  # Can't decrease below 1
            return 2 if np.random.rand() < 0.5 else 1
        else:
            return current_K + 1 if np.random.rand() < 0.5 else current_K - 1
    


    
    @staticmethod
    def log_U_hierarchical_prior(
        U0: np.ndarray,                  # shape (|M0|, K)
        U_a_list: list,                  # length A, each shape (|M_a|, K)
        M_a_dict: list,                  # length A, each is a list of global object indices
        tau: float,
        Sigma_rho: np.ndarray       # shape (K,K)
                    # function log_mvnorm(x, mean, cov) -> float
    ) -> float:

        logp = 0.0

        # 1) log for each U^(0)[j,:] ~ N(0, Sigma_rho)
        n_global = U0.shape[0]
        for j in range(n_global):
            x_j = U0[j,:]                # a 1D vector of length K
            zero_vec = np.zeros_like(x_j)
            logp += np.log(multivariate_normal(x_j, zero_vec, Sigma_rho))

        # 2) for each assessor a, for each j in M_a
        A = len(U_a_list)
        for a_idx in range(A):
            Ua = U_a_list[a_idx]        # shape (|M_a|, K)
            Ma = M_a_dict.get(a_idx,[])            # list of global indices
            for row_loc, j_global in enumerate(Ma):
                # row in U^(a) => U_a_list[a_idx][row_loc,:]
                x_aj = Ua[row_loc,:]
                # mean is tau * U0[j_global,:]
                mean_aj = tau * U0[j_global,:]
                # cov is (1 - tau^2)*Sigma_rho                
                cov_aj = (1.0 - tau**2) * Sigma_rho


                logp += np.log(multivariate_normal(x_aj, mean_aj, cov_aj))

        return logp
    @staticmethod
    def sample_conditional_column(Z, rho):
        """
        Z is shape (n, K). For each row i, we want the bridging col
        of shape (n,) that respects the correlation among columns.
        
        We assume an equicorrelation or some covariance Sigma_full 
        of shape (K+1, K+1).
        """
        n, K = Z.shape
        Kplus1 = K + 1

        # Build the (K+1)x(K+1) covariance:
        Sigma_full = BasicUtils.build_Sigma_rho(Kplus1, rho)
        # Partition Sigma_full:
        # Sigma_gg = Sigma_full[0:K,0:K]
        # Sigma_dg = Sigma_full[K,   0:K]
        # Sigma_dd = Sigma_full[K,   K]
        
        Sigma_gg = Sigma_full[:K, :K]
        Sigma_dg = Sigma_full[K, :K]       # shape (K,)
        Sigma_dd = Sigma_full[K, K]        # scalar

        # Invert Sigma_gg once for all
        Sigma_gg_inv = np.linalg.inv(Sigma_gg)

        bridging_col = np.zeros(n)
        for i in range(n):
            x_i = Z[i,:]  # existing coords
            # conditional mean
            mu_cond = Sigma_dg @ Sigma_gg_inv @ x_i
            # conditional var
            var_cond = Sigma_dd - Sigma_dg @ Sigma_gg_inv @ Sigma_dg
            # sample
            bridging_col[i] = np.random.normal(mu_cond, np.sqrt(var_cond))

        return bridging_col
    @staticmethod
    def U0_conditional_update(
        j_global,        # index of the row in U0 we want to update
        U0,              # current U0, shape (n_global, K)
        U_a_dict,        # dictionary of child latents {a: U^a}, each shape (len(M_a), K)
        M_a_dict,        # {a: list_of_indices_in_M_a}, tells which global indices belong to assessor a
        tau,             # correlation parameter
        Sigma_rho,       # K x K covariance for the base distribution
        rng              # random number generator, e.g., np.random.default_rng()
    ):
        """
        Perform a direct Gibbs draw for row j_global of U0 given all child rows U^(a).
        """
        # 1) Gather all child-latent vectors that correspond to the same "global" item j_global
        #    For each a in U_a_dict, find the local index i_loc where j_global appears in M_a_dict[a].
        #    If j_global is not in M_a_dict[a], skip it. Otherwise get U_a[i_loc].
        child_vectors = []
        for a, U_a in U_a_dict.items():
            if j_global in M_a_dict[a]:
                i_loc = M_a_dict[a].index(j_global)
                child_vectors.append(U_a[i_loc, :])
        
        A_j = len(child_vectors)  # how many assessors actually have j_global in their list

        # 2) If no child has j_global, posterior = prior => Normal(0, Sigma_rho)
        if A_j == 0:
            post_mean = np.zeros_like(U0[j_global, :])
            post_cov = Sigma_rho
        else:
            # 3) Compute the posterior mean & covariance for that row
            sum_child = np.sum(child_vectors, axis=0)  # sum_{a=1..A_j} u_j^(a)

            denom = (1 - tau**2) + A_j * (tau**2)
            # Posterior mean
            post_mean = (tau / denom) * sum_child

            # Posterior covariance
            shrink_factor = (1 - tau**2) / denom
            post_cov = shrink_factor * Sigma_rho
        
        # 4) Draw a new sample from N(post_mean, post_cov)
        new_row = rng.multivariate_normal(post_mean, post_cov)
    

        return new_row 

####################Below are separate coding for hpo####
    @staticmethod
    def gumbel_inv_cdf(p: float, eps: float = 1e-15) -> float:
        # Clip p so it lies in [eps, 1 - eps] to avoid log(0)
        p_clipped = np.clip(p, eps, 1 - eps)
        return -np.log(-np.log(p_clipped))
    
    @staticmethod
    def log_U_a_prior(U_a_dict: Dict[int, np.ndarray], tau: float, rho: float, K: int, M_a_dict: Dict[int, List[int]], U0: np.ndarray) -> float:
        """
        Compute the log prior probability for assessor-level latent variables.
        
        Each assessor a has latent variables U_a ~ N(tau * U0[j], (1 - tau^2)*Sigma_rho)
        for each global item j in M_a_dict[a].

        Parameters:
        U_a_dict: Dictionary with keys as assessor IDs and values as latent matrices (shape: (|M_a|, K)).
        tau: The branching parameter.
        rho: The correlation parameter.
        K: Dimensionality of the latent space.
        M_a_dict: Dictionary with keys as assessor IDs and values as lists of global item indices for that assessor.
        U0: Global latent matrix (shape: (|M0|, K)).

        Returns:
        log_prior_total: The sum of log prior probabilities over all assessor-level latent variables.
        """
        Sigma_rho =BasicUtils.build_Sigma_rho(K,rho)
        log_prior_total = 0.0

        for a, U_a in U_a_dict.items():
            # Get the list of global items for assessor a.
            Ma = M_a_dict.get(a, [])
            log_prior = 0.0
            for i, j in enumerate(Ma):
                mean_vec = tau * U0[j, :]

                cov_mat= (1.0 - tau**2) * Sigma_rho
                log_prob = multivariate_normal.logpdf(
                            U_a[i, :],
                            mean=mean_vec,
                            cov=cov_mat,
                            allow_singular=True
                        )

            log_prior += log_prob
            log_prior_total += log_prior

        return log_prior_total
    @staticmethod
    def transform_U_to_eta(U: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """
        Transform latent positions U to eta using Gumbel link function.
        
        Parameters:
        -----------
        U : np.ndarray
            Matrix of latent positions (n_global × K)
        alpha : np.ndarray
            Vector of assessor/item effects (n_global × 1)
            
        Returns:
        --------
        np.ndarray
            Matrix of transformed positions (n_global × K)
        """
        n_global, K = U.shape
        
        # Initialize output matrix
        eta = np.zeros((n_global, K))
        
        # Gumbel inverse link function
        def gumbel_inv(p):
            return -np.log(-np.log(p))
        
        # Transform each row
        for j in range(n_global):
            # Step 1: Convert to probabilities using normal CDF
            p_vec = norm.cdf(U[j, :])
            
            # Step 2: Apply Gumbel inverse link function
            gumbel_vec = np.array([gumbel_inv(px) for px in p_vec])
            
            # Step 3: Add assessor/item effect
            eta[j, :] = gumbel_vec + alpha[j]
        
        return eta
    @staticmethod
    ### The objective of this function is buidling a hierarchical partial order of H(U) given M0, Ma, Oa_list and U_alist 
    def build_hierarchical_partial_orders(
        M0,
        assessors,
        M_a_dict,
        U0,           # shape: (|M0|, K)
        U_a_dict,
        alpha,       
        link_inv=None
    ):
        if link_inv is None:
            # Default to Gumbel quantile
            link_inv = StatisticalUtils.gumbel_inv_cdf

        n_global, K = U0.shape
        eta0 = np.zeros_like(U0)
        for j_global in range(n_global):
            p_vec = norm.cdf(U0[j_global, :])  # coordinate-wise
            gumbel_vec = np.array([link_inv(px) for px in p_vec])
            eta0[j_global, :] = gumbel_vec + alpha[j_global]
        
        h0 = BasicUtils.generate_partial_order(eta0)

        h_U = {}
        h_U={0:h0}

        # Loop over assessors
        for idx_a, a in enumerate(assessors):
            # 1) Build the *full partial order* on M_a
            Ma = M_a_dict.get(a,[])               # e.g. [0,2,4]
            Ua = U_a_dict.get(a,[])               # shape (|M_a|, K)
            # (a) Compute eta^(a) for each item j in M_a
            #     eqn (21): eta_j^{(a)} = G^-1( Phi(U_j^{(a)}) ) + alpha_j
            # We do it row by ro
            eta_a = np.zeros_like(Ua)
            for i_loc, j_global in enumerate(Ma):
                p_vec = norm.cdf(Ua[i_loc, :])
                gumbel_vec = np.array([link_inv(px) for px in p_vec])
                eta_a[i_loc, :] = gumbel_vec + alpha[j_global]

            # adjacency for M_a
            h_a = BasicUtils.generate_partial_order(eta_a)
            # store in dictionary
            h_U[a] = h_a

        return h_U
    
    @staticmethod
    def dict_array_equal(d1, d2):
        """Recursively compare two dictionaries where values may be NumPy arrays."""
        if d1.keys() != d2.keys():
            return False
        for key in d1:
            v1, v2 = d1[key], d2[key]
            if isinstance(v1, dict) and isinstance(v2, dict):
                if not StatisticalUtils.dict_array_equal(v1, v2):
                    return False
            elif isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray):
                if not np.array_equal(v1, v2):
                    return False
            else:
                if v1 != v2:
                    return False
        return True
    @staticmethod
    def generate_total_order_for_choice_set_with_queue_jump(
        subset: List[int],
        M_a: List[int],
        h_local: np.ndarray,
        prob_noise: float
    ) -> List[int]:
        """
        Given:
        - 'subset': A list of *global* item IDs we want to order.
        - 'M_a': The assessor's entire set of global item IDs (size = |M_a|).
        - 'h_local': A local partial-order matrix of shape (|M_a|, |M_a|),
                    where h_local[i,j]=1 => "item M_a[i] < item M_a[j]" in the assessor's order.
        - 'prob_noise': Probability of a 'jump' (i.e., random pick) in the queue-jump.

        Returns:
        A total order of items in 'subset', as a list of global item IDs.
        """

        # 1) Build a map from *global* item => local index in M_a
        #    so we can slice h_local properly.
        global2local = { g: i for i, g in enumerate(M_a) }

        # 2) Identify which global items in 'subset' are also in M_a,
        #    and convert them to local indices
        local_subset_idx = []
        local_subset_global = []  # store the same items, but parallel to local indices
        for g in subset:
            if g in global2local:          # only items that exist in M_a
                local_idx = global2local[g]
                local_subset_idx.append(local_idx)
                local_subset_global.append(g)

        # If no overlap, return empty
        if not local_subset_idx:
            return []

        # 3) Extract the local submatrix for these items
        #    shape = (len(local_subset_idx), len(local_subset_idx))
        h_matrix_subset = h_local[np.ix_(local_subset_idx, local_subset_idx)]

        # 4) We'll do the queue-jump logic in local SUBSET indices = [0..(n_sub-1)]
        n_sub = len(local_subset_idx)
        # So we make a direct mapping from "subset index" => "global item ID"
        # e.g. subset_idx2global[i] = local_subset_global[i]
        # And we'll keep 'remaining' as [0..n_sub-1].
        subset_idx2global = { i: local_subset_global[i] for i in range(n_sub) }

        remaining = list(range(n_sub))  # local indices in [0..n_sub-1]
        total_order_local = []


        while remaining:
            m = len(remaining)
            if m == 1:
                total_order_local.append(remaining[0])
                break

            # Build sub-submatrix for 'remaining'
            # shape => (m, m)
            h_rem = h_matrix_subset[np.ix_(remaining, remaining)]

            # Transitive reduction of that sub-submatrix
            tr_rem = BasicUtils.transitive_reduction(h_rem)

            # Count total # of linear extensions
            N_total = BasicUtils.nle(tr_rem)

            # Compute candidate probabilities for each local_idx in [0..m-1]
            candidate_probs = []
            for local_idx in range(m):
                # Number of linear extensions that start with 'local_idx'
                # This uses BasicUtils.num_extensions_with_first
                # but that function expects the partial order submatrix + top elements, etc.
                # So local_idx is an index in [0..m-1].
                # We pass 'tr_rem' and local_idx to BasicUtils.num_extensions_with_first
                N_first = BasicUtils.num_extensions_with_first(tr_rem, local_idx)
                p_no_jump = (1 - prob_noise) * (N_first / N_total)
                candidate_probs.append(p_no_jump)

            # Probability of 'jump' => prob_noise, distributed uniformly among m candidates
            p_jump = prob_noise * (1.0 / m)
            candidate_probs = [p + p_jump for p in candidate_probs]

            # normalize
            total_p = sum(candidate_probs)
            candidate_probs = [p / total_p for p in candidate_probs]

            # Sample an index from 'remaining' with these weights
            chosen_subindex = random.choices(range(m), weights=candidate_probs, k=1)[0]
            chosen_local = remaining[chosen_subindex]

            total_order_local.append(chosen_local)
            remaining.remove(chosen_local)

        # 5) Convert 'total_order_local' (which are indices in [0..n_sub-1])
        #    back to *global* item IDs
        global_order = [subset_idx2global[i] for i in total_order_local]

        return global_order


