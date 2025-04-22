"""
MCMC implementation for partial order inference.
"""

import sys
import itertools
import random
import numpy as np
from typing import List, Dict, Any
from scipy.stats import beta, gamma

from ..utils.basic_utils import BasicUtils
from ..utils.statistical_utils import StatisticalUtils
from .likelihood_cache import LogLikelihoodCache

def mcmc_partial_order(
    observed_orders: List[List[int]],
    choice_sets: List[List[int]],
    num_iterations: int,
    K: int,
    X: np.ndarray,
    dr: float,  # multiplicative step size for rho
    drbeta: float,  # multiplicative step size for beta
    sigma_mallow: float,
    sigma_beta: float,
    noise_option: str,
    mcmc_pt: List[float],
    rho_prior: float, 
    noise_beta_prior: float,
    mallow_ua: float,
    random_seed: int
) -> Dict[str, Any]:


    # ----------------------------------------------------------------
    # 1. Setup: Map items to indices, initialize states, seed, etc.
    # ----------------------------------------------------------------
    rng = np.random.default_rng(random_seed)
    items = sorted(set(itertools.chain.from_iterable(choice_sets)))
    n = len(items)
    item_to_index = {item: idx for idx, item in enumerate(items)}
    index_to_item = {idx: item for item, idx in item_to_index.items()}

    # Convert observed orders to index form.
    observed_orders_idx = [
        [item_to_index[it] for it in order] for order in observed_orders
    ]

    # Initialize MCMC state.
    Z = np.zeros((n, K), dtype=float)  # latent matrix

    p = X.shape[0]
    print(f"p value is : {p}")
    beta= rng.normal(loc=0.0, scale=sigma_beta, size=(p,))
    alpha = X.T @ beta
    Sigma_prop = (drbeta**2) * (sigma_beta**2) * np.eye(p)
    eta= StatisticalUtils.transform_U_to_eta(Z, alpha)
    h_Z = BasicUtils.generate_partial_order(eta)  # partial order from Z

    # Initialize parameters using the provided prior hyperparameters.
    rho = StatisticalUtils.rRprior(rho_prior)  # initial rho from its prior

    prob_noise =  StatisticalUtils.rPprior(noise_beta_prior)  # Beta(1, noise_beta_prior)
    mallow_theta =  StatisticalUtils.rTprior(mallow_ua)


    # ----------------------------------------------------------------
    # 2. Prepare Storage for MCMC results
    # ----------------------------------------------------------------
    Z_trace = []
    rho_trace = []
    beta_trace = []
    prob_noise_trace = []
    mallow_theta_trace = []
    proposed_rho_vals = []
    proposed_beta_vals = []
    proposed_prob_noise_vals = []
    proposed_mallow_theta_vals = []
    proposed_Zs = []
    acceptance_decisions = []
    acceptance_rates = []
    log_likelihood_currents = []
    log_likelihood_primes = []
    h_trace = []
    num_acceptances = 0


    # Precompute progress intervals (10% increments).
    progress_intervals = [int(num_iterations * frac) for frac in np.linspace(0.1, 1.0, 10)]

    # Unpack update probabilities for clarity.
    rho_pct, noise_pct, U_pct, beta_pct = mcmc_pt

    # ----------------------------------------------------------------
    # 3. Main MCMC Loop
    # ----------------------------------------------------------------
    for iteration in range(1, num_iterations + 1):
        r = random.random()

        # ---- A) Update rho ----
        if r < rho_pct:
            delta = random.uniform(dr, 1.0 / dr)
            rho_prime = 1.0 - (1.0 - rho) * delta
            if not (0.0 < rho_prime < 1.0):
                rho_prime = rho

            # For the rho update, assume Z remains unchanged.
            log_prior_current = StatisticalUtils.dRprior(rho,rho_prior) + StatisticalUtils.log_U_prior(Z, rho, K)
            log_prior_proposed = StatisticalUtils.dRprior(rho_prime,rho_prior) + StatisticalUtils.log_U_prior(Z, rho_prime, K)

            log_likelihood_current_value = LogLikelihoodCache.calculate_log_likelihood(
                Z, h_Z, observed_orders_idx, choice_sets, item_to_index,
                prob_noise, mallow_theta, noise_option
            )
            # Since Z is unchanged, we use the same likelihood.
            log_likelihood_proposed_value = log_likelihood_current_value

            log_acceptance_ratio = (log_prior_proposed + log_likelihood_proposed_value) - \
                                   (log_prior_current + log_likelihood_current_value) - np.log(delta)

            log_likelihood_currents.append(log_likelihood_current_value)
            log_likelihood_primes.append(log_likelihood_proposed_value)

            acceptance_probability = min(1.0, np.exp(log_acceptance_ratio))
            if random.random() < acceptance_probability:
                rho = rho_prime
                num_acceptances += 1
                acceptance_decisions.append(1)
            else:
                acceptance_decisions.append(0)
            proposed_rho_vals.append(rho_prime)

        # ---- B) Update noise parameter ----
        elif r < (rho_pct + noise_pct):
            if noise_option == "mallows_noise":
                epsilon = np.random.normal(0, 1)
                mallow_theta_prime = mallow_theta * np.exp(sigma_mallow * epsilon)

                log_prior_current = StatisticalUtils.dTprior(mallow_theta, ua=mallow_ua)
                log_prior_proposed = StatisticalUtils.dTprior(mallow_theta_prime, ua=mallow_ua)

                llk_current =LogLikelihoodCache.calculate_log_likelihood(
                    Z, h_Z, observed_orders_idx, choice_sets, item_to_index,
                    prob_noise, mallow_theta, noise_option
                )
                llk_prime =LogLikelihoodCache.calculate_log_likelihood(
                    Z, h_Z, observed_orders_idx, choice_sets, item_to_index,
                    prob_noise, mallow_theta_prime, noise_option
                )

                log_likelihood_currents.append(llk_current)
                log_likelihood_primes.append(llk_prime)

                log_acceptance_ratio = (log_prior_proposed + llk_prime) - (log_prior_current + llk_current)+ np.log(mallow_theta / mallow_theta_prime)
                acceptance_probability = min(1.0, np.exp(log_acceptance_ratio))
                if random.random() < acceptance_probability:
                    mallow_theta = mallow_theta_prime
                    num_acceptances += 1
                    acceptance_decisions.append(1)
                else:
                    acceptance_decisions.append(0)
                proposed_mallow_theta_vals.append(mallow_theta_prime)

            elif noise_option == "queue_jump":
                prob_noise_prime = StatisticalUtils.rPprior(noise_beta_prior)

                log_prior_current = StatisticalUtils.dPprior(prob_noise, beta_param=noise_beta_prior)
                log_prior_proposed = StatisticalUtils.dPprior(prob_noise_prime, beta_param=noise_beta_prior)

                llk_current = LogLikelihoodCache.calculate_log_likelihood(
                    Z, h_Z, observed_orders_idx, choice_sets, item_to_index,
                    prob_noise, mallow_theta, noise_option
                )
                llk_prime = LogLikelihoodCache.calculate_log_likelihood(
                    Z, h_Z, observed_orders_idx, choice_sets, item_to_index,
                    prob_noise_prime, mallow_theta, noise_option
                )

                log_likelihood_currents.append(llk_current)
                log_likelihood_primes.append(llk_prime)

                log_acceptance_ratio = llk_prime -llk_current
                acceptance_probability = min(1.0, np.exp(log_acceptance_ratio))
                if random.random() < acceptance_probability:
                    prob_noise = prob_noise_prime
                    num_acceptances += 1
                    acceptance_decisions.append(1)
                else:
                    acceptance_decisions.append(0)
                proposed_prob_noise_vals.append(prob_noise_prime)

        # ---- C) Update U (latent matrix Z) via a single row update ----
        elif r <= (rho_pct + noise_pct + U_pct):
            i = random.randint(0, n - 1)
            current_row = Z[i, :].copy()
            # Build a proposal covariance matrix for the row update.
            # For example, we build a matrix with off-diagonals equal to rho and diagonal equal to 1.
            Sigma  = BasicUtils.build_Sigma_rho( K,rho)


            proposed_row = np.random.multivariate_normal(current_row, Sigma)
            Z_prime = Z.copy()
            Z_prime[i, :] = proposed_row
            eta_prime = StatisticalUtils.transform_U_to_eta(Z_prime, alpha)

            h_Z_prime = BasicUtils.generate_partial_order(eta_prime)

            log_prior_current = StatisticalUtils.log_U_prior(Z, rho, K)
            log_prior_proposed = StatisticalUtils.log_U_prior(Z_prime, rho, K)

            llk_current = LogLikelihoodCache.calculate_log_likelihood(
                Z, h_Z, observed_orders_idx, choice_sets, item_to_index,
                prob_noise, mallow_theta, noise_option
            )
            llk_prime = LogLikelihoodCache.calculate_log_likelihood(
                Z_prime, h_Z_prime, observed_orders_idx, choice_sets, item_to_index,
                prob_noise, mallow_theta, noise_option
            )

            log_likelihood_currents.append(llk_current)
            log_likelihood_primes.append(llk_prime)

            log_acceptance_ratio = (log_prior_proposed + llk_prime) - (log_prior_current + llk_current)
            acceptance_probability = min(1.0, np.exp(log_acceptance_ratio))
            if random.random() < acceptance_probability:
                Z = Z_prime
                h_Z = h_Z_prime
                num_acceptances += 1
                acceptance_decisions.append(1)
            else:
                acceptance_decisions.append(0)
            proposed_Zs.append(Z_prime)

        else:
            epsilon =  rng.multivariate_normal(np.zeros(p), Sigma_prop)
            beta_prime = beta + epsilon
            alpha_prime = X.T @ beta_prime   
            eta_prime = StatisticalUtils.transform_U_to_eta(Z, alpha_prime)
            h_Z_prime = BasicUtils.generate_partial_order(eta_prime)

            lp_current = StatisticalUtils.dBetaprior(beta,sigma_beta)
            lp_proposed = StatisticalUtils.dBetaprior(beta_prime,sigma_beta)
            llk_current = LogLikelihoodCache.calculate_log_likelihood(
                Z, h_Z, observed_orders_idx, choice_sets, item_to_index,
                prob_noise, mallow_theta, noise_option
            )
            llk_prime = LogLikelihoodCache.calculate_log_likelihood(
                Z, h_Z_prime, observed_orders_idx, choice_sets, item_to_index,
                prob_noise, mallow_theta, noise_option
            )
            log_acceptance_ratio = (lp_proposed + llk_prime) - (lp_current + llk_current)
            acceptance_probability = min(1.0, np.exp(log_acceptance_ratio))


            log_likelihood_currents.append(llk_current)
            log_likelihood_primes.append(llk_prime)
            
            if random.random() < acceptance_probability:
                beta = beta_prime
                alpha = alpha_prime
                h_Z = h_Z_prime
                num_acceptances += 1
                acceptance_decisions.append(1)
            else:
                acceptance_decisions.append(0)
            proposed_beta_vals.append(beta_prime)
                

        # Store current state. 
        if iteration % 100 == 0:
            Z_trace.append(Z.copy())
            h_trace.append(h_Z.copy())
            beta_trace.append(beta.copy())
            rho_trace.append(rho)
            prob_noise_trace.append(prob_noise)
            mallow_theta_trace.append(mallow_theta)

        current_acceptance_rate = num_acceptances / iteration
        acceptance_rates.append(current_acceptance_rate)

        if iteration in progress_intervals:
            print(f"Iteration {iteration}/{num_iterations} - Accept Rate: {current_acceptance_rate:.2%}")

    overall_acceptance_rate = num_acceptances / num_iterations
    print(f"\nOverall Acceptance Rate after {num_iterations} iterations: {overall_acceptance_rate:.2%}")

    return {
        "Z_trace": Z_trace,
        "h_trace": h_trace,
        "index_to_item": index_to_item,
        "item_to_index": item_to_index,
        "rho_trace": rho_trace,
        "beta_trace": beta_trace,
        "prob_noise_trace": prob_noise_trace,
        "mallow_theta_trace": mallow_theta_trace,
        "proposed_rho_vals": proposed_rho_vals,
        "proposed_beta_vals": proposed_beta_vals,
        "proposed_prob_noise_vals": proposed_prob_noise_vals,
        "proposed_mallow_theta_vals": proposed_mallow_theta_vals,
        "proposed_Zs": proposed_Zs,
        "acceptance_rates": acceptance_rates,
        "acceptance_decisions": acceptance_decisions,
        "log_likelihood_currents": log_likelihood_currents,
        "log_likelihood_primes": log_likelihood_primes,
        "overall_acceptance_rate": overall_acceptance_rate
    }
