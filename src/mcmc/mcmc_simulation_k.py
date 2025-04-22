import itertools
import random
import time 
import numpy as np
from typing import List, Dict, Any
from scipy.stats import beta, gamma
import pandas as pd 
from src.utils import BasicUtils, StatisticalUtils
from src.mcmc.likelihood_cache import LogLikelihoodCache
# (Assuming these modules are accessible.)

def mcmc_partial_order_k(
    observed_orders: List[List[int]],
    choice_sets: List[List[int]],
    num_iterations: int,
    # Update for rho:
    X: np.ndarray,
    dr: float,  # multiplicative step size for rho
    drbeta: float,  # multiplicative step size for beta
    # Parameter controlling random-walk for mallow_theta:
    sigma_mallow: float,
    sigma_beta: float,
    noise_option: str,
    mcmc_pt: List[float],

    # Prior hyperparameters:
    rho_prior, 
    noise_beta_prior: float,
    mallow_ua: float,
    K_prior: float,
    random_seed: int
) -> Dict[str, Any]:
    """
    Perform MCMC sampling to infer the partial order h, plus parameters (rho, prob_noise, mallow_theta).
    The code includes:
      - an update for rho using a multiplicative step (dr),
      - an update for noise parameter (depending on noise_option),
      - an update for the latent matrix Z (interpreted as U).
    
    Returns a dictionary containing traces for Z, rho, noise, Mallows theta, and other diagnostics.
    """

    # ----------------------------------------------------------------
    # 1. Setup: Map items to indices, initialize states, seed, etc.
    # ----------------------------------------------------------------

    items = sorted(set(itertools.chain.from_iterable(choice_sets)))
    n = len(items)
    item_to_index = {item: idx for idx, item in enumerate(items)}
    index_to_item = {idx: item for item, idx in item_to_index.items()}

    # Convert observed orders to index form.
    observed_orders_idx = [
        [item_to_index[it] for it in order] for order in observed_orders
    ]


    rng = np.random.default_rng(random_seed)
    K = 1 # Initial number of latent dimensions from the mean of the k prior 
    # Initialize MCMC state.
    Z = np.zeros((n, K), dtype=float)  # latent matrix
    p = X.shape[0]

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
    h_trace = []
    K_trace=[]
    beta_trace = []
    update_records=[]


    rho_trace = []
    prob_noise_trace = []
    mallow_theta_trace = []

    proposed_rho_vals = []
    proposed_prob_noise_vals = []
    proposed_mallow_theta_vals = []
    proposed_beta_vals=[ ]
    proposed_Zs = []
    acceptance_decisions = []
    acceptance_rates = []
    log_likelihood_currents = []
    log_likelihood_primes = []

    num_acceptances = 0
    # -----------------[ Per-Iteration Timing Lists ]-------------


    iteration_list = []
    update_category_list = []
    prior_timing_list = []      # time for prior computations in this iteration
    likelihood_timing_list = [] # time for likelihood calculation in this iteration
    update_timing_list = [] 


    # Precompute progress intervals (10% increments).
    progress_intervals = [int(num_iterations * frac) for frac in np.linspace(0.1, 1.0, 10)]

    # Unpack update probabilities for clarity.
    rho_pct, noise_pct, U_pct , beta_pct, K_pct = mcmc_pt
    llk_current=-float("inf")
    llk_prime=-float("inf")



    # ----------------------------------------------------------------
    # 3. Main MCMC Loop
    # ----------------------------------------------------------------
    for iteration in range(1, num_iterations + 1):
        r = random.random()
        upd_start = time.time()  # Start timing the update block

        # Reset per-iteration timers
        iter_prior_time = 0.0
        iter_likelihood_time = 0.0
        iter_update_time = 0.0

        # Record iteration number
        iteration_list.append(iteration)
        accepted_this_iter = False
        update_category = None
        llk_prime = LogLikelihoodCache.calculate_log_likelihood(
                    Z, h_Z, observed_orders_idx, choice_sets, item_to_index,
                    prob_noise, mallow_theta, noise_option
                )     
        # ---- A) Update rho ----
        if r < rho_pct:
            update_category = "rho"  # For timing
            delta = random.uniform(dr, 1.0 / dr)
            rho_prime = 1.0 - (1.0 - rho) * delta
            if not (0.0 < rho_prime < 1.0):
                rho_prime = rho

            # For the rho update, assume Z remains unchanged.
            log_prior_current = StatisticalUtils.dRprior(rho,rho_prior) + StatisticalUtils.log_U_prior(Z, rho, K)
            log_prior_proposed = StatisticalUtils.dRprior(rho_prime,rho_prior) + StatisticalUtils.log_U_prior(Z, rho_prime, K)

            # Since Z is unchanged, we use the same likelihood.
            llk_prime=llk_current 

            log_acceptance_ratio = (log_prior_proposed ) - \
                                   (log_prior_current) - np.log(delta)



            acceptance_probability = min(1.0, np.exp(log_acceptance_ratio))
            if random.random() < acceptance_probability:
                rho = rho_prime
                num_acceptances += 1
                acceptance_decisions.append(1)
                accepted_this_iter = True
                llk_current=llk_prime
            else:
                acceptance_decisions.append(0)
            proposed_rho_vals.append(rho_prime)

        # ---- B) Update noise parameter ----
        elif r < (rho_pct + noise_pct):
            update_category = "noise"
            if noise_option == "mallows_noise":

                epsilon = np.random.normal(0, 1)
                mallow_theta_prime = mallow_theta * np.exp(sigma_mallow * epsilon)

                log_prior_current = StatisticalUtils.dTprior(mallow_theta, ua=mallow_ua)
                log_prior_proposed = StatisticalUtils.dTprior(mallow_theta_prime, ua=mallow_ua)

                llk_prime = LogLikelihoodCache.calculate_log_likelihood(
                    Z, h_Z, observed_orders_idx, choice_sets, item_to_index,
                    prob_noise, mallow_theta_prime, noise_option
                )
                log_acceptance_ratio = (log_prior_proposed + llk_prime) - (log_prior_current + llk_current)+ np.log(mallow_theta / mallow_theta_prime)
                acceptance_probability = min(1.0, np.exp(log_acceptance_ratio))
                if random.random() < acceptance_probability:
                    mallow_theta = mallow_theta_prime
                    num_acceptances += 1
                    acceptance_decisions.append(1)
                    accepted_this_iter = True
                    llk_current=llk_prime
                else:
                    acceptance_decisions.append(0)
                proposed_mallow_theta_vals.append(mallow_theta_prime)

            elif noise_option == "queue_jump":
                prob_noise_prime = StatisticalUtils.rPprior(noise_beta_prior)

                log_prior_current = StatisticalUtils.dPprior(prob_noise, beta_param=noise_beta_prior)
                log_prior_proposed = StatisticalUtils.dPprior(prob_noise_prime, beta_param=noise_beta_prior)


                llk_prime = LogLikelihoodCache.calculate_log_likelihood(
                    Z, h_Z, observed_orders_idx, choice_sets, item_to_index,
                    prob_noise_prime, mallow_theta, noise_option
                )



                log_acceptance_ratio = llk_prime -llk_current
                acceptance_probability = min(1.0, np.exp(log_acceptance_ratio))
                if random.random() < acceptance_probability:
                    prob_noise = prob_noise_prime
                    num_acceptances += 1
                    accepted_this_iter = True
                    acceptance_decisions.append(1)
                    llk_current=llk_prime
                else:
                    acceptance_decisions.append(0)
                proposed_prob_noise_vals.append(prob_noise_prime)
        
        # ---- C) Update U (latent matrix Z) via a single row update ----
        elif r <= (rho_pct + noise_pct +  U_pct):
            update_category = "U"
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


            llk_prime = LogLikelihoodCache.calculate_log_likelihood(
                Z_prime, h_Z_prime, observed_orders_idx, choice_sets, item_to_index,
                prob_noise, mallow_theta, noise_option
            )



            log_acceptance_ratio = (log_prior_proposed + llk_prime) - (log_prior_current + llk_current)
            acceptance_probability = min(1.0, np.exp(log_acceptance_ratio))
            if random.random() < acceptance_probability:
                Z = Z_prime
                h_Z = h_Z_prime
                num_acceptances += 1
                acceptance_decisions.append(1)
                accepted_this_iter = True
                llk_current=llk_prime
            else:
                acceptance_decisions.append(0)
            proposed_Zs.append(Z_prime)

        elif r <= (rho_pct + noise_pct + beta_pct + U_pct ):
            update_category = "beta"
            epsilon =  rng.multivariate_normal(np.zeros(p), Sigma_prop)
            beta_prime = beta + epsilon
            alpha_prime = X.T @ beta_prime   
            eta_prime = StatisticalUtils.transform_U_to_eta(Z, alpha_prime)
            h_Z_prime = BasicUtils.generate_partial_order(eta_prime)

            lp_current = StatisticalUtils.dBetaprior(beta,sigma_beta)
            lp_proposed = StatisticalUtils.dBetaprior(beta_prime,sigma_beta)

            llk_prime = LogLikelihoodCache.calculate_log_likelihood(
                Z, h_Z_prime, observed_orders_idx, choice_sets, item_to_index,
                prob_noise, mallow_theta, noise_option
            )
            log_acceptance_ratio = (lp_proposed + llk_prime) - (lp_current + llk_current)
            acceptance_probability = min(1.0, np.exp(log_acceptance_ratio))



            
            if random.random() < acceptance_probability:
                beta = beta_prime
                alpha = alpha_prime
                h_Z = h_Z_prime
                num_acceptances += 1
                acceptance_decisions.append(1)
                llk_current=llk_prime
            else:
                acceptance_decisions.append(0)
            proposed_beta_vals.append(beta_prime)           
    
        else:
            update_category = "K"
            if K == 1:
                move = "up"
            else:
                move = "up" if random.random() < 0.5 else "down"      
            if move == "up":
                K_prime = K + 1
                col_ins = random.randint(0, K)  # position to insert new column, pick from k+1 positions 
            
                b_col = StatisticalUtils.sample_conditional_column(Z, rho)  # shape (n,)
                Z_prime = np.insert(Z, col_ins, b_col, axis=1)  # => shape (n, K+1)           
                eta_prime = StatisticalUtils.transform_U_to_eta(Z_prime, alpha)

                h_Z_prime = BasicUtils.generate_partial_order(eta_prime)

                log_prior_K = StatisticalUtils.dKprior(K, K_prior)
                log_prior_K_prime = StatisticalUtils.dKprior(K_prime, K_prior)


                llk_current = LogLikelihoodCache.calculate_log_likelihood(
                    Z, h_Z, observed_orders_idx, choice_sets,
                    item_to_index, prob_noise, mallow_theta, noise_option
                )#

                llk_prime = LogLikelihoodCache.calculate_log_likelihood(
                    Z_prime, h_Z_prime, observed_orders_idx, choice_sets,
                    item_to_index, prob_noise, mallow_theta, noise_option
                )#


                log_acc = (
                    (log_prior_K_prime + llk_prime)
                    - (log_prior_K + llk_current)
                )
                accept_prob = min(1.0, np.exp(log_acc))
                if random.random() < accept_prob:
                    Z = Z_prime
                    K = K_prime
                    h_Z = h_Z_prime
                    num_acceptances += 1
                    acceptance_decisions.append(1)
                    accepted_this_iter = True
                    llk_current=llk_prime
                else:
                    acceptance_decisions.append(0)

            else:
                # move == "down"
                K_prime = K - 1
                # Pick a random column to remove from [0..K-1] 
                col_del = random.randint(0, K-1)
                Z_prime = np.delete(Z, col_del, axis=1)
                eta_prime = StatisticalUtils.transform_U_to_eta(Z_prime, alpha)

                h_Z_prime = BasicUtils.generate_partial_order(eta_prime)

                log_prior_K = StatisticalUtils.dKprior(K, K_prior)
                log_prior_K_prime = StatisticalUtils.dKprior(K_prime, K_prior)


                llk_prime = LogLikelihoodCache.calculate_log_likelihood(
                    Z_prime, h_Z_prime, observed_orders_idx, choice_sets,
                    item_to_index, prob_noise, mallow_theta, noise_option
                )

                log_acc = (log_prior_K_prime + llk_prime) - (log_prior_K + llk_current)
                accept_prob = min(1.0, np.exp(log_acc))

                if random.random() < accept_prob:
                    Z = Z_prime
                    K = K_prime
                    h_Z = h_Z_prime
                    num_acceptances += 1
                    acceptance_decisions.append(1)
                    accepted_this_iter = True
                    llk_current=llk_prime
                else:
                    acceptance_decisions.append(0)

        # Store current state. 
        if iteration % 100 == 0:
            Z_trace.append(Z.copy())
            h_trace.append(h_Z.copy())
            K_trace.append(K)
            beta_trace.append(beta)
            rho_trace.append(rho)
            prob_noise_trace.append(prob_noise)
            mallow_theta_trace.append(mallow_theta)
            update_records.append((iteration, update_category, accepted_this_iter))


        log_likelihood_currents.append(llk_current)
        log_likelihood_primes.append(llk_prime)
        current_acceptance_rate = num_acceptances / iteration
        acceptance_rates.append(current_acceptance_rate)

        if iteration in progress_intervals:
            print(f"Iteration {iteration}/{num_iterations} - Accept Rate: {current_acceptance_rate:.2%}")

    overall_acceptance_rate = num_acceptances / num_iterations
    print(f"\nOverall Acceptance Rate after {num_iterations} iterations: {overall_acceptance_rate:.2%}")
    update_df = pd.DataFrame(update_records, columns=["iteration", "category", "accepted"])
    return {
        "Z_trace": Z_trace,
        "h_trace": h_trace,
        "K_trace": K_trace,
        "beta_trace": beta_trace,
        "index_to_item": index_to_item,
        "item_to_index": item_to_index,
        "rho_trace": rho_trace,
        "prob_noise_trace": prob_noise_trace,
        "mallow_theta_trace": mallow_theta_trace,
        "proposed_rho_vals": proposed_rho_vals,
        "proposed_prob_noise_vals": proposed_prob_noise_vals,
        "proposed_mallow_theta_vals": proposed_mallow_theta_vals,
        "proposed_beta_vals": proposed_beta_vals,
        "proposed_Zs": proposed_Zs,
        "acceptance_rates": acceptance_rates,
        "acceptance_decisions": acceptance_decisions,
        "log_likelihood_currents": log_likelihood_currents,
        "log_likelihood_primes": log_likelihood_primes,
        "overall_acceptance_rate": overall_acceptance_rate,
        "update_df": update_df
    }
