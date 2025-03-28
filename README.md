# Bayesian Partial Order Inference

A Python package for Bayesian inference of strong partial orders from noisy observations using Markov Chain Monte Carlo (MCMC) methods. This implementation is based on the framework described in [Muir Watt et al. (2012)](https://doi.org/10.1214/12-AOS1029).

## Features

- Sampling partial orders and also the total orders given hyperparameters
- Bayesian inference of strong partial orders using MCMC
- Support for different noise models:
  - Queue jump noise model
  - Mallows noise model
- Visualization of:
  - MCMC traces
  - Inferred partial orders
  - True vs. inferred order comparisons
- Comprehensive logging and result storage
- Configurable MCMC parameters and priors

## The Bayesian Partial Order 

### Partial Order

A strong partial order is a binary relation $\,\prec\,$ over a set of items that satisfies:

- Irreflexivity: $\,\neg(a \prec a)\,$
- Antisymmetry: if $\,a \prec b\,$ then $\,\neg(b \prec a)\,$
- Transitivity: if $\,a \prec b\,$ and $\,b \prec c\,$ then $\,a \prec c\,$

### Latent Space Model

The model uses a latent space representation where:

- Each item $j$ has a $K$-dimensional latent position $U_j \in \mathbb{R}^K$.
- The correlation between dimensions is controlled by the parameter $\rho$.
- The transformed latent positions $\eta_i$ are given by $\eta_i = U_i + \alpha_i$, where $\alpha_i$ represents covariate effects, e.g. $\beta_j \times x_j$.

The mapping from $\eta$ to the partial order $h$ is defined as:
$h_{ij} = \begin{cases}
1 & \text{if } \eta_i \prec \eta_j \\
0 & \text{otherwise}
\end{cases}$

#### Theorem (Partial Order Model)

For $\alpha$ and $\Sigma_\rho$ defined above, if we take:

- $U_{j,:} \sim \mathcal{N}(0, \Sigma_\rho)$ independently for each $j \in M$,
- $\eta_{j,:} = G^{-1}\bigl(\Phi(U_{j,:})\bigr) + \alpha_j\,1_K^T$, where $\alpha_j = \beta_j \times x_j$,
- $y \sim p\bigl(\cdot \mid h(\eta(U, \beta))\bigr)$,

then certain partial‐order properties follow (see the reference for detailed proofs).

### MCMC Inference

The posterior distribution is given by:
$\pi(\rho, U, \beta \mid Y) \;\propto\; \pi(\rho)\,\pi(\beta)\,\pi(U \mid \rho)\,p\bigl(Y \mid h(\eta(U,\beta))\bigr).$

We sample from this posterior using MCMC. Specific update steps include:

- **Updating $\rho$:** Using a Beta prior (e.g. $\text{Beta}(1, \rho_\text{prior})$) with mean around 0.9.
- **Updating $p_{\mathrm{noise}}$:** Using a Metropolis step with a Beta prior (e.g. $\text{Beta}(1, 9)$) with mean around 0.1.
- **Updating the latent positions $U$:** Via a random-walk proposal on each row of $U$.

Priors:
- $\rho \sim \text{Beta}(1, \rho_{\text{prior}})$
- $\tau \sim \text{Uniform}(0, 1)$
- $K \sim \text{Truncated-Poisson}(\lambda)$
- $\beta$ is the predetermined covariate effects

The likelihood function incorporates:
- Partial order constraints
- Noise models (queue-jump or Mallows)

## Project Structure

