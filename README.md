# Bayesian Partial Order Inference

A Python package for Bayesian inference of strong partial orders from noisy observations using Markov Chain Monte Carlo (MCMC) methods. This implementation is based on the framework described in Nicholls, G. K. and Muir Watt, A. (2011).

## Features

- **Bayesian inference** of the partial orders using MCMC
- **Sampling partial orders** and its total orders
- **Support for different noise models**:

  - Queue jump noise model
  - Mallows noise model
- **Visualization** of:

  - Partial orders
  - MCMC traces
  - True vs. inferred partial orders and model hyperparameters
  - Posterior parameter distribution
- **Comprehensive logging** and result storage

## The Bayesian Partial Order

### Partial Order Definition

A strong partial order is a binary relation $\,\prec\,$ over a set of items that satisfies:

- **Irreflexivity**: $\,\neg(a \prec a)\,$
- **Antisymmetry**: If $\,a \prec b\,$ then $\,\neg(b \prec a)\,$
- **Transitivity**: If $\,a \prec b\,$ and $\,b \prec c\,$ then $\,a \prec c\,$

### Theorem (Partial Order Model)

For $\alpha$ and $\Sigma_\rho$ defined above, if we take:

- $U_{j,:} \sim \mathcal{N}(0, \Sigma_\rho)$, independently for each $j \in M$,
- $\eta_{j,:} = G^{-1}\bigl(\Phi(U_{j,:})\bigr) + \alpha_j \, 1_K^T$,
- $y \sim p(⋅∣h(\eta(U,\beta)))$

The model uses a latent space representation where:

- Each item $j$ has a $K$-dimensional latent position $U_j \in \mathbb{R}^K$.
- The correlation between dimensions is controlled by parameter $\rho$.
- The transformed latent positions $\eta_i$ are given by $\eta_i = G^{-1}\bigl(\Phi(U_{j,:})\bigr) + \alpha_i$, where $\alpha_i$ represents covariate effects, e.g. $\beta_j \times x_j$.

The mapping from $\eta$ to the partial order $h$ is defined as:

$$
h_{ij} =
\begin{cases}
1 & \text{if } \eta_i \prec \eta_j,\\
0 & \text{otherwise}.
\end{cases}
$$

### MCMC Inference

The posterior distribution is given by:

$$
\pi(\rho, U, \beta \mid Y) \;\propto\; \pi(\rho)\,\pi(\beta)\,\pi(U \mid \rho)\,p\bigl(Y \mid h(\eta(U,\beta))\bigr).
$$

We sample from this posterior using MCMC. Specific update steps include:

- **Updating $\rho$**: Using a Beta prior (e.g., $\text{Beta}(1, \rho_\text{prior})$) with a mean around 0.9.
- **Updating $p_{\mathrm{noise}}$**: Using a Metropolis step with a Beta prior (e.g., $\text{Beta}(1, 9)$) with a mean around 0.1.
- **Updating the latent positions $U$**: Via a random-walk proposal, updating each row vector randomly.

**Prior distributions**:

- $\rho \sim \text{Beta}(1, \rho_{\text{prior}})$
- $\tau \sim \text{Uniform}(0, 1)$
- $K \sim \text{Truncated-Poisson}(\lambda)$
- $\beta \sim \text{Normal}(0, \sigma^2)$ for covariate effects

**The likelihood function** incorporates:

- Partial order constraints
- Noise models (queue-jump or Mallows)

## Project structure

```
.
├── config/
│   └── mcmc_config.yaml
│   └── data_generator_config.yaml
├── data/
├── notebook/
│   └── mcmc_simulation.ipynb
├── src/
│   ├── data/
│   │   └── data_generator.py
│   ├── mcmc/
│   │   ├── mcmc_simulation.py
│   │   └── likelihood_cache.py
│   ├── utils/
│   │   ├── basic_utils.py
│   │   ├── statistical_utils.py
│   │   └── generation_utils.py
│   └── visualization/
│       └── po_plot.py
├── requirements.txt
├── README.md
└── setup.py
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/hollyli-dq/po_inference.git
cd po_inference
```

2. Create and Activate a Virtual Environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Test example

Run main.py in the model with the given test case, or go to notebook to view the example.

```bash
# Run with default settings
sh scripts/run.sh 
```

or

```bash
# Run with default settings
python scripts/main.py 
```

### Command Line Interface

The main script can be run using the provided shell script:

```bash
bash scripts/run.sh
```

This will execute the analysis with default parameters:

- 20,000 MCMC iterations
- 1,000 burn-in iterations
- 3-dimensional partial order
- Queue jump noise model

You can override these parameters by passing additional arguments:

```bash
bash scripts/run.sh --iterations 50000 --burn-in 2000 --dimension 4
```

### Configuration

The analysis is configured through `config/mcmc_config.yaml`, which contains:

- MCMC parameters (iterations, burn-in, thinning)
- Prior distributions
- Visualization settings
- Data generation parameters (if generating synthetic data)

### Output

The analysis generates several outputs:

1. **Results Files**:

   - `output/results/mcmc_samples/{data_name}_results.json`: MCMC samples and summary statistics
   - `output/results/mcmc_samples/{data_name}_partial_order.npy`: Inferred partial order matrix
2. **Visualizations**:

   - `output/figures/mcmc_traces/{data_name}_mcmc_plots.pdf`: MCMC trace plots
   - `output/figures/partial_orders/{data_name}_inferred_po.pdf`: Inferred partial order visualization
   - `output/figures/partial_orders/{data_name}_true_po.pdf`: True partial order visualization (if available)
3. **Logs**:

   - `output/logs/run_{timestamp}.log`: Detailed execution log

## Dependencies

- Python 3.8+
- NumPy
- Matplotlib
- PyYAML
- NetworkX (for graph operations)

## References

* Nicholls, G. K., Lee, J. E., Karn, N., Johnson, D., Huang, R., & Muir-Watt, A. (2024). [Bayesian Inference for Partial Orders from Random Linear Extensions: Power Relations from 12th Century Royal Acta](https://doi.org/10.48550/arXiv.2212.05524)*
* Chuxuan, Jiang, C., Nicholls, G. K., & Lee, J. E. (2023). [Bayesian Inference for Vertex-Series-Parallel Partial Orders](http://arxiv.org/abs/2306.15827).
* Nicholls, G. K. and Muir Watt, A. (2011). **Partial Order Models for Episcopal Social Status in 12th Century England.** *Proceedings of the 26th International Workshop on Statistical Modelling (Valencia, Spain), July 5–11, 2011*, pp. 437–440.
