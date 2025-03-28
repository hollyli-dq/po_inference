# Bayesian Partial Order Inference

A Python package for Bayesian inference of strong partial orders from noisy observations using Markov Chain Monte Carlo (MCMC) methods. This implementation is based on the framework described in [Muir Watt et al. (2012)](https://doi.org/10.1214/12-AOS1029) 

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

A strong partial order is a binary relation $\prec$ over a set of items that satisfies:

- Irreflexivity: $\neg(a \prec a)$
- Antisymmetry: if $a \prec b$ then $\neg(b \prec a)$
- Transitivity: if $a \prec b$ and $b \prec c$ then $a \prec c$


### Latent Space Model
The model uses a latent space representation where:

- Each item $j$ has a K-dimensional latent position $U_j \in \mathbb{R}^K$, where $k$ is the dimension of the latent matrix $U$
- The correlation between dimensions is controlled by parameter $\rho$
- The transformed latent positions $\eta_i$ are given by:
  $\eta_i = U_i + \alpha_i$
  where $\alpha_i$ represents covariate effects, and this is given by the $\beta_j*x_j$.

The mapping from $\eta$ to the partial order $h$ is defined as:
$h_{ij} = \begin{cases}
1 & \text{if } \eta_i \prec \eta_j \\
0 & \text{otherwise}
\end{cases}$


#### Theorem (Partial Order Model)

For $\alpha$ and $\Sigma_\rho$ defined above, if we take:

- $U_{j,:} \sim \mathcal{N}(0, \Sigma_\rho)$, independent for each $j \in M$,
- $\eta_{j,:} = G^{-1}\bigl(\Phi(U_{j,:})\bigr) + \alpha_j \,1_K^T$, and $\alpha_j=\beta_j*x_j$ 
- $y \sim p\bigl(\cdot \mid h(\eta(U, \beta))\bigr)$,


### MCMC Inference

The posterior distribution is given by:

$$
\pi(\rho, U, \beta \mid Y) \propto \pi(\rho)\,\pi(\beta)\,\pi(U \mid \rho)\,p\Bigl(Y \mid h\bigl(\eta(U,\beta)\bigr)\Bigr).
$$


We sample from this posterior using MCMC. Specific update steps include:

- Updating $\rho$: Using a Beta prior (e.g., $\text{Beta}(1, \rho_\text{prior})$) with a mean around 0.9.
- Updating $p_{\mathrm{noise}}$: Using a Metropolis step with a Beta prior (e.g., $\text{Beta}(1, 9)$) with a mean around 0.1.
- Updating the latent positions $U$: Via a random-walk proposal given a row vector updated for each iteration.

Prior distributions:
- $\rho \sim \text{Beta}(1, \rho_{\text{prior}})$
- $\tau \sim \text{Uniform}(0, 1)$
- $K \sim \text{Truncated-Poisson}(\lambda)$
- $\beta \sim \text{Normal}(0, \sigma^2)$ for covariate effects

The likelihood function incorporates:
- Partial order constraints
- Noise models (queue-jump or Mallows(not provided yet))

## Project structure
```
.
├── config/
│   └── mcmc_config.yaml
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
├── README.md.txt
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

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Test example

run main.py in the model with the given test case, or go notebook to view the example.

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

The `mcmc_config.yaml` file controls various aspects of data generation:

```yaml
data:
  path: "data/sample_data.json"
  output_dir: "output"
  generate_data: true

mcmc:
  num_iterations: 2000
  K: 3
  update_probabilities:
    rho: 0.2
    noise: 0.3
    U: 0.3
    K: 0.2

prior:
  rho_prior: 0.16667
  noise_beta_prior: 9
  mallow_ua: 10
  K_prior: 3

```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
