# Bayesian Partial Order Inference

This project implements a Bayesian framework for inferring partial orders from observed rankings using Markov Chain Monte Carlo (MCMC) methods. The implementation includes data generation, inference, and visualization tools for analyzing partial order relationships.

## The Bayesian Partial Order 

### Partial Order

A partial order is a binary relation $\preceq$ over a set of items that satisfies:

- Reflexivity: $a \preceq a$
- Antisymmetry: if $a \preceq b$ and $b \preceq a$, then $a = b$
- Transitivity: if $a \preceq b$ and $b \preceq c$, then $a \preceq c$

### Latent Space Model

The model uses a latent space representation where:

- Each item $j$ has a K-dimensional latent position $U_j \in \mathbb{R}^K$
- The correlation between dimensions is controlled by parameter $\rho$
- The transformed latent positions $\eta_j$ are given by:
  $  \eta_j = U_j + \alpha_j$

#### Theorem(Partial Order Model)

For $\alpha$ and $\Sigma_\rho$ defined above, if we take

- $U_{j,:} \sim \mathcal{N}(0, \Sigma_\rho)$, independent for each $j \in M$,
- $\eta_{j,:} = G^{-1}\bigl(\Phi(U_{j,:})\bigr) + \alpha_j \,1_K^T$, and
- $y \sim p\bigl(\cdot \mid h(\eta(U, \beta))\bigr)$,

### MCMC Inference
The posterior distribution is given by:

$$
\pi(\rho, U, \beta \mid Y) \propto \pi(\rho)\,\pi(\beta)\,\pi(U \mid \rho)\,p\Bigl(Y \mid h\bigl(\eta(U,\beta)\bigr)\Bigr).
$$

We sample from this posterior using MCMC. Specific update steps include:

- Updating $\rho$$:Using a Beta prior (e.g., $$\text{Beta}(1, \rho_\text{prior})$$) with a mean around 0.9.
- Updating $p_{\mathrm{noise}}$: Using a Metropolis step with a Beta prior (e.g., $\text{Beta}(1, 9)$) with a mean around 0.1.
- Updating the latent positions $U$: Via a random-walk proposal given a row vector updated for each iteration.


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
├── README.md.txt
└── setup.py
```

## Features

1. **Data Generation**

   - Synthetic partial order generation
   - Configurable number of items and dimensions
   - Queue-jump noise models
2. **MCMC Inference**

   - Multiple parameter estimation
   - Convergence diagnostics
3. **Visualization**

   - Partial order graphs
   - MCMC trace plots
   - Parameter posterior distributions

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
pip install -e .
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

The package provides a command-line interface for running the analysis:

```bash
# Run with default settings
python -m po_inference

# Generate data only
python -m po_inference --generate-data

# Run inference only (with existing data)
python -m po_inference --inference-only

# Specify config files
python -m po_inference --data-config path/to/data_config.yaml --mcmc-config path/to/mcmc_config.yaml

# Set output directory
python -m po_inference --output-dir results/experiment1


```

### Python API

#### Data Generation

```python
from src.data.data_generator import generate_data
from src.utils.basic_utils import load_config

# Load configuration
config = load_config('config/data_generator_config.yaml')

# Generate synthetic data
data = generate_data(config)
```

#### MCMC Inference

```
from src.inference.po_inference import run_inference
from src.utils.basic_utils import load_config
import json

# Load configuration
config = load_config('config/mcmc_config.yaml')

# Load data
with open(config['data']['path'], 'r') as f:
    data = json.load(f)

# Run MCMC inference
results = run_inference(data, config)

```

#### Visualization

```python
from src.visualization.po_plot import POPlot

# Initialize plotter
plotter = POPlot()

# Plot partial order
plotter.plot_partial_order(matrix=h_true, title="True Partial Order")

# Plot MCMC results
plotter.plot_mcmc_inferred_variables(mcmc_results, true_param, config)
```

## Configuration

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
