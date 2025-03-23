# Bayesian Partial Order Inference

This project implements a Bayesian framework for inferring partial orders from observed rankings using Markov Chain Monte Carlo (MCMC) methods. The implementation includes data generation, inference, and visualization tools for analyzing partial order relationships.

## Mathematical Framework

### Partial Order Model

A partial order is a binary relation $\preceq$ over a set of items that satisfies:

- Reflexivity: $a \preceq a$
- Antisymmetry: if $a \preceq b$ and $b \preceq a$, then $a = b$
- Transitivity: if $a \preceq b$ and $b \preceq c$, then $a \preceq c$

### Latent Space Model

The model uses a latent space representation where:

- Each item $j$ has a K-dimensional latent position $U_j \in \mathbb{R}^K$
- The correlation between dimensions is controlled by parameter $\rho$
- The transformed latent positions $\eta_j$ are given by:
  $$
  \eta_j = U_j + \alpha_j
  $$

where $\alpha_i$ represents covariate effects.

### MCMC Inference

The posterior distribution is sampled using MCMC with:

- Prior distributions:
  - $\rho \sim \text{Beta}(1, \rho_\text{prior})$
  - $\tau \sim \text{Uniform}(0, 1)$
  - $K \sim \text{Truncated-Poisson}(\lambda)$
- The poesterior function is:
  $$
  π(ρ,U,β∣Y)∝π(ρ)π(β)π(U∣ρ)p(Y∣h(η(U,β)))
  $$

```
.
├── config/
│   └── mcmc_config.yaml
├── data/
│   └── sample_data.yaml
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
git clone https://github.com/yourusername/po_inference.git
cd po_inference
```

2. Install dependencies:

```bash
pip install -e .
```

## Usage

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

The `data_generator_config.yaml` file controls various aspects of data generation:

```yaml
generation:
  n: 10  # Number of items
  N: 100  # Number of observations
  K: 2   # Number of dimensions

prior:
  rho_prior: 1.0
  noise_beta_prior: 1.0
  K_prior: 1.0

noise:
  noise_option: "queue_jump"  # or "mallows_noise"
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
