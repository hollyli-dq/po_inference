# Bayesian Partial Order Inference

A Python package for Bayesian inference of strong partial orders from noisy observations using Markov Chain Monte Carlo (MCMC) methods. This implementation is based on the framework described in [Muir Watt et al. (2012)](https://doi.org/10.1214/12-AOS1029) 

## Features

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

## Mathematical Framework

### Partial Order Model

A strong partial order is a binary relation \(\prec\) over a set of items that satisfies:

- Irreflexivity: \(\neg(a \prec a)\)
- Antisymmetry: if \(a \prec b\) then \(\neg(b \prec a)\)
- Transitivity: if \(a \prec b\) and \(b \prec c\) then \(a \prec c\)

### Latent Space Model

The model uses a latent space representation where:

- Each item \(i\) has a K-dimensional latent position \(U_i \in \mathbb{R}^K\)
- The correlation between dimensions is controlled by parameter \(\rho\)
- The transformed latent positions \(\eta_i\) are given by:
  \[ \eta_i = U_i + \alpha_i \]
  where \(\alpha_i\) represents covariate effects.

The mapping from \(\eta\) to the partial order \(h\) is defined as:
\[ h_{ij} = \begin{cases}
1 & \text{if } \eta_i \prec \eta_j \\
0 & \text{otherwise}
\end{cases} \]

### MCMC Inference

The posterior distribution is sampled using MCMC with:

- Prior distributions:
  - \(\rho \sim \text{Beta}(1, \rho_{\text{prior}})\)
  - \(\tau \sim \text{Uniform}(0, 1)\)
  - \(K \sim \text{Truncated-Poisson}(\lambda)\)
  - \(\beta \sim \text{Normal}(0, \sigma^2)\) for covariate effects
- Likelihood function incorporating:
  - Partial order constraints
  - Noise models (queue-jump or Mallows)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/po_inference.git
cd po_inference
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

```
po_inference/
├── config/
│   └── mcmc_config.yaml        # Configuration for MCMC inference and data generation
├── data/
│   └── po_list_data.json      # Input data file
├── output/
│   ├── figures/
│   │   ├── mcmc_traces/      # MCMC trace plots
│   │   └── partial_orders/   # Partial order visualization plots
│   ├── results/
│   │   ├── mcmc_samples/    # MCMC samples
│   │   └── summary_stats/   # Summary statistics
│   └── logs/                # Log files
├── scripts/
│   └── run.sh               # Main execution script
├── src/
│   ├── inference/
│   │   └── po_inference.py  # Main inference module
│   ├── mcmc/
│   │   └── mcmc_simulation.py  # MCMC implementation
│   ├── utils/
│   │   ├── basic_utils.py      # Basic utility functions
│   │   ├── generation_utils.py # Data generation utilities
│   │   └── statistical_utils.py # Statistical utilities
│   └── visualization/
│       └── po_plot.py          # Plotting utilities
├── main.py                     # Main entry point
└── requirements.txt            # Python dependencies
```

## Usage

### Running the Analysis

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


## License

This project is licensed under the MIT License - see the LICENSE file for details.
