"""
Test module for MCMC inference.
"""

import pytest
import numpy as np
from src.inference.mcmc import MCMCInference

@pytest.fixture
def config():
    """Create a test configuration."""
    return {
        'mcmc': {
            'n_iterations': 100,
            'burn_in': 10,
            'thinning': 5,
            'priors': {
                'rho': {'type': 'beta', 'params': [1, 1]},
                'noise': {'type': 'beta', 'params': [1, 1]},
                'tau': {'type': 'beta', 'params': [1, 1]},
                'theta': {'type': 'gamma', 'params': [1, 1]}
            }
        }
    }

@pytest.fixture
def data():
    """Create test data."""
    n_items = 3
    return {
        'n_items': n_items,
        'items': list(range(n_items)),
        'true_partial_order': {
            'matrix': np.array([[0, 1, 1],
                              [0, 0, 1],
                              [0, 0, 0]])
        },
        'assessors': [
            {
                'id': 1,
                'items': [0, 1, 2],
                'complete_rankings': [
                    np.array([0, 1, 2])
                ],
                'partial_rankings': [
                    [np.array([0, 1]), np.array([2])]
                ]
            }
        ]
    }

def test_mcmc_initialization(config):
    """Test MCMC initialization."""
    mcmc = MCMCInference(config)
    assert mcmc.config == config
    assert mcmc.mcmc_config == config['mcmc']
    assert mcmc.priors == config['mcmc']['priors']

def test_mcmc_state_initialization(config, data):
    """Test MCMC state initialization."""
    mcmc = MCMCInference(config)
    state = mcmc._initialize_state(data)
    
    assert isinstance(state, dict)
    assert 'partial_order' in state
    assert 'rho' in state
    assert 'noise' in state
    assert 'tau' in state
    assert 'theta' in state
    
    assert state['partial_order'].shape == (data['n_items'], data['n_items'])
    assert 0 <= state['rho'] <= 1
    assert 0 <= state['noise'] <= 1
    assert 0 <= state['tau'] <= 1
    assert state['theta'] > 0

def test_parameter_updates(config, data):
    """Test parameter updates."""
    mcmc = MCMCInference(config)
    state = mcmc._initialize_state(data)
    
    # Test rho update
    new_rho = mcmc._update_rho(state, data)
    assert 0 <= new_rho <= 1
    
    # Test noise update
    new_noise = mcmc._update_noise(state, data)
    assert 0 <= new_noise <= 1
    
    # Test tau update
    new_tau = mcmc._update_tau(state, data)
    assert 0 <= new_tau <= 1
    
    # Test theta update
    new_theta = mcmc._update_theta(state, data)
    assert new_theta > 0
    
    # Test partial order update
    new_po = mcmc._update_partial_order(state, data)
    assert new_po.shape == (data['n_items'], data['n_items'])
    assert np.all((new_po == 0) | (new_po == 1))

def test_transitivity_ensuring():
    """Test transitivity ensuring."""
    mcmc = MCMCInference({})
    matrix = np.array([[0, 1, 0],
                      [0, 0, 1],
                      [0, 0, 0]])
    
    transitive_matrix = mcmc._ensure_transitivity(matrix)
    assert transitive_matrix[0, 2] == 1  # Should be transitive

def test_acceptance_ratio(config, data):
    """Test acceptance ratio computation."""
    mcmc = MCMCInference(config)
    state = mcmc._initialize_state(data)
    
    current = 0.5
    proposal = 0.6
    
    ratio = mcmc._compute_acceptance_ratio(current, proposal, state, data)
    assert isinstance(ratio, float)
    assert ratio >= 0

def test_result_storage(config, data):
    """Test result storage."""
    mcmc = MCMCInference(config)
    results = {'traces': {}}
    state = mcmc._initialize_state(data)
    
    mcmc._store_results(results, state)
    
    for param in state:
        assert param in results['traces']
        assert len(results['traces'][param]) == 1

def test_final_estimate(config, data):
    """Test final estimate computation."""
    mcmc = MCMCInference(config)
    results = {
        'traces': {
            'partial_order': [
                np.array([[0, 1, 1],
                         [0, 0, 1],
                         [0, 0, 0]]),
                np.array([[0, 1, 1],
                         [0, 0, 1],
                         [0, 0, 0]])
            ]
        }
    }
    
    final_matrix = mcmc._compute_final_estimate(results)
    assert final_matrix.shape == (data['n_items'], data['n_items'])
    assert np.all((final_matrix == 0) | (final_matrix == 1))
    assert mcmc._ensure_transitivity(final_matrix).all() == final_matrix.all() 