#!/bin/bash

# Run script for Bayesian Partial Order Inference
# This script sets up the environment and runs the Python CLI

# Get the directory containing this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Change to project root directory
cd "$PROJECT_ROOT"

# Set up environment variables
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Create output directories if they don't exist
mkdir -p output/{results,figures/{mcmc_traces,partial_orders},logs}

# Check if help was requested
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    echo "Bayesian Partial Order Inference"
    echo "==============================="
    echo "Usage: scripts/run_inference.sh [OPTIONS]"
    echo
    echo "For detailed help and options, run:"
    echo "  scripts/run_po_inference.py --help"
    echo
    echo "For example usage, run:"
    echo "  scripts/run_po_inference.py --examples"
    exit 0
fi

# Run the Python script
echo "Starting partial order inference..."
python3 scripts/run_po_inference.py "$@"

# Check if the run was successful
if [ $? -eq 0 ]; then
    echo "Analysis completed successfully!"
    echo "Results can be found in the output directory"
else
    echo "Error occurred during analysis"
    echo "Check logs in output/logs for details"
    exit 1
fi 