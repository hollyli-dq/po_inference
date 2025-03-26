#!/bin/bash

# Run analysis script for partial order inference

# Get the directory containing this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Change to project root directory
cd "$PROJECT_ROOT"

# Verify we're in the correct directory
if [ ! -f "main.py" ]; then
    echo "Error: Could not find main.py in the project root directory"
    echo "Current directory: $(pwd)"
    exit 1
fi

# Set up environment variables
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Create necessary directories
mkdir -p output/{figures/{mcmc_traces,partial_orders},results/{mcmc_samples,summary_stats},logs}

# Verify config files exist

if [ ! -f "config/mcmc_config.yaml" ]; then
    echo "Error: mcmc_config.yaml not found in config directory"
    exit 1
fi

# Set up logging
LOG_FILE="output/logs/run_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG_FILE")"

# Function to log messages
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Run main script with command line arguments
log_message "Starting partial order inference..."
python main.py \
    --iterations 20000 \
    --burn-in 1000 \
    --dimension 3 \
    --noise-model queue_jump \
    --output-dir output \
    "$@" 2>&1 | tee -a "$LOG_FILE"

# Check if the run was successful
if [ $? -eq 0 ]; then
    log_message "Analysis completed successfully!"
    log_message "Results can be found in the output directory"
else
    log_message "Error occurred during analysis"
    log_message "Check logs in $LOG_FILE for details"
    exit 1
fi 