#!/bin/bash

# Run script for Bayesian Partial Order Inference

# Get the directory containing this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to project root directory
cd "$PROJECT_ROOT"

# Set up environment variables
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Create necessary directories
mkdir -p output/{figures/{mcmc_traces,partial_orders},results,logs}

# Parse command-line arguments
GENERATE_DATA=false
INFERENCE_ONLY=false
VERBOSE=false
OUTPUT_DIR="output"
DATA_CONFIG="config/data_generator_config.yaml"
MCMC_CONFIG="config/mcmc_config.yaml"
ITERATIONS=""

# Parse command-line options
while [[ $# -gt 0 ]]; do
  case $1 in
    --generate-data)
      GENERATE_DATA=true
      shift
      ;;
    --inference-only)
      INFERENCE_ONLY=true
      shift
      ;;
    --verbose)
      VERBOSE=true
      shift
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --data-config)
      DATA_CONFIG="$2"
      shift 2
      ;;
    --mcmc-config)
      MCMC_CONFIG="$2"
      shift 2
      ;;
    --iterations)
      ITERATIONS="$2"
      shift 2
      ;;
    --plot-only)
      PLOT_ONLY=true
      shift
      ;;
    --data-file)
      DATA_FILE="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Build the command
CMD="python -m src.cli"

if $GENERATE_DATA; then
  CMD="$CMD --generate-data"
fi

if $INFERENCE_ONLY; then
  CMD="$CMD --inference-only"
fi

if $VERBOSE; then
  CMD="$CMD --verbose"
fi

CMD="$CMD --output-dir $OUTPUT_DIR --data-config $DATA_CONFIG --mcmc-config $MCMC_CONFIG"

if [ -n "$ITERATIONS" ]; then
  CMD="$CMD --iterations $ITERATIONS"
fi

if $PLOT_ONLY; then
  CMD="$CMD --plot-only"
fi

if [ -n "$DATA_FILE" ]; then
  CMD="$CMD --data-file $DATA_FILE"
fi

# Run the command
echo "Starting partial order inference..."
echo "Command: $CMD"
eval $CMD

# Check if the run was successful
if [ $? -eq 0 ]; then
  echo "Analysis completed successfully!"
  echo "Results can be found in the $OUTPUT_DIR directory"
  exit 0
else
  echo "Error occurred during analysis"
  echo "Check logs for details"
  exit 1
fi 