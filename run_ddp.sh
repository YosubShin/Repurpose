#!/bin/bash

# Distributed training launcher script for HPC environments
# This script provides several ways to launch distributed training

set -e

# Default values
NPROC_PER_NODE=2
CONFIG_PATH="configs/Repurpose.yaml"
LOG_LEVEL="INFO"
PYTHON_PATH=""

# Print usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    -n, --nproc-per-node    Number of processes per node (default: 2)
    -c, --config            Path to config file (default: configs/Repurpose.yaml)
    -l, --log-level         Log level (default: INFO)
    -p, --python-path       Python executable path (default: python)
    -h, --help              Show this help message

Examples:
    # Run with 2 GPUs on single node
    $0 -n 2

    # Run with custom config
    $0 -n 4 -c configs/custom.yaml

    # Run with debug logging
    $0 -n 2 -l DEBUG

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--nproc-per-node)
            NPROC_PER_NODE="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        -l|--log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        -p|--python-path)
            PYTHON_PATH="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            usage
            exit 1
            ;;
    esac
done

# Determine Python executable
if [ -z "$PYTHON_PATH" ]; then
    if command -v python &> /dev/null; then
        PYTHON_PATH=python
    elif command -v python3 &> /dev/null; then
        PYTHON_PATH=python3
    else
        echo "Error: Python not found. Please specify python path with -p"
        exit 1
    fi
fi

echo "=========================================="
echo "Distributed Training Launcher"
echo "=========================================="
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "CONFIG_PATH: $CONFIG_PATH"
echo "LOG_LEVEL: $LOG_LEVEL"
echo "PYTHON_PATH: $PYTHON_PATH"
echo "=========================================="

# Check if config file exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file $CONFIG_PATH not found"
    exit 1
fi

# Check if running in SLURM environment
if [ ! -z "$SLURM_JOB_ID" ]; then
    echo "SLURM environment detected"
    echo "SLURM_JOB_ID: $SLURM_JOB_ID"
    echo "SLURM_NTASKS: $SLURM_NTASKS"
    echo "SLURM_PROCID: $SLURM_PROCID"
    
    # In SLURM, use srun to launch the job
    exec srun $PYTHON_PATH main.py \
        --config_path "$CONFIG_PATH" \
        --log-level "$LOG_LEVEL"
        
elif command -v torchrun &> /dev/null; then
    echo "Using torchrun (recommended)"
    
    # Use torchrun (recommended for PyTorch 1.10+)
    exec torchrun \
        --nproc_per_node="$NPROC_PER_NODE" \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=localhost \
        --master_port=29500 \
        main.py \
        --config_path "$CONFIG_PATH" \
        --log-level "$LOG_LEVEL"
        
else
    echo "Using torch.distributed.launch (deprecated but available)"
    
    # Fallback to torch.distributed.launch
    exec $PYTHON_PATH -m torch.distributed.launch \
        --nproc_per_node="$NPROC_PER_NODE" \
        --use_env \
        main.py \
        --config_path "$CONFIG_PATH" \
        --log-level "$LOG_LEVEL"
fi