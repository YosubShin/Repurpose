#!/bin/bash
#SBATCH --job-name=repurpose_train_multigpu
#SBATCH --output=/home/yosubs/koa_scratch/repurpose/outputs/training-%j.out
#SBATCH --error=/home/yosubs/koa_scratch/repurpose/outputs/training-%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-shared
#SBATCH --gres=gpu:2          # Request 2 GPUs for multi-GPU training
#SBATCH --cpus-per-task=8     # More CPUs for multi-GPU
#SBATCH --mem=64GB            # More memory for multi-GPU
#SBATCH --nodes=1             # Single node (multi-GPU on same node)
#SBATCH --ntasks=1            # Single task for DataParallel
#SBATCH --ntasks-per-node=1   # One task per node

# Script to run Repurpose multi-GPU training
# Usage: sbatch slurm_multi_gpu_training.sh [config_file] [strategy]

CONFIG_FILE="${1:-configs/Repurpose.yaml}"
STRATEGY="${2:-auto}"  # auto, dp, ddp, single

echo "Starting Repurpose multi-GPU training job"
echo "Config file: $CONFIG_FILE"
echo "Strategy: $STRATEGY"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: ${SLURM_MEM_PER_NODE}MB"
echo "Start time: $(date)"

# Load required modules
echo "Loading modules..."
module load lang/Anaconda3
module load vis/FFmpeg

# Activate the conda environment
echo "Activating conda environment..."
source activate repurpose
export PATH="/home/yosubs/.conda/envs/repurpose/bin:$PATH"

# Set up cuDNN libraries
if [ -d "$CONDA_PREFIX/lib/python3.9/site-packages/nvidia/cudnn/lib" ]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.9/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
    echo "cuDNN library path added to LD_LIBRARY_PATH"
fi

# Environment verification
echo "=== ENVIRONMENT VERIFICATION ==="
echo "Python: $(which python) ($(python --version))"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"

# Check GPU setup
echo "=== GPU INFORMATION ==="
nvidia-smi

# Set environment variables for optimal performance
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

# For distributed training (optional)
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# Create output directory
mkdir -p /home/yosubs/koa_scratch/repurpose/outputs

# Change to working directory
cd /home/yosubs/co/Repurpose

# Update config with strategy if specified
if [ "$STRATEGY" != "auto" ]; then
    echo "Overriding config strategy to: $STRATEGY"
    # For simplicity, we'll pass strategy as environment variable
    export REPURPOSE_STRATEGY="$STRATEGY"
fi

# Run GPU detection and analysis first
echo "=== RUNNING GPU ANALYSIS ==="
python detect_gpu_setup.py --output-json gpu_analysis.json

# Test multi-GPU setup
echo "=== TESTING MULTI-GPU SETUP ==="
python test_multi_gpu.py --config-path "$CONFIG_FILE" --strategy "$STRATEGY" --verbose

# Check if test passed
TEST_EXIT_CODE=$?
if [ $TEST_EXIT_CODE -ne 0 ]; then
    echo "❌ Multi-GPU test failed. Aborting training."
    exit $TEST_EXIT_CODE
fi

echo "✅ Multi-GPU test passed. Starting training..."

# Determine launch command based on strategy and available GPUs
GPU_COUNT=$(python -c 'import torch; print(torch.cuda.device_count())')

if [ "$STRATEGY" = "ddp" ] && [ "$GPU_COUNT" -gt 1 ]; then
    echo "=== LAUNCHING DISTRIBUTED DATA PARALLEL TRAINING ==="
    # For DDP, we need to modify SLURM config or use torchrun
    # Update SLURM header for DDP:
    # #SBATCH --ntasks=2
    # #SBATCH --ntasks-per-node=2
    
    # Use torchrun for single-node multi-GPU DDP
    python -m torch.distributed.launch \
        --nproc_per_node=$GPU_COUNT \
        --use_env \
        main.py \
        --config_path "$CONFIG_FILE" \
        --log-level INFO
    
elif [ "$STRATEGY" = "dp" ] || ([ "$STRATEGY" = "auto" ] && [ "$GPU_COUNT" -gt 1 ]); then
    echo "=== LAUNCHING DATA PARALLEL TRAINING ==="
    python main.py \
        --config_path "$CONFIG_FILE" \
        --log-level INFO
        
else
    echo "=== LAUNCHING SINGLE GPU TRAINING ==="
    python main.py \
        --config_path "$CONFIG_FILE" \
        --log-level INFO
fi

# Check training exit status
TRAINING_EXIT_CODE=$?
END_TIME=$(date)

echo "=== TRAINING COMPLETION SUMMARY ==="
echo "End time: $END_TIME"
echo "Job duration: $SECONDS seconds"
echo "Exit code: $TRAINING_EXIT_CODE"

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully!"
    
    # Log success
    echo "SUCCESS" > "/home/yosubs/koa_scratch/repurpose/outputs/training_${SLURM_JOB_ID}_status.txt"
    echo "Completed at: $END_TIME" >> "/home/yosubs/koa_scratch/repurpose/outputs/training_${SLURM_JOB_ID}_status.txt"
    
else
    echo "❌ Training failed with exit code: $TRAINING_EXIT_CODE"
    
    # Log failure
    echo "FAILED: $TRAINING_EXIT_CODE" > "/home/yosubs/koa_scratch/repurpose/outputs/training_${SLURM_JOB_ID}_status.txt"
    echo "Failed at: $END_TIME" >> "/home/yosubs/koa_scratch/repurpose/outputs/training_${SLURM_JOB_ID}_status.txt"
fi

# Show final resource usage
echo "=== FINAL RESOURCE USAGE ==="
df -h /home/yosubs/koa_scratch/repurpose/
nvidia-smi

echo "Job finished at: $END_TIME"
exit $TRAINING_EXIT_CODE