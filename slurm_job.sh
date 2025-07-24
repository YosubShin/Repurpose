#!/bin/bash
#SBATCH --job-name=repurpose_ddp      # Job name
#SBATCH --output=logs/slurm_%j.out    # Standard output log
#SBATCH --error=logs/slurm_%j.err     # Standard error log
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=2           # Number of tasks per node (= number of GPUs)
#SBATCH --cpus-per-task=4             # Number of CPU cores per task
#SBATCH --gres=gpu:2                  # Number of GPUs per node
#SBATCH --mem=32G                     # Memory per node
#SBATCH --time=24:00:00               # Time limit (hours:minutes:seconds)
#SBATCH --partition=gpu               # Partition name (adjust for your cluster)

# Uncomment and adjust these if your cluster requires specific constraints
# #SBATCH --constraint=v100            # GPU type constraint
# #SBATCH --account=your_account       # Account name
# #SBATCH --qos=normal                 # Quality of service

# Print job information
echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node List: $SLURM_JOB_NODELIST"
echo "Number of Nodes: $SLURM_JOB_NUM_NODES"
echo "Number of Tasks: $SLURM_NTASKS"
echo "Tasks per Node: $SLURM_NTASKS_PER_NODE"
echo "CPUs per Task: $SLURM_CPUS_PER_TASK"
echo "Memory per Node: $SLURM_MEM_PER_NODE"
echo "Working Directory: $SLURM_SUBMIT_DIR"
echo "=========================================="

# Environment setup
echo "Setting up environment..."

# Load modules (adjust for your cluster)
# module load python/3.9
# module load cuda/11.8
# module load gcc/9.3.0

# Activate conda environment
# source activate repurpose  # or: conda activate repurpose

# Or activate virtual environment
# source venv/bin/activate

# Set environment variables for better performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=0,1  # Adjust based on your GPU allocation

# Set PyTorch specific environment variables
export TORCH_DISTRIBUTED_DEBUG=INFO  # Enable for debugging
export NCCL_DEBUG=INFO               # Enable for NCCL debugging (use WARN for less verbose)
export NCCL_IB_DISABLE=1             # Disable InfiniBand if causing issues
export NCCL_P2P_DISABLE=1            # Disable P2P if causing issues

# Print environment information
echo "=========================================="
echo "Environment Information"
echo "=========================================="
python --version
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA devices: {torch.cuda.device_count()}')"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# Change to job directory
cd $SLURM_SUBMIT_DIR

# Create logs directory if it doesn't exist
mkdir -p logs

# Configuration
CONFIG_PATH="configs/Repurpose.yaml"
LOG_LEVEL="INFO"

# Verify config file exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file $CONFIG_PATH not found"
    exit 1
fi

# Print job start time
echo "Job started at: $(date)"

# Launch distributed training
echo "=========================================="
echo "Launching Distributed Training"
echo "=========================================="

# Method 1: Use srun with our main script directly
# This is the recommended approach for SLURM
srun python main.py \
    --config_path "$CONFIG_PATH" \
    --log-level "$LOG_LEVEL"

# Method 2: Alternative - use the launch script
# ./run_ddp.sh -n $SLURM_NTASKS_PER_NODE -c "$CONFIG_PATH" -l "$LOG_LEVEL"

# Print job end time
echo "Job finished at: $(date)"

# Print some final statistics
echo "=========================================="
echo "Job Statistics"
echo "=========================================="
sacct -j $SLURM_JOB_ID --format=JobID,JobName,State,Time,CPUTime,MaxRSS,MaxVMSize