#!/bin/bash
#SBATCH --job-name=repurpose_preprocess
#SBATCH --output=/home/yosubs/koa_scratch/repurpose/outputs/slurm-%j.out
#SBATCH --error=/home/yosubs/koa_scratch/repurpose/outputs/slurm-%j.err
#SBATCH --time=8:00:00
#SBATCH --partition=kill-shared
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --nodes=1

# Script to run Repurpose preprocessing on a dataset chunk using GPU

# Check if input file is provided
if [ $# -eq 0 ]; then
    echo "Error: No input file provided"
    echo "Usage: sbatch slurm_preprocessing_job.sh <chunk_file.json> [steps]"
    echo "  chunk_file.json: Dataset chunk to process"
    echo "  steps: Optional processing steps (default: download visual audio text)"
    exit 1
fi

CHUNK_FILE="$1"
STEPS="${2:-download visual audio text}"

# Check if chunk file exists
if [ ! -f "$CHUNK_FILE" ]; then
    echo "Error: Chunk file not found: $CHUNK_FILE"
    exit 1
fi

echo "Starting Repurpose preprocessing job for: $CHUNK_FILE"
echo "Processing steps: $STEPS"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: ${SLURM_MEM_PER_NODE}MB"
echo "Time: $(date)"

# Load required modules
echo "Loading modules..."
module load lang/Anaconda3
module load vis/FFmpeg

# Activate the conda environment
echo "Activating conda environment..."
source activate repurpose

# Verify GPU is available
echo "Checking GPU availability..."
nvidia-smi

# Verify dependencies
echo "Checking Python dependencies..."
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import clip; print('CLIP: OK')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"

# Set environment variables for optimal performance
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

# Change to the working directory
cd /home/yosubs/co/Repurpose/preprocessing

# Create output directory if it doesn't exist
mkdir -p /home/yosubs/koa_scratch/repurpose/outputs

# Get chunk name for logging
CHUNK_NAME=$(basename "$CHUNK_FILE" .json)
echo "Processing chunk: $CHUNK_NAME"

# Run the preprocessing pipeline
echo "Running preprocessing pipeline..."
python main_preprocessing.py \
    --config preprocessing_config.yaml \
    --dataset "$CHUNK_FILE" \
    --steps $STEPS \
    --log-level INFO

# Check exit status
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Preprocessing job completed successfully for $CHUNK_NAME"
    
    # Create success marker
    touch "/home/yosubs/koa_scratch/repurpose/outputs/${CHUNK_NAME}_SUCCESS"
    
    # Log completion details
    echo "=== COMPLETION SUMMARY ==="
    echo "Chunk: $CHUNK_NAME"
    echo "Steps processed: $STEPS"
    echo "Completed at: $(date)"
    echo "Job duration: $SECONDS seconds"
    
else
    echo "✗ Preprocessing job failed for $CHUNK_NAME with exit code: $EXIT_CODE"
    
    # Create failure marker
    touch "/home/yosubs/koa_scratch/repurpose/outputs/${CHUNK_NAME}_FAILED"
fi

# Show disk usage
echo ""
echo "=== DISK USAGE ==="
df -h /home/yosubs/koa_scratch/repurpose/

echo "Job finished at: $(date)"
exit $EXIT_CODE