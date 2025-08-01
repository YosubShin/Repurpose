#!/bin/bash

# Script to run RepurposeModel training with proper logging and monitoring
# Usage: ./run_training.sh [--resume]
#   --resume: Resume from the latest checkpoint if available

# Set paths based on the notebook configuration
AUDIO_DIR="/home/yosubs/koa_scratch/repurpose/data/audio_pann_features"
VISUAL_DIR="/home/yosubs/koa_scratch/repurpose/data/video_clip_features"
CAPTION_DIR="/home/yosubs/koa_scratch/repurpose/data/caption_features"
TRAIN_ANNOTATION="/home/yosubs/co/Repurpose/data/test.json"
VAL_ANNOTATION="/home/yosubs/co/Repurpose/data/val.json"

# Create necessary directories
mkdir -p checkpoints
mkdir -p logs

# Parse command line arguments
RESUME_TRAINING=false
for arg in "$@"; do
    case $arg in
        --resume)
            RESUME_TRAINING=true
            shift
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: $0 [--resume]"
            exit 1
            ;;
    esac
done

# Check for existing checkpoint to resume from (only if --resume flag is provided)
RESUME_ARG=""
if [ "$RESUME_TRAINING" = true ]; then
    if ls checkpoints/last*.ckpt 1> /dev/null 2>&1; then
        LATEST_CKPT=$(ls -t checkpoints/last*.ckpt | head -1)
        echo "Found existing checkpoint: $LATEST_CKPT"
        echo "Training will resume from this checkpoint"
        RESUME_ARG="--resume_from_checkpoint $LATEST_CKPT"
    else
        echo "WARNING: --resume flag provided but no checkpoint found, starting fresh training"
    fi
else
    if ls checkpoints/last*.ckpt 1> /dev/null 2>&1; then
        echo "NOTE: Existing checkpoints found but --resume flag not provided, starting fresh training"
        echo "      Use './run_training.sh --resume' to resume from the latest checkpoint"
    else
        echo "Starting fresh training"
    fi
fi

# Run training with comprehensive logging
python train_repurpose.py \
    --audio_dir "$AUDIO_DIR" \
    --visual_dir "$VISUAL_DIR" \
    --caption_dir "$CAPTION_DIR" \
    --train_annotation "$TRAIN_ANNOTATION" \
    --val_annotation "$VAL_ANNOTATION" \
    --batch_size 12 \
    --epochs 20 \
    --learning_rate 1e-4 \
    --d_model 128 \
    --n_head 4 \
    --n_layers 2 \
    --lambda1 0.1 \
    --lambda2 0.3 \
    --lambda3 0.1 \
    --log_interval 5 \
    --checkpoint_dir checkpoints \
    --num_workers 0 \
    --create_visualizations \
    --num_viz_samples 5 \
    --log_level INFO \
    --use_wandb \
    --wandb_project "repurpose-experiments" \
    --early_stopping_patience 0 \
    --gradient_clip 0.1 \
    --precision "16-mixed" \
    --enable_checkpointing \
    $RESUME_ARG \
    2>&1 | tee logs/training_$(date +%Y%m%d_%H%M%S).log

echo "Training completed. Check logs/ directory for detailed output."