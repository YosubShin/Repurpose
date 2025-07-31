#!/bin/bash

# Script to run RepurposeModel training with proper logging and monitoring

# Set paths based on the notebook configuration
AUDIO_DIR="/home/yosubs/koa_scratch/repurpose/data/audio_pann_features"
VISUAL_DIR="/home/yosubs/koa_scratch/repurpose/data/video_clip_features"
CAPTION_DIR="/home/yosubs/koa_scratch/repurpose/data/caption_features"
TRAIN_ANNOTATION="/home/yosubs/co/Repurpose/data/test.json"
VAL_ANNOTATION="/home/yosubs/co/Repurpose/data/val.json"

# Create necessary directories
mkdir -p checkpoints
mkdir -p logs

# Run training with comprehensive logging
python train_repurpose.py \
    --audio_dir "$AUDIO_DIR" \
    --visual_dir "$VISUAL_DIR" \
    --caption_dir "$CAPTION_DIR" \
    --train_annotation "$TRAIN_ANNOTATION" \
    --val_annotation "$VAL_ANNOTATION" \
    --batch_size 48 \
    --epochs 20 \
    --learning_rate 5e-4 \
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
    --num_viz_samples 10 \
    --log_level INFO \
    --use_wandb \
    --wandb_project "repurpose-experiments" \
    --early_stopping_patience 5 \
    --gradient_clip 0.1 \
    --precision "16-mixed" \
    --enable_checkpointing \
    2>&1 | tee logs/training_$(date +%Y%m%d_%H%M%S).log

echo "Training completed. Check logs/ directory for detailed output."