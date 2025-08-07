#!/bin/bash

# Script to run the hint injection experiment
# This will inject red dots into highlight frames and train the model

echo "=== Running Hint Injection Experiment ==="
echo "This experiment injects obvious visual hints (red dots) into highlight frames"
echo "to validate the preprocessing pipeline and model architecture."
echo ""

# Configuration
VIDEO_DIR="/home/yosubs/koa_scratch/repurpose/raw_videos"
TRAIN_ANNOTATION="/home/yosubs/co/Repurpose/data/test.json"
VAL_ANNOTATION="/home/yosubs/co/Repurpose/data/val.json"

# Output directories for hint features
VISUAL_HINT_DIR="/home/yosubs/koa_scratch/repurpose/data/video_clip_features_hint"

# Extract visual features with red dot hints
echo "Extracting visual features with red dot hints..."
echo "Output directory: $VISUAL_HINT_DIR"

python preprocessing/visual_feature_extractor_clip.py \
    --video-dir "$VIDEO_DIR" \
    --dataset "$TRAIN_ANNOTATION" \
    --output-dir "$VISUAL_HINT_DIR" \
    --inject-hints \
    --log-level INFO

# Extract visual features with red dot hints for validation set
echo "Extracting visual features with red dot hints for validation set..."
echo "Output directory: $VISUAL_HINT_DIR"

python preprocessing/visual_feature_extractor_clip.py \
    --video-dir "$VIDEO_DIR" \
    --dataset "$VAL_ANNOTATION" \
    --output-dir "$VISUAL_HINT_DIR" \
    --inject-hints \
    --log-level INFO

# Check if extraction was successful
if [ $? -ne 0 ]; then
    echo "Error: Visual feature extraction failed"
    exit 1
fi
