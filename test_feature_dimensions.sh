#!/bin/bash

# Script to test learning across different feature dimensions
# Tests dimensions: 1, 4, 16, 64, 256, 512 (4x scaling)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DIMENSIONS=(1 4 16 64 256 512)
MAX_EPOCHS=10
BATCH_SIZE=4
LEARNING_RATE=5e-3
WEIGHT_DECAY=1e-5
BETA2=0.98
LAMBDA4=5.0

# Base directories - adjust these to match your setup
AUDIO_DIR="/home/yosubs/koa_scratch/repurpose/data/audio_pann_features"
VISUAL_DIR="/home/yosubs/koa_scratch/repurpose/data/video_clip_features"
CAPTION_DIR="/home/yosubs/koa_scratch/repurpose/data/caption_features"
TRAIN_ANNOTATION="/home/yosubs/co/Repurpose/data/test.json"
VAL_ANNOTATION="/home/yosubs/co/Repurpose/data/val.json"

# Output directory
OUTPUT_BASE="dimension_tests"
mkdir -p $OUTPUT_BASE

# Summary file
SUMMARY_FILE="$OUTPUT_BASE/dimension_test_summary.csv"
echo "dimension,final_train_loss,final_val_loss,min_train_loss,min_val_loss,time_seconds,status" > $SUMMARY_FILE

# Function to extract metrics from log file
extract_metrics() {
    local log_file=$1
    local dimension=$2
    
    # Extract final and minimum losses
    final_train_loss=$(grep "train/loss_total" $log_file | tail -1 | grep -oE '[0-9]+\.[0-9]+' | head -1 || echo "N/A")
    final_val_loss=$(grep "val/loss_total" $log_file | tail -1 | grep -oE '[0-9]+\.[0-9]+' | head -1 || echo "N/A")
    min_train_loss=$(grep "train/loss_total" $log_file | grep -oE '[0-9]+\.[0-9]+' | sort -n | head -1 || echo "N/A")
    min_val_loss=$(grep "val/loss_total" $log_file | grep -oE '[0-9]+\.[0-9]+' | sort -n | head -1 || echo "N/A")
    
    # Check if learning occurred (loss decreased by at least 10%)
    if [[ "$min_train_loss" != "N/A" && "$final_train_loss" != "N/A" ]]; then
        initial_loss=$(grep "train/loss_total" $log_file | head -5 | grep -oE '[0-9]+\.[0-9]+' | head -1)
        if (( $(echo "$min_train_loss < $initial_loss * 0.9" | bc -l) )); then
            status="learning"
        else
            status="plateau"
        fi
    else
        status="failed"
    fi
    
    echo "$final_train_loss,$final_val_loss,$min_train_loss,$min_val_loss,$status"
}

# Main testing loop
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Feature Dimension Scaling Test${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Testing dimensions: ${DIMENSIONS[@]}"
echo -e "Max epochs: $MAX_EPOCHS"
echo -e "Learning rate: $LEARNING_RATE"
echo -e "Weight decay: $WEIGHT_DECAY"
echo -e "Beta2: $BETA2"
echo ""

# Track overall progress
TOTAL_DIMS=${#DIMENSIONS[@]}
CURRENT_DIM=0

for DIM in "${DIMENSIONS[@]}"; do
    CURRENT_DIM=$((CURRENT_DIM + 1))
    
    echo -e "${YELLOW}[$CURRENT_DIM/$TOTAL_DIMS] Testing dimension: $DIM${NC}"
    echo "----------------------------------------"
    
    # Set up directories for this dimension
    DIM_DIR="$OUTPUT_BASE/dim_$DIM"
    mkdir -p "$DIM_DIR/checkpoints"
    mkdir -p "$DIM_DIR/logs"
    
    # Log file for this run
    LOG_FILE="$DIM_DIR/logs/training.log"
    
    # Check if already completed
    if [ -f "$DIM_DIR/.completed" ]; then
        echo -e "${GREEN}✓ Dimension $DIM already tested, skipping...${NC}"
        continue
    fi
    
    # Start timer
    START_TIME=$(date +%s)
    
    # Export the feature dimension for Python scripts to read
    export FEATURE_DIM=$DIM
    
    echo "Running training with FEATURE_DIM=$DIM..."
    
    # Run training with minimal output to console
    python train_repurpose.py \
        --audio_dir "$AUDIO_DIR" \
        --visual_dir "$VISUAL_DIR" \
        --caption_dir "$CAPTION_DIR" \
        --train_annotation "$TRAIN_ANNOTATION" \
        --val_annotation "$VAL_ANNOTATION" \
        --batch_size $BATCH_SIZE \
        --epochs $MAX_EPOCHS \
        --learning_rate $LEARNING_RATE \
        --weight_decay $WEIGHT_DECAY \
        --beta1 0.9 \
        --beta2 $BETA2 \
        --lambda4 $LAMBDA4 \
        --gradient_clip 1.0 \
        --log_interval 10 \
        --checkpoint_dir "$DIM_DIR/checkpoints" \
        --num_workers 2 \
        --log_level WARNING \
        --use_wandb \
        --wandb_project "dimension-scaling-test" \
        --wandb_run_name "dim_${DIM}" \
        --wandb_group "dimension_test" \
        --wandb_tags "dimension_test,dim_${DIM}" \
        --early_stopping_patience 3 \
        --precision "16-mixed" \
        --enable_checkpointing \
        --limit_train_batches 50 \
        --limit_val_batches 10 \
        > "$LOG_FILE" 2>&1
    
    TRAINING_EXIT_CODE=$?
    
    # End timer
    END_TIME=$(date +%s)
    ELAPSED_TIME=$((END_TIME - START_TIME))
    
    # Check if training succeeded
    if [ $TRAINING_EXIT_CODE -eq 0 ]; then
        echo -e "${GREEN}✓ Training completed for dimension $DIM (${ELAPSED_TIME}s)${NC}"
        
        # Extract metrics and add to summary
        METRICS=$(extract_metrics "$LOG_FILE" $DIM)
        echo "$DIM,$METRICS,$ELAPSED_TIME" >> $SUMMARY_FILE
        
        # Mark as completed
        touch "$DIM_DIR/.completed"
        
        # Show quick summary
        echo "  Final train loss: $(echo $METRICS | cut -d',' -f1)"
        echo "  Min train loss: $(echo $METRICS | cut -d',' -f3)"
        echo "  Status: $(echo $METRICS | cut -d',' -f5)"
    else
        echo -e "${RED}✗ Training failed for dimension $DIM${NC}"
        echo "$DIM,N/A,N/A,N/A,N/A,$ELAPSED_TIME,failed" >> $SUMMARY_FILE
    fi
    
    # Unset the environment variable
    unset FEATURE_DIM
    
    echo ""
done

# Generate final report
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Test Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Results saved to: $SUMMARY_FILE"
echo ""
echo -e "${GREEN}Dimension Test Results:${NC}"
echo "------------------------"
column -t -s',' $SUMMARY_FILE | head -20

# Analyze trend
echo ""
echo -e "${YELLOW}Analysis:${NC}"
echo "----------"

# Find where learning stops
LAST_LEARNING_DIM=0
for DIM in "${DIMENSIONS[@]}"; do
    STATUS=$(grep "^$DIM," $SUMMARY_FILE | cut -d',' -f7)
    if [[ "$STATUS" == "learning" ]]; then
        LAST_LEARNING_DIM=$DIM
    fi
done

if [ $LAST_LEARNING_DIM -gt 0 ]; then
    echo -e "Learning detected up to dimension: ${GREEN}$LAST_LEARNING_DIM${NC}"
    NEXT_DIM=$((LAST_LEARNING_DIM * 4))
    if [ $NEXT_DIM -le 512 ]; then
        echo -e "Learning breaks between dimensions: ${YELLOW}$LAST_LEARNING_DIM and $NEXT_DIM${NC}"
    fi
else
    echo -e "${RED}No learning detected at any dimension!${NC}"
fi

echo ""
echo -e "${GREEN}Testing complete!${NC}"
echo "Full logs available in: $OUTPUT_BASE/dim_*/logs/"
echo "Checkpoints saved in: $OUTPUT_BASE/dim_*/checkpoints/"