#!/bin/bash

# Script to test different feature strategies with increasing dimensions
# Compares: repeat_label vs real_with_signal approaches

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DIMENSIONS=(4 16 64 256 512)
MAX_EPOCHS=10
BATCH_SIZE=4
LEARNING_RATE=5e-3
WEIGHT_DECAY=1e-5
BETA2=0.98
LAMBDA4=5.0

# Strategies to test
STRATEGIES=("repeat_label" "real_with_signal")
SIGNAL_MODES=("additive" "multiplicative" "partial")
SIGNAL_STRENGTHS=(0.5 1.0 2.0)

# Base directories - adjust these to match your setup
AUDIO_DIR="/home/yosubs/koa_scratch/repurpose/data/audio_pann_features"
VISUAL_DIR="/home/yosubs/koa_scratch/repurpose/data/video_clip_features"
CAPTION_DIR="/home/yosubs/koa_scratch/repurpose/data/caption_features"
TRAIN_ANNOTATION="/home/yosubs/co/Repurpose/data/test.json"
VAL_ANNOTATION="/home/yosubs/co/Repurpose/data/val.json"

# Output directory
OUTPUT_BASE="feature_strategy_tests"
mkdir -p $OUTPUT_BASE

# Summary file
SUMMARY_FILE="$OUTPUT_BASE/strategy_comparison.csv"
echo "strategy,signal_mode,signal_strength,dimension,final_train_loss,min_train_loss,status" > $SUMMARY_FILE

# Function to run a single test
run_test() {
    local STRATEGY=$1
    local DIM=$2
    local SIGNAL_MODE=${3:-"additive"}
    local SIGNAL_STRENGTH=${4:-"1.0"}
    
    # Create descriptive name
    if [[ "$STRATEGY" == "real_with_signal" ]]; then
        TEST_NAME="${STRATEGY}_${SIGNAL_MODE}_str${SIGNAL_STRENGTH}_dim${DIM}"
    else
        TEST_NAME="${STRATEGY}_dim${DIM}"
    fi
    
    echo -e "${YELLOW}Testing: $TEST_NAME${NC}"
    
    # Set up directories
    TEST_DIR="$OUTPUT_BASE/$TEST_NAME"
    mkdir -p "$TEST_DIR/checkpoints"
    mkdir -p "$TEST_DIR/logs"
    
    LOG_FILE="$TEST_DIR/logs/training.log"
    
    # Check if already completed
    if [ -f "$TEST_DIR/.completed" ]; then
        echo -e "${GREEN}✓ Test $TEST_NAME already completed, skipping...${NC}"
        return
    fi
    
    # Export environment variables
    export FEATURE_DIM=$DIM
    export FEATURE_STRATEGY=$STRATEGY
    export SIGNAL_MODE=$SIGNAL_MODE
    export SIGNAL_STRENGTH=$SIGNAL_STRENGTH
    export ADD_NOISE="true"
    export NOISE_SCALE="0.01"
    
    # Run training
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
        --checkpoint_dir "$TEST_DIR/checkpoints" \
        --num_workers 2 \
        --log_level WARNING \
        --use_wandb \
        --wandb_project "feature-strategy-test" \
        --wandb_run_name "$TEST_NAME" \
        --wandb_group "strategy_comparison" \
        --wandb_tags "strategy_test,$STRATEGY,dim_${DIM}" \
        --early_stopping_patience 3 \
        --precision "16-mixed" \
        --limit_train_batches 50 \
        --limit_val_batches 10 \
        > "$LOG_FILE" 2>&1
    
    # Extract metrics
    final_loss=$(grep "train/loss_total" $LOG_FILE | tail -1 | grep -oE '[0-9]+\.[0-9]+' | head -1 || echo "N/A")
    min_loss=$(grep "train/loss_total" $LOG_FILE | grep -oE '[0-9]+\.[0-9]+' | sort -n | head -1 || echo "N/A")
    
    # Determine status
    if [[ "$min_loss" != "N/A" ]]; then
        initial_loss=$(grep "train/loss_total" $LOG_FILE | head -5 | grep -oE '[0-9]+\.[0-9]+' | head -1)
        if (( $(echo "$min_loss < $initial_loss * 0.9" | bc -l) )); then
            status="learning"
        else
            status="plateau"
        fi
    else
        status="failed"
    fi
    
    # Save to summary
    echo "$STRATEGY,$SIGNAL_MODE,$SIGNAL_STRENGTH,$DIM,$final_loss,$min_loss,$status" >> $SUMMARY_FILE
    
    # Mark as completed
    touch "$TEST_DIR/.completed"
    
    echo "  Status: $status, Min loss: $min_loss"
    
    # Clean up environment
    unset FEATURE_DIM FEATURE_STRATEGY SIGNAL_MODE SIGNAL_STRENGTH ADD_NOISE NOISE_SCALE
}

# Main testing
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Feature Strategy Comparison Test${NC}"
echo -e "${BLUE}========================================${NC}"

# Test 1: Compare basic strategies at 512D
echo -e "\n${GREEN}Test 1: Basic Comparison at 512D${NC}"
echo "----------------------------------------"

# Test repeat_label with noise
run_test "repeat_label" 512

# Test real_with_signal with different modes
for mode in "${SIGNAL_MODES[@]}"; do
    run_test "real_with_signal" 512 "$mode" "1.0"
done

# Test 2: Best strategy across dimensions
echo -e "\n${GREEN}Test 2: Dimension Scaling with Best Strategy${NC}"
echo "----------------------------------------"

# Determine best strategy from Test 1 (you may want to manually set this)
BEST_STRATEGY="real_with_signal"
BEST_MODE="additive"
BEST_STRENGTH="1.0"

for DIM in "${DIMENSIONS[@]}"; do
    run_test "$BEST_STRATEGY" "$DIM" "$BEST_MODE" "$BEST_STRENGTH"
done

# Test 3: Signal strength optimization
echo -e "\n${GREEN}Test 3: Signal Strength Optimization at 512D${NC}"
echo "----------------------------------------"

for strength in "${SIGNAL_STRENGTHS[@]}"; do
    run_test "real_with_signal" 512 "additive" "$strength"
done

# Generate report
echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}Test Results Summary${NC}"
echo -e "${BLUE}========================================${NC}"

echo -e "\n${YELLOW}Strategy Comparison at 512D:${NC}"
grep ",512," $SUMMARY_FILE | column -t -s','

echo -e "\n${YELLOW}Best Strategy Across Dimensions:${NC}"
grep "$BEST_STRATEGY" $SUMMARY_FILE | grep ",$BEST_MODE," | column -t -s','

echo -e "\n${GREEN}Analysis:${NC}"
# Find best configuration
best_config=$(sort -t',' -k6 -n $SUMMARY_FILE | grep "learning" | head -1)
if [[ ! -z "$best_config" ]]; then
    echo "Best configuration: $best_config"
fi

echo -e "\n${GREEN}Testing complete!${NC}"
echo "Full results in: $SUMMARY_FILE"