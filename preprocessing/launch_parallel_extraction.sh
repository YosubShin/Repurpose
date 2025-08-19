#!/bin/bash

# Parallel Feature Extraction Launcher Script
# This script splits the video dataset across multiple Python processes
# to fully utilize all CPU cores and bypass GIL limitations

# Default parameters - conservative to prevent OOM
NUM_PROCESSES=2  # Reduced to prevent OOM kills
VIDEO_DIR=""
DATASETS=""
OUTPUT_DIR="data/video_clip_features"
NUM_WORKERS=4  # Reduced to prevent OOM
BATCH_SIZE=32  # Reduced to prevent OOM
LOG_LEVEL="INFO"
EXTRA_ARGS=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num-processes)
            NUM_PROCESSES="$2"
            shift 2
            ;;
        --video-dir)
            VIDEO_DIR="$2"
            shift 2
            ;;
        --datasets)
            shift
            DATASETS=""
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                DATASETS="$DATASETS $1"
                shift
            done
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --inject-hints|--use-black-white)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
        --help)
            echo "Usage: $0 --video-dir DIR --datasets train.json val.json test.json [OPTIONS]"
            echo ""
            echo "Required arguments:"
            echo "  --video-dir DIR           Directory containing video files"
            echo "  --datasets FILES          Dataset JSON files (space-separated)"
            echo ""
            echo "Optional arguments:"
            echo "  --num-processes N         Number of parallel processes (default: 4)"
            echo "  --output-dir DIR          Output directory for features (default: data/video_clip_features)"
            echo "  --num-workers N           CPU workers per process (default: 8)"
            echo "  --batch-size N            GPU batch size (default: 64)"
            echo "  --log-level LEVEL         Log level (default: INFO)"
            echo "  --inject-hints            Enable hint injection"
            echo "  --use-black-white         Use black/white frames"
            echo ""
            echo "Example:"
            echo "  $0 --video-dir videos/ --datasets train.json val.json test.json --num-processes 4"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$VIDEO_DIR" ] || [ -z "$DATASETS" ]; then
    echo "Error: --video-dir and --datasets are required"
    echo "Run with --help for usage information"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/logs"

# Count total videos to determine split sizes
echo "Counting total videos in dataset files..."
TOTAL_VIDEOS=$(python3 -c "
import json
import sys

video_ids = set()
datasets = '$DATASETS'.split()

for dataset_path in datasets:
    try:
        with open(dataset_path, 'r') as f:
            data = json.load(f)
            for item in data:
                video_ids.add(item['youtube_id'])
    except:
        pass

print(len(video_ids))
")

if [ -z "$TOTAL_VIDEOS" ] || [ "$TOTAL_VIDEOS" -eq 0 ]; then
    echo "Error: Could not count videos in dataset files"
    exit 1
fi

echo "Total unique videos found: $TOTAL_VIDEOS"
echo "Splitting across $NUM_PROCESSES processes..."

# Calculate videos per process
VIDEOS_PER_PROCESS=$((TOTAL_VIDEOS / NUM_PROCESSES))
REMAINDER=$((TOTAL_VIDEOS % NUM_PROCESSES))

echo "Videos per process: ~$VIDEOS_PER_PROCESS"
echo ""

# Launch parallel processes
PIDS=()
for ((i=0; i<$NUM_PROCESSES; i++)); do
    # Calculate start and end indices for this process
    START_INDEX=$((i * VIDEOS_PER_PROCESS))
    
    if [ $i -eq $((NUM_PROCESSES - 1)) ]; then
        # Last process handles remainder
        END_INDEX=$TOTAL_VIDEOS
    else
        END_INDEX=$(((i + 1) * VIDEOS_PER_PROCESS))
    fi
    
    # Create log file for this process
    LOG_FILE="$OUTPUT_DIR/logs/process_${i}_${START_INDEX}_${END_INDEX}.log"
    
    # Launch the Python process
    echo "Starting process $i: videos $START_INDEX to $END_INDEX"
    echo "  Log file: $LOG_FILE"
    
    python3 visual_feature_extractor_clip.py \
        --video-dir "$VIDEO_DIR" \
        --datasets $DATASETS \
        --output-dir "$OUTPUT_DIR" \
        --num-workers "$NUM_WORKERS" \
        --batch-size "$BATCH_SIZE" \
        --log-level "$LOG_LEVEL" \
        --start-index "$START_INDEX" \
        --end-index "$END_INDEX" \
        $EXTRA_ARGS \
        > "$LOG_FILE" 2>&1 &
    
    PID=$!
    PIDS+=($PID)
    echo "  PID: $PID"
    echo ""
    
    # Small delay to avoid simultaneous startup
    sleep 1
done

echo "All $NUM_PROCESSES processes started"
echo "PIDs: ${PIDS[@]}"
echo ""

# Function to check if processes are still running
check_processes() {
    local running=0
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            ((running++))
        fi
    done
    echo $running
}

# Monitor progress
echo "Monitoring progress..."
echo "You can check individual logs in: $OUTPUT_DIR/logs/"
echo ""

START_TIME=$(date +%s)

while true; do
    RUNNING=$(check_processes)
    
    if [ "$RUNNING" -eq 0 ]; then
        break
    fi
    
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    ELAPSED_MIN=$((ELAPSED / 60))
    ELAPSED_SEC=$((ELAPSED % 60))
    
    echo -ne "\rProcesses running: $RUNNING/$NUM_PROCESSES | Elapsed: ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
    
    sleep 5
done

echo ""
echo ""

# Check exit status of each process
echo "Checking process results..."
FAILED=0
for ((i=0; i<$NUM_PROCESSES; i++)); do
    PID=${PIDS[$i]}
    wait $PID
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "Process $i (PID $PID): SUCCESS"
    else
        echo "Process $i (PID $PID): FAILED (exit code: $EXIT_CODE)"
        ((FAILED++))
    fi
done

echo ""

# Final summary
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
TOTAL_MIN=$((TOTAL_TIME / 60))
TOTAL_SEC=$((TOTAL_TIME % 60))

echo "========================================="
echo "Parallel extraction completed!"
echo "Total time: ${TOTAL_MIN}m ${TOTAL_SEC}s"
echo "Successful processes: $((NUM_PROCESSES - FAILED))/$NUM_PROCESSES"

if [ $FAILED -gt 0 ]; then
    echo ""
    echo "WARNING: $FAILED process(es) failed"
    echo "Check log files in $OUTPUT_DIR/logs/ for details"
    exit 1
else
    echo ""
    echo "All processes completed successfully!"
    
    # Count total extracted features
    FEATURE_COUNT=$(ls -1 "$OUTPUT_DIR"/*.npy 2>/dev/null | wc -l)
    echo "Total features extracted: $FEATURE_COUNT"
fi

echo "========================================="