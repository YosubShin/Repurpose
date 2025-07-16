#!/bin/bash
# Script to submit multiple SLURM preprocessing jobs for processing dataset chunks

# Configuration
SPLITS_DIR="/home/yosubs/co/Repurpose/data/splits"
OUTPUT_DIR="/home/yosubs/koa_scratch/repurpose/outputs"
SCRIPT_PATH="/home/yosubs/co/Repurpose/preprocessing/slurm_preprocessing_job.sh"

# Default processing steps
DEFAULT_STEPS="download visual audio text"

# Parse command line arguments
DATASET_TYPE="train"  # Default to train dataset
PROCESSING_STEPS="$DEFAULT_STEPS"
NUM_JOBS=""
DRY_RUN=false

# Help function
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Submit parallel SLURM jobs for Repurpose dataset preprocessing.

OPTIONS:
    -d, --dataset TYPE     Dataset type to process (train, val, test, all) [default: train]
    -s, --steps STEPS      Processing steps (download, visual, audio, text) [default: all steps]
    -n, --num-jobs NUM     Maximum number of jobs to submit (limits chunks processed)
    -r, --dry-run          Show what would be submitted without actually submitting
    -h, --help             Show this help message

EXAMPLES:
    $0                                    # Process train dataset with all steps
    $0 -d val -s "visual audio text"      # Process val dataset, skip download
    $0 -d all -n 10                      # Process all datasets, max 10 jobs
    $0 --dry-run                         # Preview what would be submitted

STEPS:
    download    Download videos from YouTube
    visual      Extract CLIP visual features  
    audio       Extract PANNs audio features
    text        Extract text features (transcription + embeddings)

DATASETS:
    train       Training dataset chunks
    val         Validation dataset chunks  
    test        Test dataset chunks
    all         All dataset chunks
EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--dataset)
            DATASET_TYPE="$2"
            shift 2
            ;;
        -s|--steps)
            PROCESSING_STEPS="$2"
            shift 2
            ;;
        -n|--num-jobs)
            NUM_JOBS="$2"
            shift 2
            ;;
        -r|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

echo "=== REPURPOSE PARALLEL PREPROCESSING SUBMISSION ==="
echo "Dataset type: $DATASET_TYPE"
echo "Processing steps: $PROCESSING_STEPS"
echo "Splits directory: $SPLITS_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "SLURM script: $SCRIPT_PATH"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Check if splits directory exists
if [ ! -d "$SPLITS_DIR" ]; then
    echo "Error: Splits directory $SPLITS_DIR does not exist"
    echo "Run split_dataset.py first to create dataset chunks"
    exit 1
fi

# Check if SLURM script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: SLURM script not found at $SCRIPT_PATH"
    exit 1
fi

# Find chunk files based on dataset type
if [ "$DATASET_TYPE" = "all" ]; then
    CHUNK_FILES=(${SPLITS_DIR}/*_chunk_*.json)
else
    CHUNK_FILES=(${SPLITS_DIR}/${DATASET_TYPE}_chunk_*.json)
fi

NUM_CHUNKS=${#CHUNK_FILES[@]}

if [ $NUM_CHUNKS -eq 0 ]; then
    echo "Error: No chunk files found for dataset type: $DATASET_TYPE"
    echo "Available files in $SPLITS_DIR:"
    ls -la "$SPLITS_DIR"/*.json 2>/dev/null || echo "  No JSON files found"
    exit 1
fi

# Limit number of jobs if specified
if [ -n "$NUM_JOBS" ] && [ "$NUM_JOBS" -lt "$NUM_CHUNKS" ]; then
    echo "Limiting to first $NUM_JOBS chunks (out of $NUM_CHUNKS available)"
    CHUNK_FILES=("${CHUNK_FILES[@]:0:$NUM_JOBS}")
    NUM_CHUNKS=$NUM_JOBS
fi

echo "Found $NUM_CHUNKS chunk files to process"

# Show chunk files
echo ""
echo "Chunk files to process:"
for chunk_file in "${CHUNK_FILES[@]}"; do
    chunk_name=$(basename "$chunk_file")
    chunk_size=$(jq length "$chunk_file" 2>/dev/null || echo "unknown")
    echo "  $chunk_name ($chunk_size videos)"
done

# Check GPU availability
echo ""
echo "Checking GPU queue status..."
sinfo -p gpu --format="%.15P %.5a %.10l %.6D %.6t %.15N %.8c %.8m %.10G" 2>/dev/null || echo "Unable to check GPU queue"

echo ""
echo "Current GPU queue:"
squeue -p gpu --format="%.10i %.15j %.8u %.2t %.10M %.5D %R" 2>/dev/null || echo "Unable to check current queue"

# Confirm submission
echo ""
if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN: Would submit $NUM_CHUNKS preprocessing jobs"
    echo "Processing steps: $PROCESSING_STEPS"
    for chunk_file in "${CHUNK_FILES[@]}"; do
        chunk_name=$(basename "$chunk_file")
        echo "  Would submit: sbatch $SCRIPT_PATH $chunk_file '$PROCESSING_STEPS'"
    done
    exit 0
fi

read -p "Do you want to proceed with submitting $NUM_CHUNKS preprocessing jobs? (y/N): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Submit jobs
echo ""
echo "Submitting jobs..."
submitted_jobs=()
failed_submissions=0

for chunk_file in "${CHUNK_FILES[@]}"; do
    if [ -f "$chunk_file" ]; then
        chunk_name=$(basename "$chunk_file")
        echo "Submitting job for: $chunk_name"
        
        # Submit the job and capture job ID
        if job_output=$(sbatch "$SCRIPT_PATH" "$chunk_file" "$PROCESSING_STEPS" 2>&1); then
            if [[ $job_output =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
                job_id="${BASH_REMATCH[1]}"
                submitted_jobs+=($job_id)
                echo "  → Job ID: $job_id"
            else
                echo "  → Warning: Unexpected output: $job_output"
                ((failed_submissions++))
            fi
        else
            echo "  → Error submitting job: $job_output"
            ((failed_submissions++))
        fi
        
        # Small delay to avoid overwhelming the scheduler
        sleep 1
    fi
done

echo ""
echo "=== SUBMISSION SUMMARY ==="
echo "Total chunks: $NUM_CHUNKS"
echo "Successfully submitted: $((NUM_CHUNKS - failed_submissions))"
echo "Failed submissions: $failed_submissions"
echo "Dataset type: $DATASET_TYPE"
echo "Processing steps: $PROCESSING_STEPS"

if [ ${#submitted_jobs[@]} -gt 0 ]; then
    echo ""
    echo "Submitted job IDs: ${submitted_jobs[*]}"
    echo ""
    echo "=== MONITORING COMMANDS ==="
    echo "Check job status:      squeue -u $USER"
    echo "Check GPU queue:       squeue -p gpu"
    echo "Check specific jobs:   squeue -j $(IFS=,; echo "${submitted_jobs[*]}")"
    echo "Cancel all jobs:       scancel $(IFS=' '; echo "${submitted_jobs[*]}")"
    echo "Cancel by user:        scancel -u $USER"
    echo ""
    echo "=== OUTPUT LOCATIONS ==="
    echo "Job logs:              $OUTPUT_DIR/slurm-*.out"
    echo "Error logs:            $OUTPUT_DIR/slurm-*.err"
    echo "Success markers:       $OUTPUT_DIR/*_SUCCESS"
    echo "Failure markers:       $OUTPUT_DIR/*_FAILED"
    echo "Feature outputs:       /home/yosubs/koa_scratch/repurpose/data/"
    
    # Save job information
    job_info_file="$OUTPUT_DIR/submitted_jobs_$(date +%Y%m%d_%H%M%S).json"
    cat > "$job_info_file" << EOF
{
  "submission_time": "$(date -Iseconds)",
  "dataset_type": "$DATASET_TYPE",
  "processing_steps": "$PROCESSING_STEPS",
  "num_chunks": $NUM_CHUNKS,
  "job_ids": [$(IFS=,; echo "\"${submitted_jobs[*]//,/\",\"}")"],
  "chunk_files": [$(printf '"%s",' "${CHUNK_FILES[@]}" | sed 's/,$//')]
}
EOF
    echo "Job info saved to: $job_info_file"
fi

echo ""
echo "All jobs submitted successfully!"
echo ""
echo "=== NEXT STEPS ==="
echo "1. Monitor job progress: watch 'squeue -u $USER'"
echo "2. Check completion: ls $OUTPUT_DIR/*_SUCCESS"
echo "3. Check failures: ls $OUTPUT_DIR/*_FAILED"
echo "4. Verify outputs: ls /home/yosubs/koa_scratch/repurpose/data/"