# Parallel Processing Guide

This guide explains how to run the Repurpose preprocessing pipeline in parallel across multiple SLURM jobs for faster processing.

## Overview

The parallel processing workflow consists of:
1. **Dataset Splitting**: Split large dataset files into smaller chunks
2. **Job Submission**: Submit multiple SLURM jobs to process chunks in parallel
3. **Monitoring**: Track job progress and completion

## Quick Start

### 1. Split the Dataset

```bash
# Split all datasets (train, val, test) into 20 chunks each
python split_dataset.py --data-dir data --output-dir data/splits --num-splits 20

# Or split just the training set into 50 chunks
python split_dataset.py --data-dir data --output-dir data/splits --num-splits 50 --dataset train
```

### 2. Submit Parallel Jobs

```bash
# Process training dataset with all steps (download + feature extraction)
./submit_parallel_jobs.sh --dataset train

# Process validation dataset, skip download (assumes videos already downloaded)
./submit_parallel_jobs.sh --dataset val --steps "visual audio text"

# Process all datasets with limited number of jobs
./submit_parallel_jobs.sh --dataset all --num-jobs 20

# Dry run to see what would be submitted
./submit_parallel_jobs.sh --dataset train --dry-run
```

### 3. Monitor Progress

```bash
# Check job status
squeue -u $USER

# Check GPU queue
squeue -p gpu

# Check completion markers
ls /home/yosubs/koa_scratch/repurpose/outputs/*_SUCCESS
ls /home/yosubs/koa_scratch/repurpose/outputs/*_FAILED
```

## Detailed Usage

### Dataset Splitting

The `split_dataset.py` script divides dataset JSON files into smaller chunks:

```bash
python split_dataset.py [OPTIONS]

Options:
  --data-dir DIR         Directory containing dataset JSON files [default: data]
  --output-dir DIR       Output directory for split files [default: data/splits]
  --num-splits N         Number of splits to create [required]
  --dataset TYPE         Which dataset to split: train, val, test, all [default: all]
```

**Examples:**
```bash
# Split into 10 chunks each
python split_dataset.py --num-splits 10

# Split only training data into 100 chunks  
python split_dataset.py --dataset train --num-splits 100

# Custom directories
python split_dataset.py --data-dir /path/to/data --output-dir /path/to/splits --num-splits 25
```

### Job Submission

The `submit_parallel_jobs.sh` script submits SLURM jobs for parallel processing:

```bash
./submit_parallel_jobs.sh [OPTIONS]

Options:
  -d, --dataset TYPE     Dataset type: train, val, test, all [default: train]
  -s, --steps STEPS      Processing steps [default: "download visual audio text"]
  -n, --num-jobs NUM     Maximum number of jobs to submit
  -r, --dry-run          Preview without submitting
  -h, --help             Show help
```

**Processing Steps:**
- `download`: Download videos from YouTube
- `visual`: Extract CLIP visual features (requires GPU)
- `audio`: Extract PANNs audio features (requires GPU)  
- `text`: Extract text features via transcription + embeddings (requires GPU)

**Examples:**
```bash
# Full preprocessing pipeline
./submit_parallel_jobs.sh --dataset train --steps "download visual audio text"

# Skip download if videos already exist
./submit_parallel_jobs.sh --dataset val --steps "visual audio text"

# Only visual features
./submit_parallel_jobs.sh --dataset test --steps "visual"

# Limit to 10 jobs
./submit_parallel_jobs.sh --dataset all --num-jobs 10
```

## Resource Requirements

### SLURM Job Specifications
- **Time**: 12 hours (adjustable in `slurm_preprocessing_job.sh`)
- **GPU**: 1 GPU required for feature extraction
- **CPU**: 8 cores
- **Memory**: 32GB RAM
- **Partition**: `kill-shared` (changeable)

### Storage Requirements
- **Raw videos**: ~50-100GB per 1000 videos
- **Features**: ~1-2GB per 1000 videos per modality
- **Temporary space**: ~10GB during processing

## File Structure

After parallel processing, you'll have:

```
/home/yosubs/koa_scratch/repurpose/
├── data/
│   ├── video_clip_features/     # Visual features (512-dim CLIP)
│   ├── audio_pann_features/     # Audio features (2048-dim PANNs)
│   ├── caption_features/        # Text features (384-dim)
│   └── transcripts/             # Cached transcripts
├── raw_videos/                  # Downloaded videos
└── outputs/                     # Job logs and status markers
    ├── slurm-*.out             # Job stdout logs
    ├── slurm-*.err             # Job stderr logs
    ├── *_SUCCESS               # Completion markers
    ├── *_FAILED                # Failure markers
    └── submitted_jobs_*.json   # Job submission metadata
```

## Monitoring and Debugging

### Check Job Status
```bash
# All your jobs
squeue -u $USER

# Specific jobs
squeue -j 123456,123457,123458

# GPU queue
squeue -p gpu
```

### Check Completion
```bash
# Count successful chunks
ls /home/yosubs/koa_scratch/repurpose/outputs/*_SUCCESS | wc -l

# Count failed chunks
ls /home/yosubs/koa_scratch/repurpose/outputs/*_FAILED | wc -l

# Check specific chunk
ls /home/yosubs/koa_scratch/repurpose/outputs/train_chunk_001_*
```

### View Logs
```bash
# Latest job output
tail -f /home/yosubs/koa_scratch/repurpose/outputs/slurm-*.out

# Error logs
grep -r "Error\|Failed" /home/yosubs/koa_scratch/repurpose/outputs/slurm-*.err
```

### Resubmit Failed Jobs
```bash
# Find failed chunks
failed_chunks=$(ls /home/yosubs/koa_scratch/repurpose/outputs/*_FAILED | sed 's/.*\/\(.*\)_FAILED/\1/')

# Resubmit specific chunk
sbatch slurm_preprocessing_job.sh data/splits/train_chunk_001.json "visual audio text"
```

## Performance Tips

1. **Optimize chunk size**: 50-100 videos per chunk works well
2. **Skip download for retries**: Use `--steps "visual audio text"` 
3. **Monitor disk usage**: Clean up raw videos if space is limited
4. **Stagger submissions**: Use delays between job submissions
5. **Check dependencies**: Ensure all required Python packages are installed

## Troubleshooting

**Common Issues:**

1. **"No chunk files found"**
   - Run `split_dataset.py` first
   - Check `--dataset` parameter matches split file names

2. **"Module not found"**
   - Activate correct conda environment
   - Install missing dependencies: `pip install -r requirements.txt`

3. **CUDA out of memory**
   - Reduce batch size in preprocessing config
   - Use smaller video chunks

4. **Jobs stuck in queue**
   - Check GPU availability: `sinfo -p gpu`
   - Consider using different partition

5. **Feature dimension mismatch**
   - Ensure PyAV/OpenCV extract correct number of frames
   - Check video duration vs dataset metadata