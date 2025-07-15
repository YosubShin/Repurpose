# Repurpose Dataset Preprocessing Pipeline

This directory contains a complete preprocessing pipeline for the Repurpose dataset, designed to download videos and extract multi-modal features (visual, audio, and text) required for the video repurposing model.

## Overview

The preprocessing pipeline consists of four main components:

1. **Video Download** (`video_downloader.py`) - Downloads videos from YouTube using yt-dlp
2. **Visual Feature Extraction** (`visual_feature_extractor.py`) - Extracts visual features using video_features library
3. **Audio Feature Extraction** (`audio_feature_extractor.py`) - Extracts audio features using PANNs or librosa
4. **Text Feature Extraction** (`text_feature_extractor.py`) - Transcribes speech and extracts text embeddings

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Install video_features library:
```bash
pip install git+https://github.com/v-iashin/video_features.git
```

3. Install system dependencies:
```bash
# On Ubuntu/Debian
sudo apt install ffmpeg

# On macOS
brew install ffmpeg

# On Windows
# Download FFmpeg from https://ffmpeg.org/download.html
```

## Quick Start

### Process All Datasets

```bash
python main_preprocessing.py --all-datasets
```

### Process Specific Dataset

```bash
python main_preprocessing.py --dataset data/train.json
```

### Run Specific Steps

```bash
# Only download videos
python main_preprocessing.py --dataset data/train.json --steps download

# Only extract features (assumes videos are already downloaded)
python main_preprocessing.py --dataset data/train.json --steps visual audio text
```

### Test with Limited Videos

```bash
python main_preprocessing.py --dataset data/train.json --max-videos 10
```

## Individual Component Usage

### Video Download

```bash
python video_downloader.py --dataset data/train.json --output-dir raw_videos
```

### Visual Feature Extraction

```bash
python visual_feature_extractor.py --video-dir raw_videos --dataset data/train.json --output-dir data/video_clip_features
```

### Audio Feature Extraction

```bash
python audio_feature_extractor.py --video-dir raw_videos --dataset data/train.json --output-dir data/audio_pann_features
```

### Text Feature Extraction

```bash
python text_feature_extractor.py --video-dir raw_videos --dataset data/train.json --output-dir data/caption_features
```

## Configuration

The pipeline uses a YAML configuration file (`preprocessing_config.yaml`) that controls:

- Directory paths
- Processing parameters
- Feature extraction settings
- Error handling options
- Logging configuration

You can customize the configuration by editing the YAML file or creating a new one and specifying it with `--config`.

## Feature Verification

Verify that all features have been extracted correctly:

```bash
# Verify specific dataset
python main_preprocessing.py --dataset data/train.json --verify

# Verify all datasets
python main_preprocessing.py --all-datasets --verify
```

## Expected Output Structure

After running the preprocessing pipeline, you should have the following directory structure:

```
data/
├── train.json                    # Dataset metadata
├── val.json
├── test.json
├── video_clip_features/         # Visual features (512-dim)
│   ├── Vlv4Q9GkNNw.npy
│   ├── RHqpvFnmzeA.npy
│   └── ...
├── audio_pann_features/         # Audio features (2048-dim)
│   ├── Vlv4Q9GkNNw.npy
│   ├── RHqpvFnmzeA.npy
│   └── ...
├── caption_features/            # Text features (384-dim)
│   ├── Vlv4Q9GkNNw.npy
│   ├── RHqpvFnmzeA.npy
│   └── ...
└── transcripts/                 # Saved transcripts for reuse
    ├── Vlv4Q9GkNNw.json        # Raw transcript data
    ├── Vlv4Q9GkNNw.txt         # Human-readable transcript
    ├── RHqpvFnmzeA.json
    ├── RHqpvFnmzeA.txt
    └── ...

raw_videos/                      # Downloaded videos (optional cleanup)
├── Vlv4Q9GkNNw.mp4
├── RHqpvFnmzeA.mp4
└── ...
```

## Feature Dimensions

The extracted features have the following dimensions:

- **Visual Features**: 512-dimensional vectors, 1 per second of video
- **Audio Features**: 2048-dimensional vectors, 1 per second of audio
- **Text Features**: 384-dimensional vectors, 1 per second of transcribed speech (full video duration)

## Transcript Caching

The text feature extractor automatically saves transcripts for reuse:

- **JSON files** (`data/transcripts/*.json`): Raw transcript data with timestamps
- **Text files** (`data/transcripts/*.txt`): Human-readable transcripts with formatted timestamps
- **Automatic reuse**: If a transcript exists, it will be loaded instead of re-transcribing
- **Time savings**: Transcription is skipped on subsequent runs, significantly reducing processing time

Example transcript format (`RHqpvFnmzeA.txt`):
```
Transcript for RHqpvFnmzeA
==================================================

[00:24 - 00:30] Hello everyone, welcome to today's video
[00:30 - 00:45] Today we're going to discuss...
```

## Error Handling

The pipeline includes robust error handling:

- **Resume functionality**: Skip already processed videos
- **Retry logic**: Automatic retries for failed downloads/extractions
- **Fallback methods**: Alternative extraction methods if primary tools fail
- **Progress tracking**: JSON files track completion status
- **Detailed logging**: Comprehensive logs for debugging

## Performance Optimization

- **GPU acceleration**: Automatic GPU detection and usage
- **Parallel processing**: Configurable number of worker processes
- **Batch processing**: Process videos in configurable batches
- **Memory management**: Efficient handling of large video files

## Troubleshooting

### Common Issues

1. **yt-dlp download failures**:
   - Check internet connection
   - Verify video availability
   - Update yt-dlp: `pip install --upgrade yt-dlp`

2. **Feature extraction failures**:
   - Ensure FFmpeg is installed and in PATH
   - Check available GPU memory
   - Verify input video format

3. **Memory issues**:
   - Reduce batch size in configuration
   - Enable raw video cleanup
   - Use CPU instead of GPU for large videos

4. **Dependency issues**:
   - Install video_features library separately
   - Check CUDA compatibility for GPU acceleration
   - Verify FFmpeg installation

### Debugging

Enable debug logging for detailed output:

```bash
python main_preprocessing.py --dataset data/train.json --log-level DEBUG
```

Check error logs:
- `preprocessing.log` - General processing logs
- `preprocessing_errors.json` - Failed video details

## Dataset Information

The Repurpose dataset contains:

- **10,000+** videos from YouTube
- **120,000+** annotated engaging segments
- Videos from various domains (vlogs, interviews, tutorials, etc.)
- Temporal annotations for segment boundaries
- Coverage metrics for engagement levels

Each dataset entry contains:
- `youtube_id`: YouTube video identifier
- `timeRange`: Full video time range
- `segments`: List of engaging segment boundaries
- `coverage`: Percentage of video that is engaging

## License

This preprocessing pipeline is designed for the Repurpose dataset research project. Please refer to the main project license for usage terms.

## Citation

If you use this preprocessing pipeline, please cite the original paper:

```bibtex
@inproceedings{repurpose2025,
  title={Video Repurposing from User Generated Content},
  author={[Authors]},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2025}
}
```