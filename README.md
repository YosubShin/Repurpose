<h2 align="center">Video Repurposing from User Generated Content: A Large-scale Dataset and Benchmark</h2>

## News

* :fire: [2024.12.10] Our paper is accepted by AAAI-2025 !

## Introduction

This repository provides the PyTorch implementation for the paper [**Video Repurposing from User Generated Content: A Large-scale Dataset and Benchmark**](https://arxiv.org/abs/2412.08879). The research introduces **Repurpose-10K**, a large-scale dataset designed to tackle the challenge of **long-to-short video repurposing**. The dataset contains over **10,000 videos** and **120,000+ annotated clips**, making it a benchmark for automatic video repurposing.

### What is Video Repurposing?
With the rise of **short-form video platforms** like TikTok, Instagram Reels, and YouTube Shorts, there is a growing need to **efficiently extract engaging segments** from long-form content such as vlogs, interviews, and live streams. Video repurposing involves:
- Identifying **highly engaging** segments from long videos.
- Ensuring **narrative coherence** in the repurposed clips.
- Optimizing for **direct publishing** on social media.

### About **Repurpose-10K**
To address the lack of large-scale benchmarks for this task, **Repurpose-10K** was created by collecting **real-world user interactions** on **User Generated Content (UGC)**. The annotation process involves:
1. **Initial segmentation** using AI-assisted tools.
2. **User preference voting** to mark preferred clips.
3. **Manual refinement** of timestamps by content creators.

This ensures high-quality, **human-curated** ground truth labels for training video repurposing models.

## Getting Started

### Setting Up Your Environment
To ensure a smooth experience running the scripts, set up a dedicated `conda` environment by executing the following commands in your terminal:

```bash
conda create -n repurpose python=3.9
conda activate repurpose
pip install -r requirements.txt
```

### Preparing Your Data

The train/validation/test splits are provided in the `/data` directory. Follow these steps for data preparation:

1. Download the source videos using [yt-dlp](https://github.com/yt-dlp/yt-dlp).
2. Extract the required features as mentioned in our paper using these repositories:
   - [video_features](https://github.com/v-iashin/video_features)
   - [panns_inference](https://github.com/qiuqiangkong/panns_inference)
   - [sentence-transformers](https://github.com/UKPLab/sentence-transformers)
   - [whisperX](https://github.com/m-bain/whisperX)

### Training Your Model
To begin training the model, use the command below:

```bash
python main.py --config_path configs/Repurpose.yaml
```

For model evaluation, execute the following command:

```bash
python inference.py --config_path configs/Repurpose.yaml --resume your_ckpt_path
```
Replace `your_ckpt_path` with the actual path to your checkpoint file.

## Acknowledgments

We would like to extend our gratitude to the authors and contributors of the following repositories, which have been instrumental in the development of our implementation:

- https://github.com/ttgeng233/UnAV
- https://github.com/DocF/Soft-NMS
