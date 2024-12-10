<h2 align="center">Video Repurposing from User Generated Content: A Large-scale Dataset and Benchmark</h2>

## News

* :fire: [2024.12.10] Our paper is accepted by AAAI-2025 !

## Summary

* [Introduction](#introduction)
* [Getting Started](#getting-started)
* [Datasets](#datasets)

## Introduction


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

- [UnAV](https://github.com/ttgeng233/UnAV)
- [Soft-NMS](https://github.com/DocF/Soft-NMS)
