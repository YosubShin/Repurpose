#!/usr/bin/env python3
"""
Script to inspect the data pipeline and verify labels/masks are correct.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
import sys
import os

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset.RepurposeClip import RepurposeClip
from dataset.args import get_args_parser
import argparse


def inspect_data(args):
    print("=== Data Pipeline Inspection ===\n")

    # Load dataset
    dataset = RepurposeClip(
        args.meta_dir,
        args.data_dir,
        split="train",
        max_snippet_length=args.max_snippet_length,
        window_size=args.window_size,
        sampling_rate=args.sampling_rate,
        max_windows_per_video=args.max_windows_per_video,
    )

    print(f"Total samples in dataset: {len(dataset)}")

    # Inspect first few samples
    print("\n=== Inspecting First 5 Samples ===")
    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        print(f"\nSample {i}:")
        print(f"  Video ID: {sample['video_id']}")
        print(f"  Visual shape: {sample['visual_feats'].shape}")
        print(f"  Audio shape: {sample['audio_feats'].shape}")
        print(f"  Text shape: {sample['text_feats'].shape}")
        print(f"  Labels shape: {sample['labels'].shape}")
        print(f"  Labels unique values: {torch.unique(sample['labels'])}")
        print(f"  Number of positive labels: {(sample['labels'] > 0).sum().item()}")
        print(f"  Segments shape: {sample['segments'].shape}")
        print(f"  Duration: {sample['duration']}")

    # Create a small dataloader to test batching
    print("\n=== Testing Batch Creation ===")
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=dataset.collate_fn,
    )

    # Get one batch
    for batch in dataloader:
        print("\nBatch shapes:")
        print(f"  Visual: {batch['visual_feats'].shape}")
        print(f"  Audio: {batch['audio_feats'].shape}")
        print(f"  Text: {batch['text_feats'].shape}")
        print(f"  Labels: {batch['labels'].shape}")
        print(f"  Masks: {batch['masks'].shape}")
        print(f"  Segments: {batch['segments'].shape}")

        print("\nMask statistics:")
        print(f"  Mask unique values: {torch.unique(batch['masks'])}")
        print(f"  Valid positions (mask=1): {batch['masks'].sum().item()}")
        print(f"  Total positions: {batch['masks'].numel()}")

        print("\nLabel statistics:")
        print(f"  Positive labels: {(batch['labels'] > 0).sum().item()}")
        print(f"  Total labels: {batch['labels'].numel()}")
        print(
            f"  Positive ratio: {(batch['labels'] > 0).sum().item() / batch['labels'].numel():.4f}"
        )

        # Check if labels and masks align properly
        print("\nAlignment check:")
        for i in range(batch["labels"].shape[0]):
            mask_i = batch["masks"][i].squeeze()
            labels_i = batch["labels"][i]
            valid_labels = labels_i[mask_i > 0]
            print(
                f"  Sample {i}: {(valid_labels > 0).sum().item()} positive labels out of {valid_labels.numel()} valid positions"
            )

        break  # Only inspect first batch

    # Analyze label distribution across dataset
    print("\n=== Dataset-wide Label Analysis ===")
    total_positive = 0
    total_labels = 0

    for i in range(min(100, len(dataset))):  # Sample first 100
        sample = dataset[i]
        labels = sample["labels"]
        total_positive += (labels > 0).sum().item()
        total_labels += labels.numel()

    print(f"Sampled {min(100, len(dataset))} videos:")
    print(f"  Total positive labels: {total_positive}")
    print(f"  Total labels: {total_labels}")
    print(f"  Positive ratio: {total_positive / total_labels:.4f}")

    # Check for any data loading issues
    print("\n=== Checking for Data Issues ===")
    issues = []

    for i in range(min(10, len(dataset))):
        sample = dataset[i]

        # Check for all-zero features
        if sample["visual_feats"].abs().sum() == 0:
            issues.append(f"Sample {i}: Visual features are all zero")
        if sample["audio_feats"].abs().sum() == 0:
            issues.append(f"Sample {i}: Audio features are all zero")
        if sample["text_feats"].abs().sum() == 0:
            issues.append(f"Sample {i}: Text features are all zero")

        # Check for no positive labels
        if (sample["labels"] > 0).sum() == 0:
            issues.append(f"Sample {i}: No positive labels")

    if issues:
        print("Found issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("No obvious data issues found in sampled data.")


def main():
    parser = argparse.ArgumentParser(
        "Inspect data pipeline", parents=[get_args_parser()], add_help=False
    )
    args = parser.parse_args()

    inspect_data(args)


if __name__ == "__main__":
    main()
