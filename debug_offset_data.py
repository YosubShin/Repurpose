#!/usr/bin/env python3
"""
Debug script to check offset data processing and why offsets are non-integers.
"""

import json
import numpy as np
import torch
from compatible_dataset import SequenceDataset, create_sequence_dataloader
import argparse


def check_json_annotations(json_file):
    """Check the raw annotations in JSON file."""
    print("\n" + "=" * 80)
    print("CHECKING RAW JSON ANNOTATIONS")
    print("=" * 80)

    with open(json_file, "r") as f:
        data = json.load(f)

    print(f"\nTotal videos: {len(data)}")

    # Check first few videos
    for i, (video_id, video_data) in enumerate(list(data.items())[:3]):
        print(f"\n--- Video {i+1}: {video_id} ---")
        print(f"Duration: {video_data.get('duration', 'N/A')} seconds")
        print(f"Number of highlight segments: {len(video_data['timestamps'])}")

        # Check first few segments
        for j, segment in enumerate(video_data["timestamps"][:3]):
            print(f"\n  Segment {j+1}:")
            print(f"    Start: {segment[0]} seconds")
            print(f"    End: {segment[1]} seconds")
            print(f"    Duration: {segment[1] - segment[0]:.2f} seconds")

            # Check if these are already in frames or seconds
            if segment[0] > 1000:
                print(
                    f"    WARNING: Large values suggest these might be in milliseconds or frames"
                )


def check_dataset_offsets(dataset, num_samples=3):
    """Check how offsets are computed in the dataset."""
    print("\n" + "=" * 80)
    print("CHECKING DATASET OFFSET COMPUTATION")
    print("=" * 80)

    for idx in range(min(num_samples, len(dataset))):
        print(f"\n--- Sample {idx + 1} ---")
        sample = dataset[idx]

        # Check the shapes
        print(f"Video ID: {sample.get('video_id', 'N/A')}")
        print(f"Labels shape: {sample['labels'].shape}")
        print(f"Offsets shape: {sample['offsets'].shape}")

        # Find positive labels
        labels = sample["labels"].numpy()
        offsets = sample["offsets"].numpy()
        positive_idx = np.where(labels > 0.5)[0]

        if len(positive_idx) > 0:
            print(f"Positive positions: {len(positive_idx)}")

            # Check first few positive positions
            for i in range(min(5, len(positive_idx))):
                pos = positive_idx[i]
                left_off = offsets[pos, 0]
                right_off = offsets[pos, 1]

                print(f"\n  Position {pos}:")
                print(f"    Left offset: {left_off:.4f}")
                print(f"    Right offset: {right_off:.4f}")
                print(
                    f"    Is integer? Left: {left_off == int(left_off)}, Right: {right_off == int(right_off)}"
                )

                # Check the segment this represents
                segment_start = pos - left_off
                segment_end = pos + right_off
                print(f"    Segment: [{segment_start:.2f}, {segment_end:.2f}]")

            # Check for consecutive positives (should be in same segment)
            for i in range(len(positive_idx) - 1):
                if positive_idx[i + 1] == positive_idx[i] + 1:
                    # These should point to the same segment
                    pos1, pos2 = positive_idx[i], positive_idx[i + 1]
                    seg1_start = pos1 - offsets[pos1, 0]
                    seg1_end = pos1 + offsets[pos1, 1]
                    seg2_start = pos2 - offsets[pos2, 0]
                    seg2_end = pos2 + offsets[pos2, 1]

                    if (
                        abs(seg1_start - seg2_start) > 0.01
                        or abs(seg1_end - seg2_end) > 0.01
                    ):
                        print(
                            f"\n  WARNING: Consecutive positions {pos1} and {pos2} point to different segments!"
                        )
                        print(f"    Segment 1: [{seg1_start:.2f}, {seg1_end:.2f}]")
                        print(f"    Segment 2: [{seg2_start:.2f}, {seg2_end:.2f}]")
                    break  # Just check one pair as example


def check_dataloader_batch(dataloader):
    """Check a batch from the dataloader."""
    print("\n" + "=" * 80)
    print("CHECKING DATALOADER BATCH")
    print("=" * 80)

    batch = next(iter(dataloader))

    print(f"\nBatch keys: {batch.keys()}")
    print(f"Batch size: {batch['labels'].shape[0]}")
    print(f"Sequence length: {batch['labels'].shape[1]}")

    # Check offsets tensor
    offsets = batch["offsets"]
    print(f"\nOffsets tensor:")
    print(f"  Shape: {offsets.shape}")
    print(f"  Dtype: {offsets.dtype}")
    print(f"  Min: {offsets.min().item():.4f}")
    print(f"  Max: {offsets.max().item():.4f}")
    print(f"  Mean: {offsets.mean().item():.4f}")

    # Check if offsets are integers
    offsets_np = offsets.numpy()
    is_integer = np.allclose(offsets_np, np.round(offsets_np), atol=0.01)
    print(f"  Are offsets close to integers? {is_integer}")

    if not is_integer:
        # Find examples of non-integer offsets
        non_int_mask = ~np.isclose(offsets_np, np.round(offsets_np), atol=0.01)
        non_int_positions = np.where(non_int_mask)

        if len(non_int_positions[0]) > 0:
            print(f"\n  Examples of non-integer offsets:")
            for i in range(min(5, len(non_int_positions[0]))):
                b, t, d = (
                    non_int_positions[0][i],
                    non_int_positions[1][i],
                    non_int_positions[2][i],
                )
                value = offsets_np[b, t, d]
                print(
                    f"    Batch {b}, Time {t}, Dim {d}: {value:.4f} (nearest int: {round(value)})"
                )

    # Check ground truth segments
    if "gt_segments" in batch:
        print(f"\nGround truth segments in batch:")
        for b in range(min(2, batch["labels"].shape[0])):
            segments = batch["gt_segments"][b]
            print(f"  Batch {b}: {segments}")


def analyze_offset_pattern(dataloader):
    """Analyze the pattern of offsets across the dataset."""
    print("\n" + "=" * 80)
    print("ANALYZING OFFSET PATTERNS")
    print("=" * 80)

    all_offsets = []
    all_durations = []

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 5:  # Check first 5 batches
            break

        labels = batch["labels"]
        offsets = batch["offsets"]

        # Get positive positions
        positive_mask = labels > 0.5
        positive_offsets = offsets[positive_mask]

        if len(positive_offsets) > 0:
            all_offsets.append(positive_offsets.numpy())
            durations = positive_offsets[:, 0] + positive_offsets[:, 1]
            all_durations.append(durations.numpy())

    if all_offsets:
        all_offsets = np.concatenate(all_offsets)
        all_durations = np.concatenate(all_durations)

        print(f"\nOffset statistics across dataset:")
        print(f"  Total positive positions analyzed: {len(all_offsets)}")
        print(
            f"  Left offset - Min: {all_offsets[:, 0].min():.4f}, Max: {all_offsets[:, 0].max():.4f}, Mean: {all_offsets[:, 0].mean():.4f}"
        )
        print(
            f"  Right offset - Min: {all_offsets[:, 1].min():.4f}, Max: {all_offsets[:, 1].max():.4f}, Mean: {all_offsets[:, 1].mean():.4f}"
        )
        print(
            f"  Duration - Min: {all_durations.min():.4f}, Max: {all_durations.max():.4f}, Mean: {all_durations.mean():.4f}"
        )

        # Check for fractional parts
        left_fractions = all_offsets[:, 0] - np.floor(all_offsets[:, 0])
        right_fractions = all_offsets[:, 1] - np.floor(all_offsets[:, 1])

        print(f"\nFractional parts analysis:")
        print(
            f"  Left offset fractions - Min: {left_fractions.min():.4f}, Max: {left_fractions.max():.4f}, Mean: {left_fractions.mean():.4f}"
        )
        print(
            f"  Right offset fractions - Min: {right_fractions.min():.4f}, Max: {right_fractions.max():.4f}, Mean: {right_fractions.mean():.4f}"
        )

        # Check unique fractional values (might reveal a pattern)
        unique_left_fractions = np.unique(np.round(left_fractions, 4))[:10]
        unique_right_fractions = np.unique(np.round(right_fractions, 4))[:10]
        print(f"  First 10 unique left fractions: {unique_left_fractions}")
        print(f"  First 10 unique right fractions: {unique_right_fractions}")

        # Check if fractions follow a pattern (e.g., multiples of some value)
        if len(unique_left_fractions) > 1:
            diffs = np.diff(unique_left_fractions)
            if np.allclose(diffs, diffs[0], atol=0.001):
                print(f"  Left fractions appear to be multiples of: {diffs[0]:.6f}")

        # Check segment duration distribution
        print(f"\nSegment duration distribution:")
        duration_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, np.inf]
        hist, _ = np.histogram(all_durations, bins=duration_bins)
        for i in range(len(hist)):
            if hist[i] > 0:
                print(
                    f"  [{duration_bins[i]:.0f}-{duration_bins[i+1]:.0f}): {hist[i]} segments"
                )


def main():
    parser = argparse.ArgumentParser(description="Debug offset data processing")
    parser.add_argument("--train-json", required=True, help="Training JSON file")
    parser.add_argument("--visual-dir", required=True, help="Visual features directory")
    parser.add_argument("--audio-dir", required=True, help="Audio features directory")
    parser.add_argument(
        "--caption-dir", required=True, help="Caption features directory"
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")

    args = parser.parse_args()

    print("=" * 80)
    print("OFFSET DATA DEBUGGING")
    print("=" * 80)

    # 1. Check raw JSON annotations
    check_json_annotations(args.train_json)

    # 2. Create dataset and check offset computation
    feature_dirs = {
        "visual": args.visual_dir,
        "audio": args.audio_dir,
        "caption": args.caption_dir,
    }

    dataset = SequenceDataset(
        feature_dirs=feature_dirs, annotation_file=args.train_json
    )

    check_dataset_offsets(dataset)

    # 3. Create dataloader and check batches
    dataloader = create_sequence_dataloader(
        feature_dirs=feature_dirs,
        annotation_file=args.train_json,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
    )

    check_dataloader_batch(dataloader)

    # 4. Analyze patterns
    analyze_offset_pattern(dataloader)

    print("\n" + "=" * 80)
    print("DEBUGGING COMPLETE")
    print("=" * 80)
    print("\nPossible issues to investigate:")
    print("1. If offsets are non-integers, check the FPS/sampling rate conversion")
    print(
        "2. If annotations are in seconds but features are sampled at a different rate"
    )
    print(
        "3. Check if there's a mismatch between annotation timestamps and feature extraction rate"
    )


if __name__ == "__main__":
    main()
