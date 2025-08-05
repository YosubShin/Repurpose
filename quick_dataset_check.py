#!/usr/bin/env python3
"""
Quick dataset inspection - hardcoded paths for convenience.
"""

import json
import os
import numpy as np
from collections import defaultdict, Counter


def load_and_check_features(feature_path):
    """Load a feature file and return its shape and status."""
    if not os.path.exists(feature_path):
        return None, "missing"

    try:
        features = np.load(feature_path)
        return features.shape, "ok"
    except Exception as e:
        return None, f"error: {e}"


def quick_check():
    """Quick check with hardcoded paths."""

    # Hardcoded paths - adjust these as needed
    data_root = "/home/yosubs/koa_scratch/repurpose/data"

    feature_dirs = {
        'audio': f"{data_root}/audio_pann_features",
        'visual': f"{data_root}/video_clip_features",
        'caption': f"{data_root}/caption_features"
    }

    splits = {
        'train': f"data/train.json",
        'val': f"data/val.json",
        'test': f"data/test.json"
    }

    print("QUICK DATASET CHECK")
    print("="*50)

    # Check if paths exist
    print("Checking paths...")
    all_paths_exist = True
    for split_name, ann_file in splits.items():
        exists = os.path.exists(ann_file)
        print(f"  {split_name} annotations: {'✓' if exists else '✗'} {ann_file}")
        if not exists:
            all_paths_exist = False

    for modality, feature_dir in feature_dirs.items():
        exists = os.path.exists(feature_dir)
        print(f"  {modality} features: {'✓' if exists else '✗'} {feature_dir}")
        if not exists:
            all_paths_exist = False

    if not all_paths_exist:
        print("\n❌ Some paths are missing. Please update the paths in the script.")
        return

    print("\n" + "="*50)

    # Quick analysis for each split
    for split_name, annotation_file in splits.items():
        print(f"\n{split_name.upper()} SPLIT:")

        with open(annotation_file, 'r') as f:
            annotations = json.load(f)

        total_videos = len(annotations)
        complete_videos = 0

        # Sample a few videos to check
        sample_videos = list(annotations.keys())[:min(10, len(annotations))]

        print(f"  Total videos: {total_videos}")
        print(f"  Checking sample of {len(sample_videos)} videos...")

        modality_available = Counter()
        sample_shapes = defaultdict(set)

        for video_id in sample_videos:
            video_complete = True
            video_shapes = {}

            for modality, feature_dir in feature_dirs.items():
                feature_path = os.path.join(feature_dir, f"{video_id}.npy")
                shape, status = load_and_check_features(feature_path)

                if status == "ok":
                    modality_available[modality] += 1
                    sample_shapes[modality].add(shape)
                    video_shapes[modality] = shape
                else:
                    video_complete = False

            if video_complete:
                complete_videos += 1
                # Check length consistency
                lengths = [shape[0] for shape in video_shapes.values()]
                if len(set(lengths)) > 1:
                    print(
                        f"    ⚠️  Length mismatch in {video_id}: {video_shapes}")

        print(
            f"  Sample complete videos: {complete_videos}/{len(sample_videos)}")

        for modality, count in modality_available.items():
            print(f"  {modality}: {count}/{len(sample_videos)} available")
            if modality in sample_shapes:
                shapes = list(sample_shapes[modality])
                print(
                    f"    Sample shapes: {shapes[:3]}{'...' if len(shapes) > 3 else ''}")

    print("\n" + "="*50)
    print("✅ Quick check complete!")
    print("Run 'python inspect_dataset.py' for detailed analysis.")


if __name__ == '__main__':
    quick_check()
