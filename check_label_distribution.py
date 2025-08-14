#!/usr/bin/env python3
"""Check the actual distribution of labels in the dataset."""

import json
import numpy as np


def check_label_distribution(annotation_file):
    with open(annotation_file, "r") as f:
        data = json.load(f)

    all_segments = []
    for video_id, video_data in data.items():
        segments = video_data.get("segments", [])
        all_segments.extend(segments)

    if all_segments:
        # Calculate percentage of positive segments
        positive_ratio = len(all_segments) / sum(
            video_data.get("num_frames", 0) for video_data in data.values()
        )
        print(f"Annotation file: {annotation_file}")
        print(f"Number of positive segments: {len(all_segments)}")
        print(f"Estimated positive ratio: {positive_ratio:.3f}")
        print(f"Recommended focal_alpha: {1 - positive_ratio:.3f}")
    else:
        print(f"No segments found in {annotation_file}")


if __name__ == "__main__":
    # Check both train and val distributions
    check_label_distribution("/home/yosubs/co/Repurpose/data/test.json")
    check_label_distribution("/home/yosubs/co/Repurpose/data/val.json")
