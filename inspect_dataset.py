#!/usr/bin/env python3
"""
Dataset inspection script for Repurpose dataset.
Analyzes feature availability, modality coverage, and length mismatches across train/val/test splits.
"""

import json
import os
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path
import argparse


def load_features(feature_path):
    """Load a feature file and return its shape."""
    if not os.path.exists(feature_path):
        return None

    try:
        features = np.load(feature_path)
        return features.shape
    except Exception as e:
        print(f"Error loading {feature_path}: {e}")
        return None


def analyze_split(split_name, annotation_file, feature_dirs):
    """Analyze a single dataset split."""
    print(f"\n{'='*60}")
    print(f"ANALYZING {split_name.upper()} SPLIT")
    print(f"{'='*60}")

    # Load annotations
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    # Handle both list and dict formats
    if isinstance(annotations, list):
        # Convert list to dict format using youtube_id field
        annotations_dict = {item['youtube_id']: item for item in annotations}
    else:
        annotations_dict = annotations

    total_videos = len(annotations_dict)
    print(f"Total videos in {split_name}: {total_videos}")

    # Track statistics
    modality_counts = Counter()
    feature_availability = defaultdict(list)
    length_mismatches = []
    missing_features = defaultdict(int)
    feature_shapes = defaultdict(list)

    videos_with_all_modalities = 0
    videos_with_issues = []

    for video_id, ann in annotations_dict.items():
        video_modalities = []
        video_shapes = {}
        video_missing = []

        # Check each modality
        for modality, feature_dir in feature_dirs.items():
            feature_path = os.path.join(feature_dir, f"{video_id}.npy")
            shape = load_features(feature_path)

            if shape is not None:
                video_modalities.append(modality)
                video_shapes[modality] = shape
                feature_shapes[modality].append(shape)
                modality_counts[modality] += 1
            else:
                video_missing.append(modality)
                missing_features[modality] += 1

        # Record feature availability for this video
        feature_availability[tuple(sorted(video_modalities))].append(video_id)

        # Check if all modalities are present
        if len(video_modalities) == len(feature_dirs):
            videos_with_all_modalities += 1

            # Check for length mismatches (assuming first dimension is time)
            lengths = [shape[0] for shape in video_shapes.values()]
            if len(set(lengths)) > 1:
                length_mismatches.append({
                    'video_id': video_id,
                    'lengths': {mod: shape[0] for mod, shape in video_shapes.items()},
                    'shapes': video_shapes
                })
        else:
            videos_with_issues.append({
                'video_id': video_id,
                'available': video_modalities,
                'missing': video_missing
            })

    # Print summary statistics
    print(f"\nFEATURE AVAILABILITY:")
    print(
        f"Videos with all modalities: {videos_with_all_modalities}/{total_videos} ({100*videos_with_all_modalities/total_videos:.1f}%)")

    print(f"\nMODALITY COVERAGE:")
    for modality, count in modality_counts.items():
        percentage = 100 * count / total_videos
        print(f"  {modality}: {count}/{total_videos} videos ({percentage:.1f}%)")

    print(f"\nMISSING FEATURES:")
    for modality, count in missing_features.items():
        percentage = 100 * count / total_videos
        print(f"  {modality}: {count} missing ({percentage:.1f}%)")

    print(f"\nFEATURE COMBINATIONS:")
    for combo, video_list in sorted(feature_availability.items(), key=lambda x: -len(x[1])):
        count = len(video_list)
        percentage = 100 * count / total_videos
        combo_str = "+".join(combo) if combo else "NO_FEATURES"
        print(f"  {combo_str}: {count} videos ({percentage:.1f}%)")

    # Analyze feature shapes
    print(f"\nFEATURE SHAPES:")
    for modality in feature_dirs.keys():
        if modality in feature_shapes:
            shapes = feature_shapes[modality]
            unique_shapes = list(set(shapes))
            print(f"  {modality}:")
            print(f"    Total files: {len(shapes)}")
            print(f"    Unique shapes: {len(unique_shapes)}")

            if len(unique_shapes) <= 5:
                for shape in sorted(unique_shapes):
                    count = shapes.count(shape)
                    percentage = 100 * count / len(shapes)
                    print(f"      {shape}: {count} files ({percentage:.1f}%)")
            else:
                # Show shape statistics
                lengths = [shape[0]
                           for shape in shapes]  # First dimension (time)
                print(
                    f"      Length stats (first dim): min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.1f}, std={np.std(lengths):.1f}")

    # Report length mismatches
    print(f"\nLENGTH MISMATCHES:")
    if length_mismatches:
        print(f"Found {len(length_mismatches)} videos with length mismatches:")
        for i, mismatch in enumerate(length_mismatches[:10]):  # Show first 10
            print(f"  {i+1}. {mismatch['video_id']}: {mismatch['lengths']}")
        if len(length_mismatches) > 10:
            print(f"  ... and {len(length_mismatches) - 10} more")
    else:
        print("No length mismatches found!")

    # Report videos with missing features
    print(f"\nVIDEOS WITH MISSING FEATURES:")
    if videos_with_issues:
        print(f"Found {len(videos_with_issues)} videos with missing features:")
        for i, issue in enumerate(videos_with_issues[:10]):  # Show first 10
            print(
                f"  {i+1}. {issue['video_id']}: missing {issue['missing']}, has {issue['available']}")
        if len(videos_with_issues) > 10:
            print(f"  ... and {len(videos_with_issues) - 10} more")
    else:
        print("All videos have all modalities!")

    return {
        'total_videos': total_videos,
        'videos_with_all_modalities': videos_with_all_modalities,
        'modality_counts': dict(modality_counts),
        'missing_features': dict(missing_features),
        'length_mismatches': length_mismatches,
        'videos_with_issues': videos_with_issues,
        'feature_shapes': {k: list(v) for k, v in feature_shapes.items()}
    }


def main():
    parser = argparse.ArgumentParser(
        description='Inspect Repurpose dataset features')
    parser.add_argument('--data-root', type=str, default='/home/yosubs/koa_scratch/repurpose/data',
                        help='Root directory containing features and annotations')
    parser.add_argument('--output', type=str,
                        help='Output file to save detailed results (optional)')
    args = parser.parse_args()

    # Define paths
    data_root = Path(args.data_root)

    # Feature directories
    feature_dirs = {
        'audio': str(data_root / 'audio_pann_features'),
        'visual': str(data_root / 'video_clip_features'),
        'caption': str(data_root / 'caption_features')
    }

    # Annotation files
    splits = {
        'train': str('data/train.json'),
        'val': str('data/val.json'),
        'test': str('data/test.json')
    }

    print("REPURPOSE DATASET INSPECTION")
    print(f"Data root: {data_root}")
    print(f"Feature directories: {feature_dirs}")

    # Verify paths exist
    missing_paths = []
    for split_name, ann_file in splits.items():
        if not os.path.exists(ann_file):
            missing_paths.append(f"{split_name} annotations: {ann_file}")

    for modality, feature_dir in feature_dirs.items():
        if not os.path.exists(feature_dir):
            missing_paths.append(f"{modality} features: {feature_dir}")

    if missing_paths:
        print("\nERROR: Missing paths:")
        for path in missing_paths:
            print(f"  - {path}")
        return

    # Analyze each split
    all_results = {}
    for split_name, annotation_file in splits.items():
        if os.path.exists(annotation_file):
            results = analyze_split(split_name, annotation_file, feature_dirs)
            all_results[split_name] = results
        else:
            print(
                f"\nSkipping {split_name} - annotation file not found: {annotation_file}")

    # Overall summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")

    total_all_videos = sum(r['total_videos'] for r in all_results.values())
    total_complete_videos = sum(r['videos_with_all_modalities']
                                for r in all_results.values())

    print(f"Total videos across all splits: {total_all_videos}")
    print(
        f"Videos with all modalities: {total_complete_videos}/{total_all_videos} ({100*total_complete_videos/total_all_videos:.1f}%)")

    print(f"\nPer-split summary:")
    for split_name, results in all_results.items():
        complete_pct = 100 * \
            results['videos_with_all_modalities'] / results['total_videos']
        print(
            f"  {split_name}: {results['videos_with_all_modalities']}/{results['total_videos']} complete ({complete_pct:.1f}%)")

    # Save detailed results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nDetailed results saved to: {args.output}")


if __name__ == '__main__':
    main()
