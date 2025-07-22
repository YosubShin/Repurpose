#!/usr/bin/env python3
"""
Script to inspect .npy files for video IDs in JSON files
Checks the shapes of video_clip_features, audio_pann_features, and caption_features
"""

import json
import numpy as np
import os
from pathlib import Path
import sys
import argparse


def load_test_data(test_json_path):
    """Load video IDs from test.json"""
    with open(test_json_path, 'r') as f:
        data = json.load(f)

    # Extract unique video IDs
    video_ids = list(set([item['youtube_id'] for item in data]))
    return video_ids


def inspect_features_for_video(video_id, data_dir):
    """Inspect feature files for a specific video ID"""
    results = {
        'video_id': video_id,
        'video_clip_features': None,
        'audio_pann_features': None,
        'caption_features': None,
        'missing_files': [],
        'length_mismatches': []
    }

    # Define feature file paths
    feature_paths = {
        'video_clip_features': f"{data_dir}/video_clip_features/{video_id}.npy",
        'audio_pann_features': f"{data_dir}/audio_pann_features/{video_id}.npy",
        'caption_features': f"{data_dir}/caption_features/{video_id}.npy"
    }

    # Check each feature file
    for feature_name, file_path in feature_paths.items():
        if os.path.exists(file_path):
            try:
                # Load the numpy array
                data = np.load(file_path)
                results[feature_name] = {
                    'shape': data.shape,
                    'dtype': str(data.dtype),
                    'file_size_mb': os.path.getsize(file_path) / (1024 * 1024),
                    'length': data.shape[0] if len(data.shape) > 0 else 0
                }
            except Exception as e:
                results[feature_name] = {
                    'error': str(e)
                }
        else:
            results['missing_files'].append(feature_name)

    # Check for length mismatches between features
    available_features = {name: results[name] for name in ['video_clip_features', 'audio_pann_features', 'caption_features']
                          if results[name] is not None and 'length' in results[name]}

    if len(available_features) >= 2:
        lengths = [(name, data['length'])
                   for name, data in available_features.items()]
        lengths.sort(key=lambda x: x[1])  # Sort by length

        min_length = lengths[0][1]
        max_length = lengths[-1][1]

        if max_length > 0:
            length_ratio = max_length / min_length
            length_diff = max_length - min_length

            # Flag significant mismatches
            if length_ratio > 1.1 or length_diff > 10:  # More than 10% difference or more than 10 frames difference
                results['length_mismatches'] = {
                    'features': lengths,
                    'min_length': min_length,
                    'max_length': max_length,
                    'length_ratio': length_ratio,
                    'length_diff': length_diff
                }

    return results


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Inspect .npy feature files for video IDs in JSON files')
    parser.add_argument('--json_file', type=str, default="data/test.json",
                        help='Path to JSON file containing video IDs (default: data/test.json)')
    parser.add_argument('--data_dir', type=str, default="/data/repurpose/data",
                        help='Path to data directory containing feature files (default: /data/repurpose/data)')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Path to output file to save results (optional)')

    args = parser.parse_args()

    # Configuration
    json_file_path = args.json_file
    data_dir = args.data_dir
    output_file = args.output_file

    # Check if files exist
    if not os.path.exists(json_file_path):
        print(f"Error: {json_file_path} not found")
        sys.exit(1)

    if not os.path.exists(data_dir):
        print(f"Error: {data_dir} not found")
        sys.exit(1)

    # Load video IDs from JSON file
    print(f"Loading video IDs from {json_file_path}...")
    video_ids = load_test_data(json_file_path)
    print(f"Found {len(video_ids)} unique video IDs")

    # Inspect features for each video
    print("\nInspecting feature files...")
    print("=" * 80)

    summary = {
        'total_videos': len(video_ids),
        'videos_with_all_features': 0,
        'videos_with_some_features': 0,
        'videos_with_no_features': 0,
        'videos_with_length_mismatches': 0,
        'feature_shapes': {
            'video_clip_features': set(),
            'audio_pann_features': set(),
            'caption_features': set()
        },
        'length_mismatches': [],
        'missing_by_modality': {
            'video_clip_features': 0,
            'audio_pann_features': 0,
            'caption_features': 0
        },
        'missing_combinations': {},
        'examples_by_missing_type': {
            'missing_video_only': [],
            'missing_audio_only': [],
            'missing_caption_only': [],
            'missing_video_audio': [],
            'missing_video_caption': [],
            'missing_audio_caption': [],
            'missing_all': []
        }
    }

    for i, video_id in enumerate(video_ids, 1):
        print(f"\n[{i}/{len(video_ids)}] Video ID: {video_id}")
        print("-" * 50)

        results = inspect_features_for_video(video_id, data_dir)

        # Print results
        for feature_name in ['video_clip_features', 'audio_pann_features', 'caption_features']:
            if results[feature_name] is not None:
                if 'error' in results[feature_name]:
                    print(
                        f"  {feature_name}: ERROR - {results[feature_name]['error']}")
                else:
                    shape = results[feature_name]['shape']
                    dtype = results[feature_name]['dtype']
                    size_mb = results[feature_name]['file_size_mb']
                    print(
                        f"  {feature_name}: shape={shape}, dtype={dtype}, size={size_mb:.2f}MB")
                    summary['feature_shapes'][feature_name].add(shape)
            else:
                print(f"  {feature_name}: MISSING")

        if results['missing_files']:
            print(f"  Missing files: {', '.join(results['missing_files'])}")

        # Check for length mismatches
        if results['length_mismatches']:
            mismatch_info = results['length_mismatches']
            print(
                f"  LENGTH MISMATCH: ratio={mismatch_info['length_ratio']:.2f}, diff={mismatch_info['length_diff']}")
            print(f"    Features: {mismatch_info['features']}")
            summary['videos_with_length_mismatches'] += 1
            summary['length_mismatches'].append({
                'video_id': video_id,
                'mismatch_info': mismatch_info
            })

        # Update summary
        missing_count = len(results['missing_files'])
        if missing_count == 0:
            summary['videos_with_all_features'] += 1
        elif missing_count < 3:
            summary['videos_with_some_features'] += 1
        else:
            summary['videos_with_no_features'] += 1

        # Track missing by modality
        for missing_feature in results['missing_files']:
            summary['missing_by_modality'][missing_feature] += 1

        # Track missing combinations and collect examples
        if results['missing_files']:
            missing_key = tuple(sorted(results['missing_files']))
            summary['missing_combinations'][missing_key] = summary['missing_combinations'].get(
                missing_key, 0) + 1

            # Collect examples (limit to 5 per type)
            example_entry = {'video_id': video_id,
                             'missing': results['missing_files']}

            if missing_count == 1:
                if 'video_clip_features' in results['missing_files']:
                    if len(summary['examples_by_missing_type']['missing_video_only']) < 5:
                        summary['examples_by_missing_type']['missing_video_only'].append(
                            example_entry)
                elif 'audio_pann_features' in results['missing_files']:
                    if len(summary['examples_by_missing_type']['missing_audio_only']) < 5:
                        summary['examples_by_missing_type']['missing_audio_only'].append(
                            example_entry)
                elif 'caption_features' in results['missing_files']:
                    if len(summary['examples_by_missing_type']['missing_caption_only']) < 5:
                        summary['examples_by_missing_type']['missing_caption_only'].append(
                            example_entry)
            elif missing_count == 2:
                if 'video_clip_features' in results['missing_files'] and 'audio_pann_features' in results['missing_files']:
                    if len(summary['examples_by_missing_type']['missing_video_audio']) < 5:
                        summary['examples_by_missing_type']['missing_video_audio'].append(
                            example_entry)
                elif 'video_clip_features' in results['missing_files'] and 'caption_features' in results['missing_files']:
                    if len(summary['examples_by_missing_type']['missing_video_caption']) < 5:
                        summary['examples_by_missing_type']['missing_video_caption'].append(
                            example_entry)
                elif 'audio_pann_features' in results['missing_files'] and 'caption_features' in results['missing_files']:
                    if len(summary['examples_by_missing_type']['missing_audio_caption']) < 5:
                        summary['examples_by_missing_type']['missing_audio_caption'].append(
                            example_entry)
            elif missing_count == 3:
                if len(summary['examples_by_missing_type']['missing_all']) < 5:
                    summary['examples_by_missing_type']['missing_all'].append(
                        example_entry)

        # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total videos: {summary['total_videos']}")
    print(f"Videos with all features: {summary['videos_with_all_features']}")
    print(f"Videos with some features: {summary['videos_with_some_features']}")
    print(f"Videos with no features: {summary['videos_with_no_features']}")
    print(
        f"Videos with length mismatches: {summary['videos_with_length_mismatches']}")

    print("\nMissing features by modality:")
    for feature_name, count in summary['missing_by_modality'].items():
        percentage = (count / summary['total_videos']) * 100
        print(f"  {feature_name}: {count} ({percentage:.1f}%)")

    print("\nMissing feature combinations:")
    for missing_combo, count in sorted(summary['missing_combinations'].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / summary['total_videos']) * 100
        combo_str = ', '.join(missing_combo)
        print(f"  Missing [{combo_str}]: {count} videos ({percentage:.1f}%)")

    print("\nFeature shapes found:")
    for feature_name, shapes in summary['feature_shapes'].items():
        if shapes:
            print(f"  {feature_name}:")
            for shape in sorted(shapes):
                print(f"    {shape}")
        else:
            print(f"  {feature_name}: No files found")

    # Print length mismatches details
    if summary['length_mismatches']:
        print("\n" + "=" * 80)
        print("LENGTH MISMATCHES DETAILS")
        print("=" * 80)
        print(
            f"Found {len(summary['length_mismatches'])} videos with significant length mismatches:")
        print()

        for mismatch in summary['length_mismatches']:
            video_id = mismatch['video_id']
            info = mismatch['mismatch_info']
            print(f"Video ID: {video_id}")
            print(f"  Length ratio: {info['length_ratio']:.2f}")
            print(f"  Length difference: {info['length_diff']}")
            print(f"  Features: {info['features']}")
            print()

    # Print examples of videos with missing features
    print("\n" + "=" * 80)
    print("EXAMPLES OF VIDEOS WITH MISSING FEATURES")
    print("=" * 80)

    example_categories = [
        ('missing_video_only', 'Missing Video Features Only'),
        ('missing_audio_only', 'Missing Audio Features Only'),
        ('missing_caption_only', 'Missing Caption Features Only'),
        ('missing_video_audio', 'Missing Video + Audio Features'),
        ('missing_video_caption', 'Missing Video + Caption Features'),
        ('missing_audio_caption', 'Missing Audio + Caption Features'),
        ('missing_all', 'Missing All Features')
    ]

    for category_key, category_name in example_categories:
        examples = summary['examples_by_missing_type'][category_key]
        if examples:
            print(
                f"\n{category_name} ({len(examples)} examples shown, may have more):")
            for example in examples:
                print(f"  - {example['video_id']}")

    # Print videos with all features
    print("\n" + "=" * 80)
    print("VIDEOS WITH ALL FEATURES")
    print("=" * 80)

    all_features_videos = []
    for i, video_id in enumerate(video_ids, 1):
        results = inspect_features_for_video(video_id, data_dir)
        if len(results['missing_files']) == 0:
            all_features_videos.append((video_id, results))

    print(f"Found {len(all_features_videos)} videos with all features:")
    print()

    # Collect output for potential file writing
    output_lines = []

    # Add summary statistics to output
    output_lines.append("SUMMARY STATISTICS")
    output_lines.append("=" * 50)
    output_lines.append(f"Total videos: {summary['total_videos']}")
    output_lines.append(
        f"Videos with all features: {summary['videos_with_all_features']}")
    output_lines.append(
        f"Videos with some features: {summary['videos_with_some_features']}")
    output_lines.append(
        f"Videos with no features: {summary['videos_with_no_features']}")
    output_lines.append(
        f"Videos with length mismatches: {summary['videos_with_length_mismatches']}")
    output_lines.append("")

    # Add missing features by modality
    output_lines.append("MISSING FEATURES BY MODALITY")
    output_lines.append("=" * 50)
    for feature_name, count in summary['missing_by_modality'].items():
        percentage = (count / summary['total_videos']) * 100
        output_lines.append(f"{feature_name}: {count} ({percentage:.1f}%)")
    output_lines.append("")

    # Add missing feature combinations
    output_lines.append("MISSING FEATURE COMBINATIONS")
    output_lines.append("=" * 50)
    for missing_combo, count in sorted(summary['missing_combinations'].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / summary['total_videos']) * 100
        combo_str = ', '.join(missing_combo)
        output_lines.append(
            f"Missing [{combo_str}]: {count} videos ({percentage:.1f}%)")
    output_lines.append("")

    # Add length mismatches to output
    if summary['length_mismatches']:
        output_lines.append("LENGTH MISMATCHES")
        output_lines.append("=" * 50)
        for mismatch in summary['length_mismatches']:
            video_id = mismatch['video_id']
            info = mismatch['mismatch_info']
            output_lines.append(f"Video ID: {video_id}")
            output_lines.append(f"  Length ratio: {info['length_ratio']:.2f}")
            output_lines.append(f"  Length difference: {info['length_diff']}")
            output_lines.append(f"  Features: {info['features']}")
            output_lines.append("")

    # Add examples of videos with missing features
    output_lines.append("EXAMPLES OF VIDEOS WITH MISSING FEATURES")
    output_lines.append("=" * 50)

    for category_key, category_name in example_categories:
        examples = summary['examples_by_missing_type'][category_key]
        if examples:
            output_lines.append(
                f"\n{category_name} ({len(examples)} examples shown, may have more):")
            for example in examples:
                output_lines.append(f"  - {example['video_id']}")
    output_lines.append("")

    # Add videos with all features
    output_lines.append("VIDEOS WITH ALL FEATURES")
    output_lines.append("=" * 50)

    for video_id, results in all_features_videos:
        video_output = f"Video ID: {video_id}"
        print(video_output)
        output_lines.append(video_output)

        for feature_name in ['video_clip_features', 'audio_pann_features', 'caption_features']:
            if results[feature_name] and 'shape' in results[feature_name]:
                shape = results[feature_name]['shape']
                dtype = results[feature_name]['dtype']
                size_mb = results[feature_name]['file_size_mb']
                feature_output = f"  {feature_name}: shape={shape}, dtype={dtype}, size={size_mb:.2f}MB"
                print(feature_output)
                output_lines.append(feature_output)
        print()
        output_lines.append("")

    # Save to file if requested
    if output_file:
        try:
            with open(output_file, 'w') as f:
                f.write('\n'.join(output_lines))
            print(f"Results saved to {output_file}")
        except Exception as e:
            print(f"Error saving to file: {e}")


if __name__ == "__main__":
    main()
