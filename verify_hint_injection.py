#!/usr/bin/env python3
"""
Sanity check script to verify that red dots were properly injected into highlight frames.
Uses CLIP's text-image similarity to check if frames with red dots are similar to "red circle" text.
"""

import json
import numpy as np
import torch
import clip
from pathlib import Path
import random
import matplotlib.pyplot as plt
import argparse
import os


def load_clip_model():
    """Load CLIP model for text encoding."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, device


def encode_text_queries(model, device, queries):
    """Encode text queries using CLIP."""
    text_tokens = clip.tokenize(queries).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        # Normalize
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features.cpu().numpy()


def compute_similarities(frame_features, text_features):
    """Compute cosine similarities between frame and text features."""
    # frame_features: [n_frames, 512]
    # text_features: [n_queries, 512]

    # Normalize frame features
    frame_features_norm = frame_features / np.linalg.norm(
        frame_features, axis=1, keepdims=True
    )

    # Compute similarities
    similarities = np.dot(frame_features_norm, text_features.T)
    return similarities


def plot_video_similarity(
    video_id, ground_truth, red_dot_similarity, save_dir="./plots"
):
    """
    Plot the similarity to "red dot" text vs ground truth labels for a video.

    Args:
        video_id: Video identifier
        ground_truth: Binary array of ground truth labels (1=highlight, 0=not)
        red_dot_similarity: Array of similarities to "red dot" text queries
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Timesteps
    timesteps = np.arange(len(ground_truth))

    # Plot 1: Ground truth labels
    ax1.fill_between(
        timesteps, 0, ground_truth, alpha=0.3, color="green", label="Highlight Segments"
    )
    ax1.set_ylabel("Ground Truth\n(1=Highlight)", fontsize=10)
    ax1.set_ylim(-0.1, 1.1)
    ax1.set_title(f"Video: {video_id} - Hint Injection Verification", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")

    # Plot 2: Similarity to "red dot" text
    ax2.plot(
        timesteps,
        red_dot_similarity,
        color="red",
        linewidth=1.5,
        label='Similarity to "red dot"',
    )

    # Add shaded regions for ground truth on similarity plot
    for i in range(len(ground_truth)):
        if ground_truth[i] == 1:
            ax2.axvspan(i - 0.5, i + 0.5, alpha=0.1, color="green")

    # Add horizontal line for average similarities
    highlight_mask = ground_truth == 1
    non_highlight_mask = ground_truth == 0

    if np.any(highlight_mask):
        avg_highlight = red_dot_similarity[highlight_mask].mean()
        ax2.axhline(
            y=avg_highlight,
            color="green",
            linestyle="--",
            alpha=0.7,
            label=f"Avg in highlights: {avg_highlight:.3f}",
        )

    if np.any(non_highlight_mask):
        avg_non_highlight = red_dot_similarity[non_highlight_mask].mean()
        ax2.axhline(
            y=avg_non_highlight,
            color="blue",
            linestyle="--",
            alpha=0.7,
            label=f"Avg outside: {avg_non_highlight:.3f}",
        )

    ax2.set_xlabel("Time (seconds)", fontsize=10)
    ax2.set_ylabel('Similarity to\n"red dot" text', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(save_dir, f"{video_id}_similarity.png")
    plt.savefig(plot_path, dpi=100, bbox_inches="tight")
    plt.close()

    return plot_path


def check_video_hints(
    video_id, features_dir, segments, model, device, verbose=False, plot=False
):
    """
    Check if red dots appear in the correct frames for a single video.

    Returns:
        dict with statistics about hint injection accuracy
    """
    # Load visual features
    feature_path = Path(features_dir) / f"{video_id}.npy"
    if not feature_path.exists():
        print(f"  Features not found for {video_id}")
        return None

    features = np.load(feature_path)
    n_frames = len(features)

    # Create ground truth labels (1 = should have red dot, 0 = should not)
    ground_truth = np.zeros(n_frames)
    for start, end in segments:
        start_idx = int(start)
        end_idx = int(end)
        ground_truth[start_idx:end_idx] = 1

    # Encode text queries
    queries = [
        "a red circle",
        "a red dot in the center",
        "a bright red spot",
        "a normal video frame",
        "a plain image",
    ]
    text_features = encode_text_queries(model, device, queries)

    # Compute similarities
    similarities = compute_similarities(features, text_features)

    # Get similarity to "red dot" queries (first 3) vs "normal" queries (last 2)
    red_dot_similarity = similarities[:, :3].max(
        axis=1
    )  # Max similarity to any red dot query
    normal_similarity = similarities[:, 3:].max(
        axis=1
    )  # Max similarity to any normal query

    # Predict: if more similar to red dot than normal, predict 1
    predictions = (red_dot_similarity > normal_similarity).astype(int)

    # Calculate statistics
    true_positives = np.sum((predictions == 1) & (ground_truth == 1))
    true_negatives = np.sum((predictions == 0) & (ground_truth == 0))
    false_positives = np.sum((predictions == 1) & (ground_truth == 0))
    false_negatives = np.sum((predictions == 0) & (ground_truth == 1))

    accuracy = (true_positives + true_negatives) / n_frames

    # Calculate precision and recall if there are positive examples
    if np.sum(ground_truth) > 0:
        recall = true_positives / np.sum(ground_truth)
        if (true_positives + false_positives) > 0:
            precision = true_positives / (true_positives + false_positives)
        else:
            precision = 0.0
    else:
        recall = 0.0
        precision = 0.0

    result = {
        "video_id": video_id,
        "n_frames": n_frames,
        "n_highlight_frames": int(np.sum(ground_truth)),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "true_positives": int(true_positives),
        "true_negatives": int(true_negatives),
        "false_positives": int(false_positives),
        "false_negatives": int(false_negatives),
        "avg_red_dot_sim_in_highlights": (
            float(red_dot_similarity[ground_truth == 1].mean())
            if np.sum(ground_truth) > 0
            else 0
        ),
        "avg_red_dot_sim_outside_highlights": (
            float(red_dot_similarity[ground_truth == 0].mean())
            if np.sum(ground_truth == 0) > 0
            else 0
        ),
        "similarity_gap": 0.0,  # Will calculate below
    }

    # Calculate similarity gap (how much more similar highlight frames are to "red dot")
    if (
        result["n_highlight_frames"] > 0
        and (n_frames - result["n_highlight_frames"]) > 0
    ):
        result["similarity_gap"] = (
            result["avg_red_dot_sim_in_highlights"]
            - result["avg_red_dot_sim_outside_highlights"]
        )

    if verbose:
        print(f"\n  Video: {video_id}")
        print(
            f"    Frames: {n_frames} total, {result['n_highlight_frames']} highlights"
        )
        print(f"    Accuracy: {accuracy:.2%}")
        print(f"    Precision: {precision:.2%}, Recall: {recall:.2%}")
        print(f"    Avg similarity to 'red dot':")
        print(f"      In highlights: {result['avg_red_dot_sim_in_highlights']:.3f}")
        print(
            f"      Outside highlights: {result['avg_red_dot_sim_outside_highlights']:.3f}"
        )
        print(f"      Gap: {result['similarity_gap']:.3f}")

    # Generate per-video plot if requested
    if plot:
        plot_path = plot_video_similarity(video_id, ground_truth, red_dot_similarity)
        result["plot_path"] = plot_path
        if verbose:
            print(f"    Plot saved to: {plot_path}")

    return result


def plot_similarity_distribution(results):
    """Plot the distribution of similarities for highlight vs non-highlight frames."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Collect all similarity gaps
    gaps = [r["similarity_gap"] for r in results if r is not None]

    # Plot 1: Histogram of similarity gaps
    axes[0].hist(gaps, bins=20, edgecolor="black")
    axes[0].set_xlabel("Similarity Gap (Highlight - Non-highlight)")
    axes[0].set_ylabel("Number of Videos")
    axes[0].set_title("Distribution of Red Dot Similarity Gaps")
    axes[0].axvline(x=0, color="r", linestyle="--", label="No difference")
    axes[0].legend()

    # Plot 2: Accuracy distribution
    accuracies = [r["accuracy"] for r in results if r is not None]
    axes[1].hist(accuracies, bins=20, edgecolor="black")
    axes[1].set_xlabel("Accuracy")
    axes[1].set_ylabel("Number of Videos")
    axes[1].set_title("Distribution of Hint Detection Accuracy")
    axes[1].axvline(x=0.5, color="r", linestyle="--", label="Random guess")
    axes[1].axvline(x=1.0, color="g", linestyle="--", label="Perfect")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("hint_injection_verification.png", dpi=150)
    print(f"\nPlot saved to hint_injection_verification.png")


def main():
    parser = argparse.ArgumentParser(
        description="Verify red dot hint injection in CLIP features"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset JSON file (e.g., test_train.json)",
    )
    parser.add_argument(
        "--features-dir",
        type=str,
        required=True,
        help="Path to directory with hint-injected features",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of random videos to check (default: 10)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed results for each video"
    )
    parser.add_argument(
        "--plot", action="store_true", help="Generate per-video similarity plots"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for sampling videos"
    )

    args = parser.parse_args()

    print("Loading CLIP model...")
    model, device = load_clip_model()

    print(f"Loading dataset from {args.dataset}...")
    with open(args.dataset, "r") as f:
        dataset = json.load(f)

    # Group by video ID and aggregate segments
    from collections import defaultdict

    video_segments = defaultdict(list)
    for entry in dataset:
        video_id = entry["youtube_id"]
        segments = entry.get("segments", [])
        video_segments[video_id].extend(segments)

    # Remove duplicates and sort segments for each video
    for video_id in video_segments:
        segments = video_segments[video_id]
        if segments:
            # Sort and deduplicate
            segments = list(set(tuple(s) for s in segments))
            segments.sort(key=lambda x: x[0])
            video_segments[video_id] = segments

    print(f"Found {len(video_segments)} unique videos in dataset")

    # Filter videos to only those with existing .npy files
    features_dir = Path(args.features_dir)
    available_videos = []
    for video_id in video_segments.keys():
        feature_path = features_dir / f"{video_id}.npy"
        if feature_path.exists():
            available_videos.append(video_id)

    print(f"Found {len(available_videos)} videos with extracted features")

    if len(available_videos) == 0:
        print("ERROR: No .npy files found in the features directory!")
        print(f"Please check: {args.features_dir}")
        return

    # Sample random videos from available ones
    random.seed(args.seed)
    sample_size = min(args.num_samples, len(available_videos))
    sampled_videos = random.sample(available_videos, sample_size)

    print(f"\nChecking {sample_size} random videos for hint injection...")
    print("=" * 60)

    results = []
    for i, video_id in enumerate(sampled_videos, 1):
        print(f"\n[{i}/{sample_size}] Checking {video_id}...")
        segments = video_segments[video_id]
        result = check_video_hints(
            video_id,
            args.features_dir,
            segments,
            model,
            device,
            verbose=args.verbose,
            plot=args.plot,
        )
        if result:
            results.append(result)

    # Print summary statistics
    if results:
        print("\n" + "=" * 60)
        print("SUMMARY STATISTICS")
        print("=" * 60)

        avg_accuracy = np.mean([r["accuracy"] for r in results])
        avg_precision = np.mean([r["precision"] for r in results])
        avg_recall = np.mean([r["recall"] for r in results])
        avg_gap = np.mean([r["similarity_gap"] for r in results])

        print(f"Videos analyzed: {len(results)}")
        print(f"Average accuracy: {avg_accuracy:.2%}")
        print(f"Average precision: {avg_precision:.2%}")
        print(f"Average recall: {avg_recall:.2%}")
        print(f"Average similarity gap: {avg_gap:.3f}")

        # Check if hint injection worked
        print("\n" + "=" * 60)
        if avg_accuracy > 0.8:
            print("✅ SUCCESS: Red dots appear to be correctly injected!")
            print(f"   High accuracy ({avg_accuracy:.1%}) indicates frames with hints")
            print(f"   are distinguishable from frames without hints.")
        elif avg_accuracy > 0.6:
            print("⚠️  WARNING: Partial hint injection detected")
            print(f"   Moderate accuracy ({avg_accuracy:.1%}) suggests some issues")
            print(f"   with hint injection or feature extraction.")
        else:
            print("❌ FAILURE: Red dots do not appear to be properly injected")
            print(f"   Low accuracy ({avg_accuracy:.1%}) indicates hints are not")
            print(f"   distinguishable in the extracted features.")

        if avg_gap > 0.1:
            print(f"\n✅ Similarity gap ({avg_gap:.3f}) confirms highlight frames")
            print(f"   are more similar to 'red dot' text queries.")
        else:
            print(f"\n⚠️  Low similarity gap ({avg_gap:.3f}) suggests weak")
            print(f"   correlation between hints and text queries.")

        # Generate plots
        plot_similarity_distribution(results)

        # Save detailed results
        output_file = "hint_verification_results.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to {output_file}")


if __name__ == "__main__":
    main()
