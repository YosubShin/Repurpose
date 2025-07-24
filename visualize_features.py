#!/usr/bin/env python3
"""
Feature Visualization Script for Debugging Data Sanity

This script visualizes text features and labels to check if the regression loss plateau
is due to poor feature-label correlation. It creates 3D plots with:
- X-axis: time (timesteps)
- Y-Z axis: PCA/UMAP projections of features
- Points marked by classification labels (c_t=1 for highlights)

Expected: If data is healthy, c_t=1 points should form distinct traces different from c_t=0
If corrupted: No visible distinction between positive and negative points
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import torch
from dataset.RepurposeClip import RepurposeClipTest
import argparse
import yaml
from datetime import datetime


class FeatureVisualizer:
    def __init__(self, config_path, target_video_ids=None):
        """
        Initialize the feature visualizer

        Args:
            config_path: Path to the config file
            target_video_ids: List of specific video IDs to analyze
        """
        self.config = self.load_config(config_path)
        self.target_video_ids = target_video_ids or [
            "AxwsdxOqQaU", "sYBddyIEQjA", "q5ycx357EoM",
            "tVUDJs6_A-0", "ceolO7WB8iA"
        ]

        # Create output directory
        self.output_dir = f"feature_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.output_dir, exist_ok=True)

        print(f"Output directory: {self.output_dir}")
        print(f"Target video IDs: {self.target_video_ids}")

    def load_config(self, config_path):
        """Load configuration file"""
        with open(config_path, 'r') as f:
            return yaml.load(f, Loader=yaml.FullLoader)

    def load_dataset_samples(self):
        """Load specific video samples from the dataset"""
        print("Loading dataset samples...")

        # Create test dataset
        dataset = RepurposeClipTest(**self.config['test_dataset'])

        # Find samples with target video IDs
        target_samples = []
        for idx, label_item in enumerate(dataset.label):
            video_id = label_item['youtube_id']
            if video_id in self.target_video_ids:
                try:
                    sample = dataset[idx]
                    target_samples.append({
                        'video_id': video_id,
                        # Only text features
                        'features': sample['feats']['text'],
                        'labels': sample['labels'],
                        'segments': sample['segments'],
                        'duration': sample['duration'],
                        'gt_segments': sample['gt_segments']
                    })
                    print(
                        f"Loaded {video_id}: {sample['features'].shape[0]} timesteps")
                except Exception as e:
                    print(f"Failed to load {video_id}: {e}")

        if not target_samples:
            print("No target samples found!")
            return []

        print(f"Successfully loaded {len(target_samples)} samples")
        return target_samples

    def analyze_sample_statistics(self, samples):
        """Analyze basic statistics of the samples"""
        print("\n" + "="*50)
        print("SAMPLE STATISTICS")
        print("="*50)

        stats_data = []

        for sample in samples:
            video_id = sample['video_id']
            features = sample['features']
            labels = np.array(sample['labels'])
            segments = np.array(sample['segments'])

            # Basic stats
            seq_len = len(labels)
            num_positive = np.sum(labels == 1)
            num_negative = np.sum(labels == 0)
            positive_ratio = num_positive / seq_len if seq_len > 0 else 0

            # Feature stats
            feature_mean = np.mean(features, axis=0)
            feature_std = np.std(features, axis=0)

            # Segment stats (for positive points)
            positive_segments = segments[labels == 1]
            if len(positive_segments) > 0:
                avg_left_offset = np.mean(positive_segments[:, 0])
                avg_right_offset = np.mean(positive_segments[:, 1])
                offset_std = np.std(positive_segments, axis=0)
            else:
                avg_left_offset = avg_right_offset = 0
                offset_std = np.array([0, 0])

            stats = {
                'video_id': video_id,
                'seq_len': seq_len,
                'num_positive': num_positive,
                'num_negative': num_negative,
                'positive_ratio': positive_ratio,
                'feature_dim': features.shape[1],
                'feature_mean_norm': np.linalg.norm(feature_mean),
                'feature_std_mean': np.mean(feature_std),
                'avg_left_offset': avg_left_offset,
                'avg_right_offset': avg_right_offset,
                'offset_std': offset_std
            }

            stats_data.append(stats)

            print(f"\n{video_id}:")
            print(f"  Length: {seq_len} timesteps")
            print(
                f"  Positive/Negative: {num_positive}/{num_negative} ({positive_ratio:.2%})")
            print(f"  Feature dim: {features.shape[1]}")
            print(f"  Feature mean norm: {stats['feature_mean_norm']:.4f}")
            print(f"  Feature std mean: {stats['feature_std_mean']:.4f}")
            if num_positive > 0:
                print(
                    f"  Avg offsets: left={avg_left_offset:.2f}, right={avg_right_offset:.2f}")
                print(
                    f"  Offset std: left={offset_std[0]:.2f}, right={offset_std[1]:.2f}")

        return stats_data

    def create_dimensionality_reductions(self, all_features, method='pca'):
        """Create 2D projections of features using PCA or UMAP"""
        print(f"\nCreating {method.upper()} projection...")

        if method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
        elif method == 'umap':
            reducer = umap.UMAP(n_components=2, random_state=42,
                                n_neighbors=15, min_dist=0.1)
        elif method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        else:
            raise ValueError(f"Unknown method: {method}")

        projections = reducer.fit_transform(all_features)

        print(
            f"Explained variance ratio (if PCA): {getattr(reducer, 'explained_variance_ratio_', 'N/A')}")

        return projections, reducer

    def plot_3d_feature_analysis(self, samples, method='pca'):
        """Create 3D plots with time on X-axis and feature projections on Y-Z"""
        print(f"\nCreating 3D visualization with {method.upper()}...")

        # Collect all features for global dimensionality reduction
        all_features = []
        sample_boundaries = []
        current_idx = 0

        for sample in samples:
            features = sample['features']
            all_features.append(features)
            sample_boundaries.append(
                (current_idx, current_idx + len(features)))
            current_idx += len(features)

        all_features = np.vstack(all_features)
        print(f"Total features: {all_features.shape}")

        # Create 2D projection
        projections, reducer = self.create_dimensionality_reductions(
            all_features, method)

        # Create plots
        fig = plt.figure(figsize=(20, 12))

        # Plot each video in a separate subplot
        n_videos = len(samples)
        cols = min(3, n_videos)
        rows = (n_videos + cols - 1) // cols

        for idx, sample in enumerate(samples):
            video_id = sample['video_id']
            labels = np.array(sample['labels'])
            start_idx, end_idx = sample_boundaries[idx]

            # Get projections for this sample
            sample_projections = projections[start_idx:end_idx]
            time_steps = np.arange(len(labels))

            # Create subplot
            ax = fig.add_subplot(rows, cols, idx + 1, projection='3d')

            # Plot negative points (c_t = 0)
            negative_mask = labels == 0
            if np.any(negative_mask):
                ax.scatter(time_steps[negative_mask],
                           sample_projections[negative_mask, 0],
                           sample_projections[negative_mask, 1],
                           c='lightblue', alpha=0.6, s=20, label='c_t=0 (background)')

            # Plot positive points (c_t = 1)
            positive_mask = labels == 1
            if np.any(positive_mask):
                ax.scatter(time_steps[positive_mask],
                           sample_projections[positive_mask, 0],
                           sample_projections[positive_mask, 1],
                           c='red', alpha=0.8, s=40, label='c_t=1 (highlights)', marker='^')

            # Customize plot
            ax.set_xlabel('Time (timesteps)')
            ax.set_ylabel(f'{method.upper()} Component 1')
            ax.set_zlabel(f'{method.upper()} Component 2')
            ax.set_title(
                f'{video_id}\n({np.sum(positive_mask)}/{len(labels)} positive)')
            ax.legend()

            # Add connecting lines for positive points to show traces
            if np.sum(positive_mask) > 1:
                pos_times = time_steps[positive_mask]
                pos_proj = sample_projections[positive_mask]
                ax.plot(pos_times, pos_proj[:, 0], pos_proj[:, 1],
                        'r--', alpha=0.5, linewidth=1)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'3d_feature_analysis_{method}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        return projections, reducer

    def plot_2d_projections(self, samples, method='pca'):
        """Create 2D projections for easier analysis"""
        print(f"\nCreating 2D projection plots with {method.upper()}...")

        # Collect all features
        all_features = []
        all_labels = []
        video_labels = []

        for sample in samples:
            features = sample['features']
            labels = sample['labels']
            video_id = sample['video_id']

            all_features.append(features)
            all_labels.extend(labels)
            video_labels.extend([video_id] * len(labels))

        all_features = np.vstack(all_features)
        all_labels = np.array(all_labels)

        # Create projection
        projections, reducer = self.create_dimensionality_reductions(
            all_features, method)

        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: All points colored by label
        ax = axes[0, 0]
        scatter = ax.scatter(projections[:, 0], projections[:, 1],
                             c=all_labels, cmap='RdYlBu_r', alpha=0.6, s=10)
        plt.colorbar(scatter, ax=ax, label='Label (c_t)')
        ax.set_title(f'All Points Colored by Label ({method.upper()})')
        ax.set_xlabel(f'{method.upper()} Component 1')
        ax.set_ylabel(f'{method.upper()} Component 2')

        # Plot 2: Separate positive and negative
        ax = axes[0, 1]
        negative_mask = all_labels == 0
        positive_mask = all_labels == 1

        if np.any(negative_mask):
            ax.scatter(projections[negative_mask, 0], projections[negative_mask, 1],
                       c='lightblue', alpha=0.5, s=8, label=f'c_t=0 ({np.sum(negative_mask)})')
        if np.any(positive_mask):
            ax.scatter(projections[positive_mask, 0], projections[positive_mask, 1],
                       c='red', alpha=0.8, s=20, label=f'c_t=1 ({np.sum(positive_mask)})')

        ax.set_title(f'Positive vs Negative Points ({method.upper()})')
        ax.set_xlabel(f'{method.upper()} Component 1')
        ax.set_ylabel(f'{method.upper()} Component 2')
        ax.legend()

        # Plot 3: Colored by video
        ax = axes[1, 0]
        unique_videos = list(set(video_labels))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_videos)))

        for i, video_id in enumerate(unique_videos):
            video_mask = np.array(video_labels) == video_id
            ax.scatter(projections[video_mask, 0], projections[video_mask, 1],
                       c=[colors[i]], alpha=0.7, s=10, label=video_id)

        ax.set_title(f'Points Colored by Video ({method.upper()})')
        ax.set_xlabel(f'{method.upper()} Component 1')
        ax.set_ylabel(f'{method.upper()} Component 2')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Plot 4: Density plot of positive points
        ax = axes[1, 1]
        if np.any(positive_mask):
            ax.scatter(projections[negative_mask, 0], projections[negative_mask, 1],
                       c='lightgray', alpha=0.3, s=5, label='c_t=0')
            scatter = ax.scatter(projections[positive_mask, 0], projections[positive_mask, 1],
                                 c=np.arange(np.sum(positive_mask)), cmap='viridis',
                                 alpha=0.8, s=20, label='c_t=1')
            plt.colorbar(scatter, ax=ax, label='Positive Point Index')

        ax.set_title(f'Positive Points Distribution ({method.upper()})')
        ax.set_xlabel(f'{method.upper()} Component 1')
        ax.set_ylabel(f'{method.upper()} Component 2')
        ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'2d_projections_{method}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def analyze_feature_label_correlation(self, samples):
        """Analyze correlation between features and labels"""
        print("\n" + "="*50)
        print("FEATURE-LABEL CORRELATION ANALYSIS")
        print("="*50)

        correlation_results = []

        for sample in samples:
            video_id = sample['video_id']
            features = sample['features']
            labels = np.array(sample['labels'])

            # Calculate correlation between each feature dimension and labels
            correlations = []
            for dim in range(features.shape[1]):
                corr = np.corrcoef(features[:, dim], labels)[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))

            if correlations:
                max_corr = np.max(correlations)
                mean_corr = np.mean(correlations)

                correlation_results.append({
                    'video_id': video_id,
                    'max_correlation': max_corr,
                    'mean_correlation': mean_corr,
                    'num_dims': len(correlations)
                })

                print(f"{video_id}:")
                print(f"  Max correlation: {max_corr:.4f}")
                print(f"  Mean correlation: {mean_corr:.4f}")
                print(f"  Feature dims: {len(correlations)}")

        return correlation_results

    def create_summary_report(self, samples, stats_data, correlation_results):
        """Create a summary report of the analysis"""
        report_path = os.path.join(self.output_dir, 'analysis_report.txt')

        with open(report_path, 'w') as f:
            f.write("FEATURE VISUALIZATION AND DATA SANITY ANALYSIS\n")
            f.write("=" * 50 + "\n\n")
            f.write(
                f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Number of videos analyzed: {len(samples)}\n")
            f.write(
                f"Video IDs: {', '.join([s['video_id'] for s in samples])}\n\n")

            f.write("SAMPLE STATISTICS:\n")
            f.write("-" * 20 + "\n")
            for stats in stats_data:
                f.write(f"Video {stats['video_id']}:\n")
                f.write(f"  Sequence length: {stats['seq_len']}\n")
                f.write(f"  Positive ratio: {stats['positive_ratio']:.2%}\n")
                f.write(f"  Feature dimension: {stats['feature_dim']}\n")
                f.write(
                    f"  Average offsets: L={stats['avg_left_offset']:.2f}, R={stats['avg_right_offset']:.2f}\n\n")

            f.write("CORRELATION ANALYSIS:\n")
            f.write("-" * 20 + "\n")
            for corr in correlation_results:
                f.write(f"Video {corr['video_id']}:\n")
                f.write(f"  Max correlation: {corr['max_correlation']:.4f}\n")
                f.write(
                    f"  Mean correlation: {corr['mean_correlation']:.4f}\n\n")

            # Overall assessment
            avg_max_corr = np.mean([c['max_correlation']
                                   for c in correlation_results])
            avg_mean_corr = np.mean([c['mean_correlation']
                                    for c in correlation_results])

            f.write("OVERALL ASSESSMENT:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Average max correlation: {avg_max_corr:.4f}\n")
            f.write(f"Average mean correlation: {avg_mean_corr:.4f}\n\n")

            if avg_max_corr < 0.1:
                f.write(
                    "⚠️  WARNING: Very low feature-label correlation detected!\n")
                f.write(
                    "This suggests potential issues with data preprocessing or labeling.\n\n")
            elif avg_max_corr < 0.3:
                f.write("⚠️  CAUTION: Low feature-label correlation.\n")
                f.write(
                    "Consider investigating data quality or feature extraction.\n\n")
            else:
                f.write("✓ Feature-label correlation appears reasonable.\n\n")

            f.write("VISUALIZATION INTERPRETATION:\n")
            f.write("-" * 30 + "\n")
            f.write("In the 3D plots:\n")
            f.write("- X-axis represents time (timesteps)\n")
            f.write("- Y-Z axes represent PCA/UMAP projections of features\n")
            f.write(
                "- Red triangles (c_t=1) should form distinct traces from blue dots (c_t=0)\n\n")
            f.write("If data is healthy:\n")
            f.write(
                "- Positive points should cluster or form distinguishable patterns\n")
            f.write(
                "- Clear separation between positive and negative points in feature space\n\n")
            f.write("If data is corrupted:\n")
            f.write("- No visible distinction between positive and negative points\n")
            f.write("- Random distribution of positive points\n")

        print(f"\nSummary report saved to: {report_path}")

    def run_analysis(self):
        """Run the complete feature analysis"""
        print("Starting Feature Visualization Analysis...")

        # Load samples
        samples = self.load_dataset_samples()
        if not samples:
            print("No samples to analyze. Exiting.")
            return

        # Analyze statistics
        stats_data = self.analyze_sample_statistics(samples)

        # Analyze correlations
        correlation_results = self.analyze_feature_label_correlation(samples)

        # Create visualizations
        methods = ['pca', 'umap']
        for method in methods:
            try:
                self.plot_3d_feature_analysis(samples, method)
                self.plot_2d_projections(samples, method)
                print(f"✓ {method.upper()} visualizations completed")
            except Exception as e:
                print(
                    f"✗ Failed to create {method.upper()} visualizations: {e}")

        # Create summary report
        self.create_summary_report(samples, stats_data, correlation_results)

        print(f"\n🎉 Analysis complete! Results saved to: {self.output_dir}")
        print("\nNext steps:")
        print("1. Check the 3D plots for distinct patterns in positive points")
        print("2. Review correlation analysis in the report")
        print("3. If correlations are very low (<0.1), investigate data preprocessing")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize features for data sanity checking')
    parser.add_argument('--config', type=str, default='configs/Repurpose.yaml',
                        help='Path to config file')
    parser.add_argument('--video_ids', type=str, nargs='+',
                        default=["AxwsdxOqQaU", "sYBddyIEQjA", "q5ycx357EoM",
                                 "tVUDJs6_A-0", "ceolO7WB8iA"],
                        help='List of video IDs to analyze')

    args = parser.parse_args()

    # Create visualizer and run analysis
    visualizer = FeatureVisualizer(args.config, args.video_ids)
    visualizer.run_analysis()


if __name__ == "__main__":
    main()
