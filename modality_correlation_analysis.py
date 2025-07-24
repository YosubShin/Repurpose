#!/usr/bin/env python3
"""
Modality Correlation Analysis Script

This script analyzes the correlation between different modalities (visual, audio, text)
at same vs different timesteps to verify feature extraction quality.

Expected behavior: Features from different modalities at the same timestep should
correlate more strongly than features from different timesteps.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from scipy.signal import correlate
from sklearn.linear_model import LogisticRegression
import torch
from dataset.RepurposeClip import RepurposeClipTest
import argparse
import yaml
from datetime import datetime
from collections import defaultdict
import json


class ModalityCorrelationAnalyzer:
    def __init__(self, config_path, target_video_ids=None):
        """
        Initialize the correlation analyzer
        
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
        self.output_dir = f"modality_correlation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
                    # Extract all modalities
                    target_samples.append({
                        'video_id': video_id,
                        'visual_features': sample['feats']['visual'],
                        'audio_features': sample['feats']['audio'],
                        'text_features': sample['feats']['text'],
                        'labels': sample['labels'],
                        'segments': sample['segments'],
                        'duration': sample['duration']
                    })
                    print(f"Loaded {video_id}: {sample['feats']['visual'].shape[0]} timesteps")
                except Exception as e:
                    print(f"Failed to load {video_id}: {e}")
        
        if not target_samples:
            print("No target samples found!")
            return []
        
        print(f"Successfully loaded {len(target_samples)} samples")
        return target_samples
    
    def compute_correlation_matrix(self, features1, features2, method='pearson'):
        """
        Compute correlation between two sets of features
        
        Args:
            features1: (T, D1) array of features from modality 1
            features2: (T, D2) array of features from modality 2
            method: 'pearson' or 'spearman'
        
        Returns:
            correlation value (averaged across dimensions)
        """
        if len(features1) != len(features2):
            raise ValueError("Feature sequences must have same length")
        
        correlations = []
        
        # Compute correlation for each pair of dimensions
        for i in range(min(features1.shape[1], 10)):  # Limit to first 10 dims for efficiency
            for j in range(min(features2.shape[1], 10)):
                if method == 'pearson':
                    corr, _ = pearsonr(features1[:, i], features2[:, j])
                else:
                    corr, _ = spearmanr(features1[:, i], features2[:, j])
                
                if not np.isnan(corr):
                    correlations.append(abs(corr))
        
        return np.mean(correlations) if correlations else 0.0
    
    def analyze_temporal_correlations(self, sample, max_offset=10):
        """
        Analyze correlations between modalities at different time offsets
        
        Args:
            sample: Dictionary containing features from all modalities
            max_offset: Maximum temporal offset to test
        
        Returns:
            Dictionary of correlation results
        """
        visual = sample['visual_features']
        audio = sample['audio_features']
        text = sample['text_features']
        
        results = {
            'visual_audio': [],
            'visual_text': [],
            'audio_text': [],
            'offsets': []
        }
        
        # Test different temporal offsets
        for offset in range(-max_offset, max_offset + 1):
            # Determine valid range for this offset
            if offset >= 0:
                start_idx = offset
                end_idx = min(len(visual), len(audio), len(text))
                slice1 = slice(start_idx, end_idx)
                slice2 = slice(0, end_idx - start_idx)
            else:
                start_idx = 0
                end_idx = min(len(visual), len(audio), len(text)) + offset
                slice1 = slice(0, end_idx)
                slice2 = slice(-offset, end_idx - offset)
            
            if end_idx - start_idx < 10:  # Need at least 10 timesteps
                continue
            
            # Compute correlations for this offset
            if offset >= 0:
                # Shift second modality forward
                corr_va = self.compute_correlation_matrix(visual[slice1], audio[slice2])
                corr_vt = self.compute_correlation_matrix(visual[slice1], text[slice2])
                corr_at = self.compute_correlation_matrix(audio[slice1], text[slice2])
            else:
                # Shift first modality forward
                corr_va = self.compute_correlation_matrix(visual[slice2], audio[slice1])
                corr_vt = self.compute_correlation_matrix(visual[slice2], text[slice1])
                corr_at = self.compute_correlation_matrix(audio[slice2], text[slice1])
            
            results['visual_audio'].append(corr_va)
            results['visual_text'].append(corr_vt)
            results['audio_text'].append(corr_at)
            results['offsets'].append(offset)
        
        return results
    
    def analyze_highlight_vs_background(self, sample):
        """
        Analyze correlation differences between highlight and background segments
        """
        labels = np.array(sample['labels'])
        visual = sample['visual_features']
        audio = sample['audio_features']
        text = sample['text_features']
        
        # Separate highlight and background indices
        highlight_mask = labels == 1
        background_mask = labels == 0
        
        results = {}
        
        # Analyze highlights
        if np.sum(highlight_mask) > 10:
            results['highlight'] = {
                'visual_audio': self.compute_correlation_matrix(
                    visual[highlight_mask], audio[highlight_mask]),
                'visual_text': self.compute_correlation_matrix(
                    visual[highlight_mask], text[highlight_mask]),
                'audio_text': self.compute_correlation_matrix(
                    audio[highlight_mask], text[highlight_mask]),
                'count': np.sum(highlight_mask)
            }
        
        # Analyze background
        if np.sum(background_mask) > 10:
            results['background'] = {
                'visual_audio': self.compute_correlation_matrix(
                    visual[background_mask], audio[background_mask]),
                'visual_text': self.compute_correlation_matrix(
                    visual[background_mask], text[background_mask]),
                'audio_text': self.compute_correlation_matrix(
                    audio[background_mask], text[background_mask]),
                'count': np.sum(background_mask)
            }
        
        return results
    
    def analyze_feature_label_regression(self, sample, max_lag=50):
        """
        Analyze correlation between logistic regression scores and labels
        
        Args:
            sample: Dictionary containing features and labels
            max_lag: Maximum lag for cross-correlation
        
        Returns:
            Dictionary of cross-correlation results
        """
        labels = np.array(sample['labels'])
        visual = sample['visual_features']
        audio = sample['audio_features'] 
        text = sample['text_features']
        
        # Skip if too few positive samples
        if np.sum(labels) < 10:
            return None
            
        results = {}
        
        # Test each modality separately and combined
        modalities = {
            'visual': visual,
            'audio': audio,
            'text': text,
            'combined': np.hstack([visual, audio, text])
        }
        
        for modality_name, features in modalities.items():
            try:
                # Train logistic regression
                lr = LogisticRegression(max_iter=1000, random_state=42)
                lr.fit(features, labels)
                
                # Compute scores
                w = lr.coef_.ravel()
                scores = features @ w
                
                # Convert labels to zero-mean format
                zero_mean_labels = (2 * labels - 1)
                zero_mean_labels = zero_mean_labels - zero_mean_labels.mean()
                
                # Compute cross-correlation
                # Using 'full' mode to get all lags from -N+1 to N-1
                cross_corr = correlate(scores, zero_mean_labels, mode='full', method='auto')
                
                # Normalize by the number of overlapping points
                norm = np.sqrt(np.sum(scores**2) * np.sum(zero_mean_labels**2))
                if norm > 0:
                    cross_corr = cross_corr / norm
                
                # Get lags
                lags = np.arange(-len(labels) + 1, len(labels))
                
                # Restrict to max_lag
                center = len(labels) - 1
                lag_mask = np.abs(lags) <= max_lag
                
                results[modality_name] = {
                    'cross_correlation': cross_corr[lag_mask],
                    'lags': lags[lag_mask],
                    'lr_score': lr.score(features, labels),
                    'weights_norm': np.linalg.norm(w),
                    'peak_lag': lags[lag_mask][np.argmax(np.abs(cross_corr[lag_mask]))],
                    'peak_correlation': np.max(np.abs(cross_corr[lag_mask]))
                }
                
            except Exception as e:
                print(f"Failed to analyze {modality_name}: {e}")
                results[modality_name] = None
        
        return results
    
    def plot_temporal_correlations(self, all_results):
        """Plot temporal correlation analysis results"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        modality_pairs = [
            ('visual_audio', 'Visual-Audio'),
            ('visual_text', 'Visual-Text'),
            ('audio_text', 'Audio-Text')
        ]
        
        # Average across all videos
        avg_results = defaultdict(lambda: defaultdict(list))
        for video_id, results in all_results.items():
            temporal = results['temporal']
            for offset_idx, offset in enumerate(temporal['offsets']):
                for pair_key, _ in modality_pairs:
                    avg_results[pair_key][offset].append(temporal[pair_key][offset_idx])
        
        # Plot average correlations
        for idx, (pair_key, pair_name) in enumerate(modality_pairs):
            ax = axes[0, idx]
            
            offsets = sorted(avg_results[pair_key].keys())
            means = [np.mean(avg_results[pair_key][off]) for off in offsets]
            stds = [np.std(avg_results[pair_key][off]) for off in offsets]
            
            ax.plot(offsets, means, 'b-', linewidth=2)
            ax.fill_between(offsets, 
                           np.array(means) - np.array(stds),
                           np.array(means) + np.array(stds),
                           alpha=0.3)
            ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)
            ax.set_xlabel('Temporal Offset (timesteps)')
            ax.set_ylabel('Average Correlation')
            ax.set_title(f'{pair_name} Correlation vs Temporal Offset')
            ax.grid(True, alpha=0.3)
            
            # Highlight peak
            peak_idx = np.argmax(means)
            ax.plot(offsets[peak_idx], means[peak_idx], 'ro', markersize=10)
            ax.text(offsets[peak_idx], means[peak_idx] + 0.01, 
                   f'Peak: {means[peak_idx]:.3f}', ha='center')
        
        # Plot individual video results
        for idx, (pair_key, pair_name) in enumerate(modality_pairs):
            ax = axes[1, idx]
            
            for video_id, results in all_results.items():
                temporal = results['temporal']
                ax.plot(temporal['offsets'], temporal[pair_key], 
                       alpha=0.7, linewidth=1, label=video_id)
            
            ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)
            ax.set_xlabel('Temporal Offset (timesteps)')
            ax.set_ylabel('Correlation')
            ax.set_title(f'{pair_name} Correlation by Video')
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'temporal_correlations.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_highlight_analysis(self, all_results):
        """Plot highlight vs background correlation analysis"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Collect data
        highlight_data = defaultdict(list)
        background_data = defaultdict(list)
        
        for video_id, results in all_results.items():
            if 'highlight_background' not in results:
                continue
            
            hb = results['highlight_background']
            if 'highlight' in hb:
                for key in ['visual_audio', 'visual_text', 'audio_text']:
                    highlight_data[key].append(hb['highlight'][key])
            
            if 'background' in hb:
                for key in ['visual_audio', 'visual_text', 'audio_text']:
                    background_data[key].append(hb['background'][key])
        
        # Plot 1: Bar plot comparison
        pairs = ['visual_audio', 'visual_text', 'audio_text']
        pair_labels = ['Visual-Audio', 'Visual-Text', 'Audio-Text']
        
        x = np.arange(len(pairs))
        width = 0.35
        
        highlight_means = [np.mean(highlight_data[p]) if highlight_data[p] else 0 for p in pairs]
        highlight_stds = [np.std(highlight_data[p]) if highlight_data[p] else 0 for p in pairs]
        background_means = [np.mean(background_data[p]) if background_data[p] else 0 for p in pairs]
        background_stds = [np.std(background_data[p]) if background_data[p] else 0 for p in pairs]
        
        bars1 = ax1.bar(x - width/2, highlight_means, width, yerr=highlight_stds,
                        label='Highlights', color='red', alpha=0.7)
        bars2 = ax1.bar(x + width/2, background_means, width, yerr=background_stds,
                        label='Background', color='blue', alpha=0.7)
        
        ax1.set_xlabel('Modality Pairs')
        ax1.set_ylabel('Average Correlation')
        ax1.set_title('Correlation: Highlights vs Background')
        ax1.set_xticks(x)
        ax1.set_xticklabels(pair_labels)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 2: Scatter plot for each video
        colors = plt.cm.Set3(np.linspace(0, 1, len(all_results)))
        
        for idx, (video_id, results) in enumerate(all_results.items()):
            if 'highlight_background' not in results:
                continue
            
            hb = results['highlight_background']
            if 'highlight' in hb and 'background' in hb:
                # Average correlation across all pairs
                h_avg = np.mean([hb['highlight'][k] for k in ['visual_audio', 'visual_text', 'audio_text']])
                b_avg = np.mean([hb['background'][k] for k in ['visual_audio', 'visual_text', 'audio_text']])
                
                ax2.scatter(b_avg, h_avg, color=colors[idx], s=100, alpha=0.7, label=video_id)
        
        # Add diagonal line
        lims = [
            np.min([ax2.get_xlim(), ax2.get_ylim()]),
            np.max([ax2.get_xlim(), ax2.get_ylim()]),
        ]
        ax2.plot(lims, lims, 'k--', alpha=0.5, zorder=0)
        
        ax2.set_xlabel('Background Correlation')
        ax2.set_ylabel('Highlight Correlation')
        ax2.set_title('Highlight vs Background Correlation by Video')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'highlight_vs_background.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_regression_correlation(self, all_results):
        """Plot logistic regression cross-correlation analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        modality_names = ['visual', 'audio', 'text', 'combined']
        colors = ['blue', 'orange', 'green', 'red']
        
        # Collect all regression results
        valid_results = []
        for video_id, results in all_results.items():
            if 'regression_analysis' in results and results['regression_analysis'] is not None:
                valid_results.append((video_id, results['regression_analysis']))
        
        if not valid_results:
            print("No valid regression analysis results to plot")
            return
        
        # Plot cross-correlation for each modality
        for idx, (modality_name, color) in enumerate(zip(modality_names, colors)):
            ax = axes[idx]
            
            # Plot individual videos
            all_correlations = []
            all_lags = None
            
            for video_id, reg_results in valid_results:
                if modality_name in reg_results and reg_results[modality_name] is not None:
                    result = reg_results[modality_name]
                    lags = result['lags']
                    cross_corr = result['cross_correlation']
                    
                    ax.plot(lags, cross_corr, alpha=0.3, color=color, linewidth=1)
                    all_correlations.append(cross_corr)
                    if all_lags is None:
                        all_lags = lags
            
            # Plot average if we have data
            if all_correlations:
                avg_corr = np.mean(all_correlations, axis=0)
                std_corr = np.std(all_correlations, axis=0)
                
                ax.plot(all_lags, avg_corr, color=color, linewidth=2, 
                       label=f'{modality_name.capitalize()} (avg)')
                ax.fill_between(all_lags, avg_corr - std_corr, avg_corr + std_corr,
                               alpha=0.2, color=color)
                
                # Mark peak
                peak_idx = np.argmax(np.abs(avg_corr))
                ax.plot(all_lags[peak_idx], avg_corr[peak_idx], 'o', 
                       color=color, markersize=8)
                ax.text(all_lags[peak_idx], avg_corr[peak_idx] + 0.01,
                       f'Peak: {avg_corr[peak_idx]:.3f}\nLag: {all_lags[peak_idx]}',
                       ha='center', fontsize=9)
            
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
            ax.set_xlabel('Lag (timesteps)')
            ax.set_ylabel('Cross-correlation')
            ax.set_title(f'{modality_name.capitalize()} - Logistic Regression Score vs Labels')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.suptitle('Cross-correlation: Logistic Regression Scores vs Zero-mean Labels', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'regression_correlation.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create summary statistics plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Collect peak correlations and lags
        peak_corrs = {mod: [] for mod in modality_names}
        peak_lags = {mod: [] for mod in modality_names}
        lr_scores = {mod: [] for mod in modality_names}
        
        for video_id, reg_results in valid_results:
            for modality_name in modality_names:
                if modality_name in reg_results and reg_results[modality_name] is not None:
                    result = reg_results[modality_name]
                    peak_corrs[modality_name].append(result['peak_correlation'])
                    peak_lags[modality_name].append(result['peak_lag'])
                    lr_scores[modality_name].append(result['lr_score'])
        
        # Plot 1: Peak correlations by modality
        positions = np.arange(len(modality_names))
        for i, modality in enumerate(modality_names):
            if peak_corrs[modality]:
                ax1.bar(i, np.mean(peak_corrs[modality]), yerr=np.std(peak_corrs[modality]),
                       color=colors[i], alpha=0.7, label=modality.capitalize())
                ax1.text(i, np.mean(peak_corrs[modality]) + 0.01, 
                        f'{np.mean(peak_corrs[modality]):.3f}',
                        ha='center', fontsize=9)
        
        ax1.set_xticks(positions)
        ax1.set_xticklabels([m.capitalize() for m in modality_names])
        ax1.set_ylabel('Peak Cross-correlation')
        ax1.set_title('Peak Cross-correlation by Modality')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Logistic regression accuracy
        for i, modality in enumerate(modality_names):
            if lr_scores[modality]:
                ax2.bar(i, np.mean(lr_scores[modality]), yerr=np.std(lr_scores[modality]),
                       color=colors[i], alpha=0.7)
                ax2.text(i, np.mean(lr_scores[modality]) + 0.01,
                        f'{np.mean(lr_scores[modality]):.3f}',
                        ha='center', fontsize=9)
        
        ax2.set_xticks(positions)
        ax2.set_xticklabels([m.capitalize() for m in modality_names])
        ax2.set_ylabel('Logistic Regression Accuracy')
        ax2.set_title('Classification Accuracy by Modality')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'regression_summary.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_summary_report(self, all_results):
        """Create a comprehensive summary report"""
        report_path = os.path.join(self.output_dir, 'correlation_analysis_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("MODALITY CORRELATION ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Number of videos analyzed: {len(all_results)}\n")
            f.write(f"Video IDs: {', '.join(all_results.keys())}\n\n")
            
            # Temporal correlation summary
            f.write("TEMPORAL CORRELATION ANALYSIS:\n")
            f.write("-" * 30 + "\n")
            
            # Find average peak correlations at offset 0
            zero_offset_corrs = defaultdict(list)
            for video_id, results in all_results.items():
                temporal = results['temporal']
                zero_idx = temporal['offsets'].index(0) if 0 in temporal['offsets'] else None
                if zero_idx is not None:
                    for pair in ['visual_audio', 'visual_text', 'audio_text']:
                        zero_offset_corrs[pair].append(temporal[pair][zero_idx])
            
            f.write("Average correlations at zero offset (same timestep):\n")
            f.write(f"  Visual-Audio: {np.mean(zero_offset_corrs['visual_audio']):.4f} "
                   f"(±{np.std(zero_offset_corrs['visual_audio']):.4f})\n")
            f.write(f"  Visual-Text: {np.mean(zero_offset_corrs['visual_text']):.4f} "
                   f"(±{np.std(zero_offset_corrs['visual_text']):.4f})\n")
            f.write(f"  Audio-Text: {np.mean(zero_offset_corrs['audio_text']):.4f} "
                   f"(±{np.std(zero_offset_corrs['audio_text']):.4f})\n\n")
            
            # Highlight vs background summary
            f.write("HIGHLIGHT VS BACKGROUND ANALYSIS:\n")
            f.write("-" * 30 + "\n")
            
            highlight_corrs = defaultdict(list)
            background_corrs = defaultdict(list)
            
            for video_id, results in all_results.items():
                if 'highlight_background' in results:
                    hb = results['highlight_background']
                    if 'highlight' in hb:
                        for pair in ['visual_audio', 'visual_text', 'audio_text']:
                            highlight_corrs[pair].append(hb['highlight'][pair])
                    if 'background' in hb:
                        for pair in ['visual_audio', 'visual_text', 'audio_text']:
                            background_corrs[pair].append(hb['background'][pair])
            
            f.write("Average correlations in highlight segments:\n")
            for pair, label in [('visual_audio', 'Visual-Audio'), 
                               ('visual_text', 'Visual-Text'),
                               ('audio_text', 'Audio-Text')]:
                if highlight_corrs[pair]:
                    f.write(f"  {label}: {np.mean(highlight_corrs[pair]):.4f} "
                           f"(±{np.std(highlight_corrs[pair]):.4f})\n")
            
            f.write("\nAverage correlations in background segments:\n")
            for pair, label in [('visual_audio', 'Visual-Audio'), 
                               ('visual_text', 'Visual-Text'),
                               ('audio_text', 'Audio-Text')]:
                if background_corrs[pair]:
                    f.write(f"  {label}: {np.mean(background_corrs[pair]):.4f} "
                           f"(±{np.std(background_corrs[pair]):.4f})\n")
            
            # Assessment
            f.write("\nASSESSMENT:\n")
            f.write("-" * 20 + "\n")
            
            # Check if correlations peak at zero offset
            peak_at_zero = True
            for pair in ['visual_audio', 'visual_text', 'audio_text']:
                avg_corrs = []
                for video_id, results in all_results.items():
                    temporal = results['temporal']
                    avg_corrs.append(temporal[pair])
                avg_corrs = np.mean(avg_corrs, axis=0)
                if np.argmax(avg_corrs) != len(avg_corrs)//2:  # Should peak at center (offset 0)
                    peak_at_zero = False
                    break
            
            if peak_at_zero:
                f.write("✓ Temporal alignment looks correct - correlations peak at zero offset\n")
            else:
                f.write("⚠️ WARNING: Correlations do not peak at zero offset - possible alignment issues\n")
            
            # Check correlation magnitudes
            avg_zero_corr = np.mean([np.mean(zero_offset_corrs[p]) for p in zero_offset_corrs])
            if avg_zero_corr < 0.1:
                f.write("⚠️ WARNING: Very low inter-modality correlations detected\n")
                f.write("   This might indicate:\n")
                f.write("   - Feature extraction issues\n")
                f.write("   - Modality misalignment\n")
                f.write("   - Poor feature quality\n")
            elif avg_zero_corr < 0.3:
                f.write("⚠️ CAUTION: Low inter-modality correlations\n")
                f.write("   Consider checking feature extraction pipeline\n")
            else:
                f.write("✓ Inter-modality correlations appear reasonable\n")
            
            # Add regression analysis summary
            f.write("\nLOGISTIC REGRESSION ANALYSIS:\n")
            f.write("-" * 30 + "\n")
            
            # Collect regression results
            reg_peak_corrs = defaultdict(list)
            reg_peak_lags = defaultdict(list)
            reg_accuracies = defaultdict(list)
            
            for video_id, results in all_results.items():
                if 'regression_analysis' in results and results['regression_analysis'] is not None:
                    reg_results = results['regression_analysis']
                    for modality in ['visual', 'audio', 'text', 'combined']:
                        if modality in reg_results and reg_results[modality] is not None:
                            reg_peak_corrs[modality].append(reg_results[modality]['peak_correlation'])
                            reg_peak_lags[modality].append(reg_results[modality]['peak_lag'])
                            reg_accuracies[modality].append(reg_results[modality]['lr_score'])
            
            if reg_peak_corrs:
                f.write("Peak cross-correlations (LR scores vs labels):\n")
                for modality in ['visual', 'audio', 'text', 'combined']:
                    if reg_peak_corrs[modality]:
                        f.write(f"  {modality.capitalize()}: {np.mean(reg_peak_corrs[modality]):.4f} "
                               f"(±{np.std(reg_peak_corrs[modality]):.4f}) at lag {np.mean(reg_peak_lags[modality]):.1f}\n")
                
                f.write("\nLogistic Regression accuracies:\n")
                for modality in ['visual', 'audio', 'text', 'combined']:
                    if reg_accuracies[modality]:
                        f.write(f"  {modality.capitalize()}: {np.mean(reg_accuracies[modality]):.3f} "
                               f"(±{np.std(reg_accuracies[modality]):.3f})\n")
                
                # Assessment of regression results
                f.write("\nRegression Analysis Assessment:\n")
                avg_peak_corr = np.mean([np.mean(reg_peak_corrs[m]) for m in reg_peak_corrs if reg_peak_corrs[m]])
                if avg_peak_corr < 0.1:
                    f.write("⚠️ Very low cross-correlations between LR scores and labels\n")
                    f.write("   This is expected given the complex nature of highlight detection\n")
                    f.write("   The transformer model is needed to capture temporal dependencies\n")
                elif avg_peak_corr < 0.3:
                    f.write("✓ Modest cross-correlations detected\n")
                    f.write("   Features contain some linearly separable signal\n")
                else:
                    f.write("✓ Good cross-correlations detected\n")
                    f.write("   Features contain reasonably strong linear signal\n")
            else:
                f.write("No regression analysis results available\n")
            
            f.write("\nRECOMMENDATIONS:\n")
            f.write("-" * 20 + "\n")
            f.write("1. If correlations don't peak at zero offset, check temporal alignment\n")
            f.write("2. If correlations are very low, verify feature extraction\n")
            f.write("3. Compare highlight vs background correlations - highlights should show stronger correlations\n")
            f.write("4. Visual-Audio correlation should typically be strongest\n")
            f.write("5. Low LR cross-correlations are expected - transformers capture complex patterns\n")
        
        print(f"\nSummary report saved to: {report_path}")
    
    def run_analysis(self):
        """Run the complete correlation analysis"""
        print("Starting Modality Correlation Analysis...")
        
        # Load samples
        samples = self.load_dataset_samples()
        if not samples:
            print("No samples to analyze. Exiting.")
            return
        
        # Analyze each video
        all_results = {}
        
        for sample in samples:
            video_id = sample['video_id']
            print(f"\nAnalyzing {video_id}...")
            
            results = {
                'video_id': video_id,
                'temporal': self.analyze_temporal_correlations(sample),
                'highlight_background': self.analyze_highlight_vs_background(sample),
                'regression_analysis': self.analyze_feature_label_regression(sample)
            }
            
            all_results[video_id] = results
            
            # Print quick summary
            temporal = results['temporal']
            zero_idx = temporal['offsets'].index(0) if 0 in temporal['offsets'] else None
            if zero_idx is not None:
                print(f"  Zero-offset correlations:")
                print(f"    Visual-Audio: {temporal['visual_audio'][zero_idx]:.4f}")
                print(f"    Visual-Text: {temporal['visual_text'][zero_idx]:.4f}")
                print(f"    Audio-Text: {temporal['audio_text'][zero_idx]:.4f}")
            
            # Print regression summary if available
            if results['regression_analysis'] is not None:
                print(f"  Logistic regression peak correlations:")
                for modality in ['visual', 'audio', 'text', 'combined']:
                    if modality in results['regression_analysis'] and results['regression_analysis'][modality] is not None:
                        reg_result = results['regression_analysis'][modality]
                        print(f"    {modality.capitalize()}: {reg_result['peak_correlation']:.4f} at lag {reg_result['peak_lag']}")
        
        # Create visualizations
        print("\nCreating visualizations...")
        self.plot_temporal_correlations(all_results)
        self.plot_highlight_analysis(all_results)
        self.plot_regression_correlation(all_results)
        
        # Create summary report
        self.create_summary_report(all_results)
        
        # Save raw results
        results_path = os.path.join(self.output_dir, 'raw_results.json')
        with open(results_path, 'w') as f:
            # Convert numpy arrays and numpy types to native Python types for JSON serialization
            json_results = {}
            for video_id, results in all_results.items():
                json_results[video_id] = {
                    'temporal': {
                        k: (v.tolist() if isinstance(v, np.ndarray) else 
                            [float(x) if isinstance(x, np.number) else x for x in v] if isinstance(v, list) else v)
                        for k, v in results['temporal'].items()
                    },
                    'highlight_background': {},
                    'regression_analysis': {}
                }
                
                # Convert highlight_background data
                hb = results['highlight_background']
                for segment_type in ['highlight', 'background']:
                    if segment_type in hb:
                        json_results[video_id]['highlight_background'][segment_type] = {
                            k: float(v) if isinstance(v, np.number) else int(v) if isinstance(v, np.integer) else v
                            for k, v in hb[segment_type].items()
                        }
                
                # Convert regression analysis data
                if results['regression_analysis'] is not None:
                    for modality, reg_data in results['regression_analysis'].items():
                        if reg_data is not None:
                            json_results[video_id]['regression_analysis'][modality] = {
                                'cross_correlation': reg_data['cross_correlation'].tolist(),
                                'lags': reg_data['lags'].tolist(),
                                'lr_score': float(reg_data['lr_score']),
                                'weights_norm': float(reg_data['weights_norm']),
                                'peak_lag': int(reg_data['peak_lag']),
                                'peak_correlation': float(reg_data['peak_correlation'])
                            }
            
            json.dump(json_results, f, indent=2)
        
        print(f"\n🎉 Analysis complete! Results saved to: {self.output_dir}")
        print("\nKey findings to check:")
        print("1. Temporal correlation plots should peak at offset=0")
        print("2. Highlight segments should show stronger correlations than background")
        print("3. Visual-Audio correlation is typically strongest")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze correlations between different modalities')
    parser.add_argument('--config', type=str, default='configs/Repurpose.yaml',
                        help='Path to config file')
    parser.add_argument('--video_ids', type=str, nargs='+',
                        default=["AxwsdxOqQaU", "sYBddyIEQjA", "q5ycx357EoM",
                                 "tVUDJs6_A-0", "ceolO7WB8iA"],
                        help='List of video IDs to analyze')
    
    args = parser.parse_args()
    
    # Create analyzer and run analysis
    analyzer = ModalityCorrelationAnalyzer(args.config, args.video_ids)
    analyzer.run_analysis()


if __name__ == "__main__":
    main()