#!/usr/bin/env python3
"""
Debug script to inspect regression loss calculation and data structures.
This will help identify if there are issues with offset data or loss computation.
"""

import torch
import torch.nn as nn
import numpy as np
from compatible_dataset import create_sequence_dataloader
from models.losses import ctr_diou_loss_1d
from train_repurpose import RepurposeModel
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt


def visualize_sample_offsets(labels, offsets, seq_mask, sample_idx=0):
    """Visualize ground truth offsets for a sample."""
    # Get data for one sample
    labels_sample = labels[sample_idx].cpu().numpy()
    offsets_sample = offsets[sample_idx].cpu().numpy()
    mask_sample = seq_mask[sample_idx].cpu().numpy()

    valid_length = int(mask_sample.sum())
    labels_valid = labels_sample[:valid_length]
    offsets_valid = offsets_sample[:valid_length]

    # Find positive positions
    positive_idx = labels_valid > 0.5

    print(f"\n=== Sample {sample_idx} Ground Truth Analysis ===")
    print(f"Valid length: {valid_length}")
    print(f"Positive positions: {np.sum(positive_idx)} / {valid_length}")

    if np.sum(positive_idx) > 0:
        pos_indices = np.where(positive_idx)[0]
        print(
            f"Positive indices: {pos_indices[:10]}..."
            if len(pos_indices) > 10
            else f"Positive indices: {pos_indices}"
        )

        # Show offset values at positive positions
        print("\nOffset values at first few positive positions:")
        for i, idx in enumerate(pos_indices[:5]):
            left_off = offsets_valid[idx, 0]
            right_off = offsets_valid[idx, 1]
            print(
                f"  Position {idx}: left={left_off:.2f}, right={right_off:.2f}, duration={left_off+right_off:.2f}"
            )

            # Check if offsets make sense
            segment_start = idx - left_off
            segment_end = idx + right_off
            print(f"    -> Segment: [{segment_start:.1f}, {segment_end:.1f}]")

        # Statistics
        positive_offsets = offsets_valid[positive_idx]
        print(f"\nOffset statistics at positive positions:")
        print(
            f"  Left offset:  min={positive_offsets[:, 0].min():.2f}, max={positive_offsets[:, 0].max():.2f}, mean={positive_offsets[:, 0].mean():.2f}"
        )
        print(
            f"  Right offset: min={positive_offsets[:, 1].min():.2f}, max={positive_offsets[:, 1].max():.2f}, mean={positive_offsets[:, 1].mean():.2f}"
        )
        print(
            f"  Duration:     min={(positive_offsets[:, 0] + positive_offsets[:, 1]).min():.2f}, max={(positive_offsets[:, 0] + positive_offsets[:, 1]).max():.2f}, mean={(positive_offsets[:, 0] + positive_offsets[:, 1]).mean():.2f}"
        )

        # Check for pattern
        if len(pos_indices) > 1:
            # Check if consecutive positives have consistent offsets
            consecutive_groups = []
            current_group = [pos_indices[0]]

            for i in range(1, len(pos_indices)):
                if pos_indices[i] == pos_indices[i - 1] + 1:
                    current_group.append(pos_indices[i])
                else:
                    if len(current_group) > 1:
                        consecutive_groups.append(current_group)
                    current_group = [pos_indices[i]]
            if len(current_group) > 1:
                consecutive_groups.append(current_group)

            if consecutive_groups:
                print(
                    f"\nFound {len(consecutive_groups)} groups of consecutive positives:"
                )
                for group_idx, group in enumerate(
                    consecutive_groups[:3]
                ):  # Show first 3 groups
                    print(
                        f"  Group {group_idx}: positions {group[0]}-{group[-1]} (length {len(group)})"
                    )
                    group_offsets = offsets_valid[group]
                    # Check offset pattern
                    left_pattern = group_offsets[:, 0]
                    right_pattern = group_offsets[:, 1]
                    print(
                        f"    Left offsets:  {left_pattern[:5]}..."
                        if len(left_pattern) > 5
                        else f"    Left offsets:  {left_pattern}"
                    )
                    print(
                        f"    Right offsets: {right_pattern[:5]}..."
                        if len(right_pattern) > 5
                        else f"    Right offsets: {right_pattern}"
                    )

                    # Check if offsets follow expected pattern (increasing left, decreasing right)
                    if len(group) > 2:
                        left_increasing = all(
                            left_pattern[i] <= left_pattern[i + 1]
                            for i in range(len(left_pattern) - 1)
                        )
                        right_decreasing = all(
                            right_pattern[i] >= right_pattern[i + 1]
                            for i in range(len(right_pattern) - 1)
                        )
                        print(
                            f"    Pattern check: left_increasing={left_increasing}, right_decreasing={right_decreasing}"
                        )


def debug_regression_loss(model, batch, device):
    """Debug regression loss calculation step by step."""

    # Extract data
    audio = batch["features"]["audio"].to(device)
    visual = batch["features"]["visual"].to(device)
    caption = batch["features"]["caption"].to(device)
    labels = batch["labels"].to(device)
    offsets = batch["offsets"].to(device)  # Ground truth offsets
    seq_mask = batch["sequence_masks"].to(device)

    batch_size = labels.shape[0]
    seq_len = labels.shape[1]

    print("\n" + "=" * 80)
    print("DEBUGGING REGRESSION LOSS CALCULATION")
    print("=" * 80)

    print(f"\nInput shapes:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Offsets shape: {offsets.shape}")
    print(f"  Seq_mask shape: {seq_mask.shape}")

    # Visualize ground truth for first sample
    visualize_sample_offsets(labels, offsets, seq_mask, sample_idx=0)

    # Forward pass
    print("\n" + "-" * 40)
    print("FORWARD PASS")
    print("-" * 40)

    logit_a, logit_v, logit_f, offset_f = model(audio, visual, caption, mask=seq_mask)

    print(f"Model outputs:")
    print(f"  logit_f shape: {logit_f.shape}")
    print(f"  offset_f shape: {offset_f.shape}")

    # Check predicted offsets
    print(f"\nPredicted offset statistics (offset_f):")
    print(f"  Min: {offset_f.min().item():.4f}")
    print(f"  Max: {offset_f.max().item():.4f}")
    print(f"  Mean: {offset_f.mean().item():.4f}")
    print(f"  Std: {offset_f.std().item():.4f}")

    # Check if ReLU is working (all should be >= 0)
    negative_offsets = (offset_f < 0).sum().item()
    print(f"  Negative values: {negative_offsets} (should be 0 due to ReLU)")

    # Compute regression loss step by step
    print("\n" + "-" * 40)
    print("LOSS CALCULATION")
    print("-" * 40)

    # Step 1: Compute DIOU loss for all positions
    reg_loss_f_all = ctr_diou_loss_1d(offset_f, offsets, reduction="none")  # [B, T]

    print(f"\n1. Raw regression loss (reg_loss_f_all):")
    print(f"   Shape: {reg_loss_f_all.shape}")
    print(f"   Min: {reg_loss_f_all.min().item():.4f}")
    print(f"   Max: {reg_loss_f_all.max().item():.4f}")
    print(f"   Mean: {reg_loss_f_all.mean().item():.4f}")

    # Step 2: Create classification mask
    cls_mask = (labels > 0.5).float()

    print(f"\n2. Classification mask (cls_mask):")
    print(f"   Shape: {cls_mask.shape}")
    print(f"   Positive positions: {cls_mask.sum().item():.0f} / {cls_mask.numel()}")
    print(
        f"   Percentage positive: {100 * cls_mask.sum().item() / cls_mask.numel():.2f}%"
    )

    # Step 3: Combined mask
    combined_mask = seq_mask * cls_mask

    print(f"\n3. Combined mask (seq_mask * cls_mask):")
    print(f"   Shape: {combined_mask.shape}")
    print(f"   Active positions: {combined_mask.sum().item():.0f}")
    print(f"   seq_mask sum: {seq_mask.sum().item():.0f}")
    print(
        f"   Overlap: {combined_mask.sum().item():.0f} positions are both valid AND positive"
    )

    # Step 4: Apply mask and sum
    masked_loss = reg_loss_f_all * combined_mask
    reg_loss_f = masked_loss.sum()

    print(f"\n4. Masked and summed loss:")
    print(f"   Masked loss shape: {masked_loss.shape}")
    print(f"   Non-zero positions in masked loss: {(masked_loss > 0).sum().item()}")
    print(f"   Final loss value: {reg_loss_f.item():.4f}")

    # Detailed analysis for first sample
    print("\n" + "-" * 40)
    print("DETAILED ANALYSIS FOR SAMPLE 0")
    print("-" * 40)

    sample_idx = 0
    sample_labels = labels[sample_idx]
    sample_offsets_gt = offsets[sample_idx]
    sample_offsets_pred = offset_f[sample_idx]
    sample_mask = seq_mask[sample_idx]
    sample_cls_mask = cls_mask[sample_idx]
    sample_combined = combined_mask[sample_idx]
    sample_loss = reg_loss_f_all[sample_idx]

    valid_len = int(sample_mask.sum().item())
    positive_positions = (sample_cls_mask > 0).nonzero(as_tuple=True)[0]

    if len(positive_positions) > 0:
        print(f"\nPositive positions (first 10): {positive_positions[:10].tolist()}")

        print(f"\nComparison at first 5 positive positions:")
        for i, pos in enumerate(positive_positions[:5]):
            pos_idx = pos.item()
            print(f"\n  Position {pos_idx}:")
            print(
                f"    GT offsets:   left={sample_offsets_gt[pos_idx, 0]:.2f}, right={sample_offsets_gt[pos_idx, 1]:.2f}"
            )
            print(
                f"    Pred offsets: left={sample_offsets_pred[pos_idx, 0]:.2f}, right={sample_offsets_pred[pos_idx, 1]:.2f}"
            )
            print(f"    Loss at this position: {sample_loss[pos_idx]:.4f}")
            print(
                f"    Masks: seq={sample_mask[pos_idx]:.0f}, cls={sample_cls_mask[pos_idx]:.0f}, combined={sample_combined[pos_idx]:.0f}"
            )

            # Compute segment boundaries
            gt_left = pos_idx - sample_offsets_gt[pos_idx, 0].item()
            gt_right = pos_idx + sample_offsets_gt[pos_idx, 1].item()
            pred_left = pos_idx - sample_offsets_pred[pos_idx, 0].item()
            pred_right = pos_idx + sample_offsets_pred[pos_idx, 1].item()

            print(
                f"    GT segment:   [{gt_left:.1f}, {gt_right:.1f}] (duration: {gt_right - gt_left:.1f})"
            )
            print(
                f"    Pred segment: [{pred_left:.1f}, {pred_right:.1f}] (duration: {pred_right - pred_left:.1f})"
            )

    # Check for issues
    print("\n" + "-" * 40)
    print("POTENTIAL ISSUES CHECK")
    print("-" * 40)

    issues = []

    # Check 1: Are there any positive labels?
    if cls_mask.sum().item() == 0:
        issues.append("WARNING: No positive labels in batch!")

    # Check 2: Are predicted offsets all zero or constant?
    if offset_f.std().item() < 0.01:
        issues.append(
            f"WARNING: Predicted offsets have very low variance (std={offset_f.std().item():.4f})"
        )

    # Check 3: Are predicted offsets in reasonable range?
    if offset_f.max().item() > 100:
        issues.append(
            f"WARNING: Predicted offsets have very large values (max={offset_f.max().item():.2f})"
        )

    # Check 4: Is the loss reasonable?
    if reg_loss_f.item() == 0 and cls_mask.sum().item() > 0:
        issues.append("WARNING: Loss is exactly 0 despite having positive labels!")

    # Check 5: Gradient flow
    if offset_f.requires_grad:
        # Create a dummy backward to check gradients
        reg_loss_f.backward(retain_graph=True)
        if model.reg_head_f[0].weight.grad is not None:
            grad_norm = model.reg_head_f[0].weight.grad.norm().item()
            print(f"Gradient norm for first regression layer: {grad_norm:.6f}")
            if grad_norm < 1e-8:
                issues.append(f"WARNING: Very small gradients (norm={grad_norm:.8f})")
        else:
            issues.append("WARNING: No gradients computed for regression head!")

    if issues:
        print("\nDetected issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\nNo obvious issues detected.")

    return {
        "reg_loss": reg_loss_f.item(),
        "num_positives": cls_mask.sum().item(),
        "offset_pred_mean": offset_f.mean().item(),
        "offset_pred_std": offset_f.std().item(),
    }


def main():
    parser = argparse.ArgumentParser(description="Debug regression loss calculation")
    parser.add_argument("--train-json", required=True, help="Training JSON file")
    parser.add_argument("--visual-dir", required=True, help="Visual features directory")
    parser.add_argument("--audio-dir", required=True, help="Audio features directory")
    parser.add_argument(
        "--caption-dir", required=True, help="Caption features directory"
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument(
        "--num-batches", type=int, default=3, help="Number of batches to debug"
    )
    parser.add_argument("--checkpoint", help="Optional model checkpoint to load")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create dataloader
    feature_dirs = {
        "visual": args.visual_dir,
        "audio": args.audio_dir,
        "caption": args.caption_dir,
    }

    train_loader = create_sequence_dataloader(
        feature_dirs=feature_dirs,
        annotation_file=args.train_json,
        batch_size=args.batch_size,
        shuffle=False,  # Don't shuffle for consistent debugging
        num_workers=2,
    )

    # Get feature dimensions from first batch
    sample_batch = next(iter(train_loader))
    dim_audio = sample_batch["features"]["audio"].shape[-1]
    dim_visual = sample_batch["features"]["visual"].shape[-1]
    dim_caption = sample_batch["features"]["caption"].shape[-1]

    print(f"\nFeature dimensions:")
    print(f"  Audio: {dim_audio}")
    print(f"  Visual: {dim_visual}")
    print(f"  Caption: {dim_caption}")

    # Create model
    model = RepurposeModel(
        dim_audio=dim_audio,
        dim_visual=dim_visual,
        dim_caption=dim_caption,
        d_model=512,
        n_head=8,
        n_self_attn_layers=3,
        n_cross_attn_layers=3,
        n_fusion_layers=3,
    ).to(device)

    if args.checkpoint:
        print(f"\nLoading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])

    model.eval()

    # Debug multiple batches
    all_metrics = []

    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= args.num_batches:
            break

        print(f"\n{'='*80}")
        print(f"BATCH {batch_idx + 1} / {args.num_batches}")
        print(f"{'='*80}")

        with torch.no_grad():
            metrics = debug_regression_loss(model, batch, device)
            all_metrics.append(metrics)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY ACROSS ALL BATCHES")
    print("=" * 80)

    avg_loss = np.mean([m["reg_loss"] for m in all_metrics])
    avg_positives = np.mean([m["num_positives"] for m in all_metrics])
    avg_pred_mean = np.mean([m["offset_pred_mean"] for m in all_metrics])
    avg_pred_std = np.mean([m["offset_pred_std"] for m in all_metrics])

    print(f"\nAverage metrics:")
    print(f"  Regression loss: {avg_loss:.4f}")
    print(f"  Positive positions per batch: {avg_positives:.1f}")
    print(f"  Predicted offset mean: {avg_pred_mean:.4f}")
    print(f"  Predicted offset std: {avg_pred_std:.4f}")

    print("\n" + "=" * 80)
    print("DEBUG COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
