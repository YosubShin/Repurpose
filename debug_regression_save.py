#!/usr/bin/env python3
"""
Debug script to save regression loss calculation data to human-readable files.
"""

import torch
import torch.nn as nn
import numpy as np
from compatible_dataset import create_sequence_dataloader
from models.losses import ctr_diou_loss_1d
from train_repurpose import RepurposeModel
import argparse
import json
import os
from datetime import datetime


def tensor_to_list(tensor):
    """Convert tensor to list for JSON serialization."""
    if isinstance(tensor, torch.Tensor):
        return tensor.cpu().numpy().tolist()
    elif isinstance(tensor, np.ndarray):
        return tensor.tolist()
    return tensor


def save_batch_debug_info(batch, model, device, output_dir, batch_idx):
    """Save detailed batch information to files."""

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Extract data
    audio = batch["features"]["audio"].to(device)
    visual = batch["features"]["visual"].to(device)
    caption = batch["features"]["caption"].to(device)
    labels = batch["labels"].to(device)
    offsets = batch["offsets"].to(device)  # Ground truth offsets
    seq_mask = batch["sequence_masks"].to(device)

    batch_size = labels.shape[0]
    seq_len = labels.shape[1]

    # Forward pass
    with torch.no_grad():
        logit_a, logit_v, logit_f, offset_f = model(
            audio, visual, caption, mask=seq_mask
        )

    # Compute losses step by step
    reg_loss_f_all = ctr_diou_loss_1d(offset_f, offsets, reduction="none")  # [B, T]
    cls_mask = (labels > 0.5).float()
    combined_mask = seq_mask * cls_mask
    masked_loss = reg_loss_f_all * combined_mask
    reg_loss_f = masked_loss.sum()

    # Prepare data for saving
    debug_data = {
        "batch_idx": batch_idx,
        "timestamp": datetime.now().isoformat(),
        "batch_size": batch_size,
        "sequence_length": seq_len,
        "shapes": {
            "audio": list(audio.shape),
            "visual": list(visual.shape),
            "caption": list(caption.shape),
            "labels": list(labels.shape),
            "offsets": list(offsets.shape),
            "seq_mask": list(seq_mask.shape),
            "logit_f": list(logit_f.shape),
            "offset_f": list(offset_f.shape),
        },
        "statistics": {
            "labels": {
                "positive_count": int(cls_mask.sum().item()),
                "total_count": int(cls_mask.numel()),
                "positive_ratio": float(cls_mask.sum().item() / cls_mask.numel()),
            },
            "gt_offsets": {
                "min": float(offsets.min().item()),
                "max": float(offsets.max().item()),
                "mean": float(offsets.mean().item()),
                "std": float(offsets.std().item()),
            },
            "pred_offsets": {
                "min": float(offset_f.min().item()),
                "max": float(offset_f.max().item()),
                "mean": float(offset_f.mean().item()),
                "std": float(offset_f.std().item()),
                "negative_count": int((offset_f < 0).sum().item()),
            },
            "loss": {
                "raw_loss_min": float(reg_loss_f_all.min().item()),
                "raw_loss_max": float(reg_loss_f_all.max().item()),
                "raw_loss_mean": float(reg_loss_f_all.mean().item()),
                "final_loss": float(reg_loss_f.item()),
                "active_positions": int(combined_mask.sum().item()),
            },
        },
        "samples": [],
    }

    # Add detailed information for each sample in the batch
    for sample_idx in range(batch_size):
        sample_labels = labels[sample_idx]
        sample_offsets_gt = offsets[sample_idx]
        sample_offsets_pred = offset_f[sample_idx]
        sample_mask = seq_mask[sample_idx]
        sample_cls_mask = cls_mask[sample_idx]
        sample_combined = combined_mask[sample_idx]
        sample_loss = reg_loss_f_all[sample_idx]

        valid_len = int(sample_mask.sum().item())
        positive_positions = (
            (sample_cls_mask > 0).nonzero(as_tuple=True)[0].cpu().numpy()
        )

        # Get first N positive positions for detailed analysis
        num_positions_to_save = min(20, len(positive_positions))

        position_details = []
        if len(positive_positions) > 0:
            for i in range(num_positions_to_save):
                pos_idx = int(positive_positions[i])
                position_details.append(
                    {
                        "position": pos_idx,
                        "gt_left_offset": float(sample_offsets_gt[pos_idx, 0].item()),
                        "gt_right_offset": float(sample_offsets_gt[pos_idx, 1].item()),
                        "pred_left_offset": float(
                            sample_offsets_pred[pos_idx, 0].item()
                        ),
                        "pred_right_offset": float(
                            sample_offsets_pred[pos_idx, 1].item()
                        ),
                        "loss": float(sample_loss[pos_idx].item()),
                        "gt_segment": [
                            float(pos_idx - sample_offsets_gt[pos_idx, 0].item()),
                            float(pos_idx + sample_offsets_gt[pos_idx, 1].item()),
                        ],
                        "pred_segment": [
                            float(pos_idx - sample_offsets_pred[pos_idx, 0].item()),
                            float(pos_idx + sample_offsets_pred[pos_idx, 1].item()),
                        ],
                    }
                )

        # Find consecutive positive groups
        consecutive_groups = []
        if len(positive_positions) > 0:
            current_group = [int(positive_positions[0])]
            for i in range(1, len(positive_positions)):
                if positive_positions[i] == positive_positions[i - 1] + 1:
                    current_group.append(int(positive_positions[i]))
                else:
                    if len(current_group) > 1:
                        consecutive_groups.append(current_group)
                    current_group = [int(positive_positions[i])]
            if len(current_group) > 1:
                consecutive_groups.append(current_group)

        sample_data = {
            "sample_idx": sample_idx,
            "valid_length": valid_len,
            "positive_count": len(positive_positions),
            "positive_positions": (
                positive_positions[:50].tolist() if len(positive_positions) > 0 else []
            ),
            "consecutive_groups": consecutive_groups[:5],  # Save first 5 groups
            "position_details": position_details,
            "video_id": (
                batch.get("video_ids", [f"video_{sample_idx}"])[sample_idx]
                if "video_ids" in batch
                else f"sample_{sample_idx}"
            ),
        }

        debug_data["samples"].append(sample_data)

    # Save as JSON
    json_path = os.path.join(output_dir, f"batch_{batch_idx:03d}_debug.json")
    with open(json_path, "w") as f:
        json.dump(debug_data, f, indent=2)

    # Also save a human-readable text summary
    txt_path = os.path.join(output_dir, f"batch_{batch_idx:03d}_summary.txt")
    with open(txt_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write(f"BATCH {batch_idx} DEBUG SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Timestamp: {debug_data['timestamp']}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Sequence Length: {seq_len}\n\n")

        f.write("STATISTICS:\n")
        f.write("-" * 40 + "\n")
        f.write(
            f"Positive Labels: {debug_data['statistics']['labels']['positive_count']} / {debug_data['statistics']['labels']['total_count']} "
        )
        f.write(
            f"({debug_data['statistics']['labels']['positive_ratio']*100:.2f}%)\n\n"
        )

        f.write("Ground Truth Offsets:\n")
        f.write(f"  Min: {debug_data['statistics']['gt_offsets']['min']:.4f}\n")
        f.write(f"  Max: {debug_data['statistics']['gt_offsets']['max']:.4f}\n")
        f.write(f"  Mean: {debug_data['statistics']['gt_offsets']['mean']:.4f}\n")
        f.write(f"  Std: {debug_data['statistics']['gt_offsets']['std']:.4f}\n\n")

        f.write("Predicted Offsets:\n")
        f.write(f"  Min: {debug_data['statistics']['pred_offsets']['min']:.4f}\n")
        f.write(f"  Max: {debug_data['statistics']['pred_offsets']['max']:.4f}\n")
        f.write(f"  Mean: {debug_data['statistics']['pred_offsets']['mean']:.4f}\n")
        f.write(f"  Std: {debug_data['statistics']['pred_offsets']['std']:.4f}\n\n")

        f.write("Loss:\n")
        f.write(
            f"  Raw Loss Range: [{debug_data['statistics']['loss']['raw_loss_min']:.4f}, {debug_data['statistics']['loss']['raw_loss_max']:.4f}]\n"
        )
        f.write(
            f"  Raw Loss Mean: {debug_data['statistics']['loss']['raw_loss_mean']:.4f}\n"
        )
        f.write(f"  Final Loss: {debug_data['statistics']['loss']['final_loss']:.4f}\n")
        f.write(
            f"  Active Positions: {debug_data['statistics']['loss']['active_positions']}\n\n"
        )

        f.write("SAMPLE DETAILS:\n")
        f.write("-" * 40 + "\n")

        for sample in debug_data["samples"][:2]:  # Write details for first 2 samples
            f.write(f"\nSample {sample['sample_idx']} ({sample['video_id']}):\n")
            f.write(f"  Valid Length: {sample['valid_length']}\n")
            f.write(f"  Positive Count: {sample['positive_count']}\n")

            if sample["consecutive_groups"]:
                f.write(
                    f"  Consecutive Groups: {len(sample['consecutive_groups'])} groups\n"
                )
                for i, group in enumerate(sample["consecutive_groups"][:3]):
                    f.write(
                        f"    Group {i}: positions {group[0]}-{group[-1]} (length {len(group)})\n"
                    )

            if sample["position_details"]:
                f.write(
                    f"\n  First {len(sample['position_details'])} positive positions:\n"
                )
                for detail in sample["position_details"][:5]:
                    f.write(f"    Position {detail['position']}:\n")
                    f.write(
                        f"      GT:   left={detail['gt_left_offset']:.2f}, right={detail['gt_right_offset']:.2f}\n"
                    )
                    f.write(
                        f"      Pred: left={detail['pred_left_offset']:.2f}, right={detail['pred_right_offset']:.2f}\n"
                    )
                    f.write(
                        f"      GT segment:   [{detail['gt_segment'][0]:.1f}, {detail['gt_segment'][1]:.1f}]\n"
                    )
                    f.write(
                        f"      Pred segment: [{detail['pred_segment'][0]:.1f}, {detail['pred_segment'][1]:.1f}]\n"
                    )
                    f.write(f"      Loss: {detail['loss']:.4f}\n")

    # Save raw tensors for first sample (for detailed analysis)
    if batch_idx == 0:
        sample_idx = 0
        valid_len = int(seq_mask[sample_idx].sum().item())

        # Save as numpy arrays
        np_data = {
            "labels": labels[sample_idx, :valid_len].cpu().numpy(),
            "gt_offsets": offsets[sample_idx, :valid_len].cpu().numpy(),
            "pred_offsets": offset_f[sample_idx, :valid_len].detach().cpu().numpy(),
            "loss_per_position": reg_loss_f_all[sample_idx, :valid_len].cpu().numpy(),
            "cls_mask": cls_mask[sample_idx, :valid_len].cpu().numpy(),
            "combined_mask": combined_mask[sample_idx, :valid_len].cpu().numpy(),
        }

        np_path = os.path.join(output_dir, f"batch_{batch_idx:03d}_sample_0_arrays.npz")
        np.savez(np_path, **np_data)

    return json_path, txt_path


def main():
    parser = argparse.ArgumentParser(
        description="Save regression loss debug data to files"
    )
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
    parser.add_argument(
        "--output-dir",
        type=str,
        default="regression_debug_output",
        help="Output directory for debug files",
    )
    parser.add_argument("--checkpoint", help="Optional model checkpoint to load")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving debug output to: {output_dir}")

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

    # Process batches
    saved_files = []

    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= args.num_batches:
            break

        print(f"\nProcessing batch {batch_idx + 1} / {args.num_batches}...")

        json_path, txt_path = save_batch_debug_info(
            batch, model, device, output_dir, batch_idx
        )

        saved_files.append((json_path, txt_path))
        print(f"  Saved: {json_path}")
        print(f"  Saved: {txt_path}")

    # Create index file
    index_path = os.path.join(output_dir, "index.txt")
    with open(index_path, "w") as f:
        f.write("Regression Debug Output Files\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Number of batches: {len(saved_files)}\n\n")

        f.write("Files:\n")
        for json_path, txt_path in saved_files:
            f.write(f"  - {os.path.basename(json_path)}\n")
            f.write(f"  - {os.path.basename(txt_path)}\n")

        f.write("\nUsage:\n")
        f.write("  - *_debug.json: Complete data in JSON format\n")
        f.write("  - *_summary.txt: Human-readable summary\n")
        f.write("  - *_arrays.npz: Raw numpy arrays for detailed analysis\n")

    print(f"\n{'='*50}")
    print(f"Debug output saved to: {output_dir}")
    print(f"Index file: {index_path}")
    print(f"Total files generated: {len(saved_files) * 2}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
