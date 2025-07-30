#!/usr/bin/env python3
"""Debug script to test data loading and label generation."""

import json
import numpy as np
import torch
from compatible_dataset import create_sequence_dataloader
import matplotlib.pyplot as plt

def main():
    # Create a small test annotation file with just 4 samples
    test_annotations = []
    
    # First, let's see what videos are available
    print("Checking available videos...")
    with open('data/train.json', 'r') as f:
        all_annotations = json.load(f)
    
    # Take first 4 videos
    test_annotations = all_annotations[:4]
    
    # Save to test file
    with open('debug_4samples.json', 'w') as f:
        json.dump(test_annotations, f, indent=2)
    
    print(f"\nCreated test annotation file with {len(test_annotations)} videos:")
    for ann in test_annotations:
        video_id = ann['youtube_id']
        time_range = ann.get('timeRangeOffset', [0, 0])
        segments = ann.get('segmentsOffset', [])
        coverage = ann.get('coverage', 0)
        print(f"  - {video_id}: range={time_range}, segments={len(segments)}, coverage={coverage:.2%}")
    
    # Create dataloader
    print("\nCreating dataloader...")
    dataloader = create_sequence_dataloader(
        feature_dirs={
            'audio': 'audio_pann_features',
            'visual': 'video_clip_features', 
            'caption': 'caption_features'
        },
        annotation_file='debug_4samples.json',
        batch_size=1,  # Process one at a time for clarity
        num_workers=0,
        shuffle=False,
        min_modalities=3
    )
    
    print(f"\nDataloader created with {len(dataloader)} batches")
    
    # Analyze each sample
    for idx, batch in enumerate(dataloader):
        print(f"\n{'='*60}")
        print(f"Sample {idx}: {batch['video_ids'][0]}")
        print(f"{'='*60}")
        
        # Get shapes
        audio_shape = batch['features']['audio'].shape
        visual_shape = batch['features']['visual'].shape
        caption_shape = batch['features']['caption'].shape
        labels_shape = batch['labels'].shape
        
        print(f"Feature shapes:")
        print(f"  Audio:   {audio_shape}")
        print(f"  Visual:  {visual_shape}")
        print(f"  Caption: {caption_shape}")
        print(f"  Labels:  {labels_shape}")
        
        # Analyze labels
        labels = batch['labels'][0].numpy()  # First sequence in batch
        seq_mask = batch['sequence_masks'][0].numpy()
        valid_length = int(seq_mask.sum())
        valid_labels = labels[:valid_length]
        
        num_positive = (valid_labels > 0.5).sum()
        num_negative = (valid_labels <= 0.5).sum()
        positive_ratio = num_positive / valid_length if valid_length > 0 else 0
        
        print(f"\nLabel statistics (valid length={valid_length}):")
        print(f"  Positive frames: {num_positive} ({positive_ratio:.2%})")
        print(f"  Negative frames: {num_negative} ({(1-positive_ratio):.2%})")
        
        # Find positive segments
        positive_segments = []
        in_segment = False
        start = 0
        
        for i in range(valid_length):
            if valid_labels[i] > 0.5 and not in_segment:
                in_segment = True
                start = i
            elif valid_labels[i] <= 0.5 and in_segment:
                in_segment = False
                positive_segments.append((start, i))
        
        if in_segment:
            positive_segments.append((start, valid_length))
        
        print(f"  Positive segments: {len(positive_segments)}")
        for seg_idx, (start, end) in enumerate(positive_segments):
            print(f"    Segment {seg_idx}: [{start}, {end}) = {end-start} frames")
        
        # Visualize labels
        plt.figure(figsize=(12, 3))
        plt.plot(valid_labels, 'b-', linewidth=0.5)
        plt.fill_between(range(valid_length), 0, valid_labels, alpha=0.3)
        plt.ylim(-0.1, 1.1)
        plt.xlabel('Frame Index')
        plt.ylabel('Label')
        plt.title(f'Labels for {batch["video_ids"][0]} (positive ratio: {positive_ratio:.2%})')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'debug_labels_sample_{idx}.png', dpi=150)
        plt.close()
        
        print(f"  Saved visualization to debug_labels_sample_{idx}.png")
    
    # Test with full batch
    print(f"\n{'='*60}")
    print("Testing with batch_size=4...")
    print(f"{'='*60}")
    
    batch_dataloader = create_sequence_dataloader(
        feature_dirs={
            'audio': 'audio_pann_features',
            'visual': 'video_clip_features',
            'caption': 'caption_features'
        },
        annotation_file='debug_4samples.json',
        batch_size=4,
        num_workers=0,
        shuffle=False,
        min_modalities=3
    )
    
    batch = next(iter(batch_dataloader))
    print(f"Batch shapes:")
    print(f"  Audio:   {batch['features']['audio'].shape}")
    print(f"  Visual:  {batch['features']['visual'].shape}")
    print(f"  Caption: {batch['features']['caption'].shape}")
    print(f"  Labels:  {batch['labels'].shape}")
    print(f"  Seq masks: {batch['sequence_masks'].shape}")
    
    # Check label diversity across batch
    all_labels = []
    for i in range(4):
        seq_mask = batch['sequence_masks'][i]
        valid_length = int(seq_mask.sum())
        labels = batch['labels'][i, :valid_length].numpy()
        all_labels.extend(labels.tolist())
    
    all_labels = np.array(all_labels)
    print(f"\nOverall label statistics across all 4 samples:")
    print(f"  Total frames: {len(all_labels)}")
    print(f"  Positive: {(all_labels > 0.5).sum()} ({(all_labels > 0.5).mean():.2%})")
    print(f"  Negative: {(all_labels <= 0.5).sum()} ({(all_labels <= 0.5).mean():.2%})")
    
    print("\nDebug complete!")
    
    # Additional masking tests
    print(f"\n{'='*60}")
    print("Testing masking behavior...")
    print(f"{'='*60}")
    
    # Test sequence masking
    for i in range(4):
        video_id = batch['video_ids'][i]
        seq_mask = batch['sequence_masks'][i]
        labels = batch['labels'][i]
        
        valid_length = int(seq_mask.sum())
        total_length = len(seq_mask)
        
        print(f"\nVideo {i} ({video_id}):")
        print(f"  Total length: {total_length}")
        print(f"  Valid length: {valid_length}")
        print(f"  Padded length: {total_length - valid_length}")
        
        # Check if padded positions have zero labels
        if total_length > valid_length:
            padded_labels = labels[valid_length:].numpy()
            print(f"  Padded label values: min={padded_labels.min():.4f}, max={padded_labels.max():.4f}")
            if not np.allclose(padded_labels, 0):
                print("  WARNING: Padded positions have non-zero labels!")
        
        # Check mask consistency
        expected_mask = torch.cat([torch.ones(valid_length), torch.zeros(total_length - valid_length)])
        mask_matches = torch.allclose(seq_mask, expected_mask)
        print(f"  Mask consistency: {'OK' if mask_matches else 'MISMATCH'}")
    
    # Test what happens when we apply masks (simulate training)
    print(f"\n{'='*60}")
    print("Simulating masked loss computation...")
    print(f"{'='*60}")
    
    from train_repurpose import FocalLoss
    
    focal_loss = FocalLoss()
    
    # Create fake predictions
    batch_size, seq_len = batch['labels'].shape
    fake_logits = torch.randn(batch_size, seq_len)
    
    # Method 1: Using boolean indexing (what train script does)
    print("\nMethod 1: Boolean indexing (current implementation)")
    total_valid = 0
    total_frames = 0
    for i in range(batch_size):
        valid_positions = batch['sequence_masks'][i].bool()
        valid_logits = fake_logits[i][valid_positions]
        valid_labels = batch['labels'][i][valid_positions]
        
        loss = focal_loss(valid_logits, valid_labels)
        
        num_valid = valid_positions.sum().item()
        total_valid += num_valid
        total_frames += seq_len
        
        print(f"  Video {i}: {num_valid} valid frames, loss shape: {loss.shape}")
    
    print(f"  Total: {total_valid}/{total_frames} valid frames ({total_valid/total_frames:.1%})")
    
    # Method 2: Flattening all valid positions (what training script does)
    print("\nMethod 2: Flattening all sequences (current training implementation)")
    all_valid_positions = batch['sequence_masks'].bool()
    all_valid_logits = fake_logits[all_valid_positions]
    all_valid_labels = batch['labels'][all_valid_positions]
    
    print(f"  Flattened shape: {all_valid_logits.shape}")
    print(f"  This concatenates all sequences: {all_valid_logits.shape[0]} total positions")
    
    # Compute loss on flattened data
    flattened_loss = focal_loss(all_valid_logits, all_valid_labels)
    print(f"  Loss computed on flattened data: {flattened_loss.item():.4f}")
    
    # Check if sequences are mixed
    print("\n  Checking sequence mixing in flattened approach:")
    start_idx = 0
    for i in range(batch_size):
        num_valid = batch['sequence_masks'][i].sum().item()
        end_idx = start_idx + int(num_valid)
        
        # Extract this sequence's portion from flattened tensor
        seq_logits = all_valid_logits[start_idx:end_idx]
        seq_labels = all_valid_labels[start_idx:end_idx]
        
        # Compare with individual sequence
        individual_valid = batch['sequence_masks'][i].bool()
        individual_logits = fake_logits[i][individual_valid]
        individual_labels = batch['labels'][i][individual_valid]
        
        matches = torch.allclose(seq_logits, individual_logits) and torch.allclose(seq_labels, individual_labels)
        print(f"    Sequence {i}: {'matches' if matches else 'MISMATCH'} (frames {start_idx}-{end_idx})")
        
        start_idx = end_idx
    
    # Check feature masks
    print(f"\n{'='*60}")
    print("Checking feature masks...")
    print(f"{'='*60}")
    
    for i in range(min(4, len(batch['video_ids']))):
        video_id = batch['video_ids'][i]
        print(f"\nVideo {i} ({video_id}):")
        for modality in ['audio', 'visual', 'caption']:
            mask = batch['feature_masks'][modality][i].item()
            print(f"  {modality}: {'available' if mask else 'MISSING'}")
            
            if not mask:
                # Check if features are zero
                features = batch['features'][modality][i]
                is_zero = torch.allclose(features, torch.zeros_like(features))
                print(f"    Features are {'zero (correct)' if is_zero else 'non-zero (ERROR)'}")

if __name__ == '__main__':
    main()