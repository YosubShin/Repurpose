#!/usr/bin/env python3
"""Diagnose why model won't learn."""

import torch
import torch.nn as nn
from compatible_dataset import create_sequence_dataloader
from train_repurpose import RepurposeModel, FocalLoss
import numpy as np

def diagnose():
    # Create dataloader
    dataloader = create_sequence_dataloader(
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
    
    batch = next(iter(dataloader))
    
    # 1. Check label distribution
    print("1. Label Distribution Check")
    print("="*60)
    labels = batch['labels']
    seq_mask = batch['sequence_masks']
    valid_labels = labels[seq_mask.bool()]
    
    print(f"Total valid labels: {len(valid_labels)}")
    print(f"Positive labels: {(valid_labels > 0.5).sum().item()} ({(valid_labels > 0.5).float().mean():.2%})")
    print(f"Negative labels: {(valid_labels <= 0.5).sum().item()} ({(valid_labels <= 0.5).float().mean():.2%})")
    
    # 2. Test Focal Loss behavior
    print("\n2. Focal Loss Behavior")
    print("="*60)
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    
    # Test with balanced predictions
    test_logits = torch.randn(100)
    test_labels = torch.cat([torch.ones(50), torch.zeros(50)])
    
    loss_balanced = focal_loss(test_logits, test_labels)
    print(f"Loss with random logits: {loss_balanced:.4f}")
    
    # Test with all negative predictions
    all_neg_logits = torch.full((100,), -10.0)  # Strong negative predictions
    loss_all_neg = focal_loss(all_neg_logits, test_labels)
    print(f"Loss with all negative predictions: {loss_all_neg:.4f}")
    
    # Test with all positive predictions
    all_pos_logits = torch.full((100,), 10.0)  # Strong positive predictions
    loss_all_pos = focal_loss(all_pos_logits, test_labels)
    print(f"Loss with all positive predictions: {loss_all_pos:.4f}")
    
    # 3. Check gradients
    print("\n3. Gradient Check")
    print("="*60)
    
    # Create simple model
    model = RepurposeModel(
        dim_audio=2048,
        dim_visual=512,
        dim_caption=384,
        d_model=64,
        n_head=4,
        n_layers=1
    )
    
    # Forward pass
    audio = batch['features']['audio']
    visual = batch['features']['visual']
    caption = batch['features']['caption']
    
    logit_a, logit_v, logit_f = model(audio, visual, caption)
    
    # Check outputs
    print(f"Logit shapes: a={logit_a.shape}, v={logit_v.shape}, f={logit_f.shape}")
    print(f"Logit ranges:")
    print(f"  Audio: [{logit_a.min():.4f}, {logit_a.max():.4f}]")
    print(f"  Visual: [{logit_v.min():.4f}, {logit_v.max():.4f}]")
    print(f"  Fusion: [{logit_f.min():.4f}, {logit_f.max():.4f}]")
    
    # Compute loss
    valid_positions = seq_mask.bool()
    logit_f_valid = logit_f[valid_positions]
    labels_valid = labels[valid_positions]
    
    # Try different loss components
    print("\n4. Loss Components")
    print("="*60)
    
    # Focal loss only
    loss_focal = focal_loss(logit_f_valid, labels_valid)
    print(f"Focal loss only: {loss_focal:.4f}")
    
    # Try BCE for comparison
    bce_loss = nn.BCEWithLogitsLoss()
    loss_bce = bce_loss(logit_f_valid, labels_valid)
    print(f"BCE loss: {loss_bce:.4f}")
    
    # Check if model parameters get gradients
    loss_focal.backward()
    
    print("\n5. Gradient Flow Check")
    print("="*60)
    total_grad_norm = 0
    params_with_grad = 0
    params_without_grad = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            total_grad_norm += grad_norm
            params_with_grad += 1
            if grad_norm > 0.01:  # Only show significant gradients
                print(f"  {name}: grad_norm = {grad_norm:.6f}")
        else:
            params_without_grad += 1
    
    print(f"\nTotal gradient norm: {total_grad_norm:.6f}")
    print(f"Parameters with gradients: {params_with_grad}")
    print(f"Parameters without gradients: {params_without_grad}")
    
    # 6. Test with different loss weights
    print("\n6. Testing Different Configurations")
    print("="*60)
    
    # Test with only multi-modal loss (no KL, no uni-modal)
    model2 = RepurposeModel(
        dim_audio=2048,
        dim_visual=512,
        dim_caption=384,
        d_model=64,
        n_head=4,
        n_layers=1,
        lambda1=0.0,  # No uni-modal loss
        lambda2=1.0,  # Only multi-modal loss
        lambda3=0.0   # No KL loss
    )
    
    optimizer = torch.optim.Adam(model2.parameters(), lr=0.01)
    
    # Try a few training steps
    for i in range(5):
        optimizer.zero_grad()
        logit_a, logit_v, logit_f = model2(audio, visual, caption)
        logit_f_valid = logit_f[valid_positions]
        
        loss = focal_loss(logit_f_valid, labels_valid)
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            probs = torch.sigmoid(logit_f_valid)
            pred_pos_ratio = (probs > 0.5).float().mean()
            print(f"  Step {i}: loss={loss:.4f}, pred_positive_ratio={pred_pos_ratio:.4f}")
    
    # 7. Check feature statistics
    print("\n7. Feature Statistics")
    print("="*60)
    print(f"Audio features: mean={audio.mean():.4f}, std={audio.std():.4f}")
    print(f"Visual features: mean={visual.mean():.4f}, std={visual.std():.4f}")
    print(f"Caption features: mean={caption.mean():.4f}, std={caption.std():.4f}")

if __name__ == '__main__':
    diagnose()