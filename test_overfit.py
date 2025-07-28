#!/usr/bin/env python3
"""
Simple test script to verify if the model can overfit on 4 samples.
This is a debugging tool to identify issues with the training pipeline.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import sys
import os

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.SimpleMCTransformer import SimpleMCTransformer
from dataset.RepurposeClip import RepurposeClip
from dataset.args import get_args_parser
import argparse


def test_overfit(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create a minimal dataset with just 4 samples
    print("\n=== Loading Dataset ===")
    train_dataset = RepurposeClip(
        args.meta_dir,
        args.data_dir, 
        split='train',
        max_snippet_length=args.max_snippet_length,
        window_size=args.window_size,
        sampling_rate=args.sampling_rate,
        max_windows_per_video=args.max_windows_per_video
    )
    
    # Use only first 4 samples
    train_dataset.data_list = train_dataset.data_list[:4]
    print(f"Using {len(train_dataset)} samples for overfitting test")
    
    # Create dataloader with batch size 4 (all samples in one batch)
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=False,  # No shuffle to see consistent results
        num_workers=0,  # Single thread for debugging
        pin_memory=False,
        collate_fn=train_dataset.collate_fn
    )
    
    # Initialize model
    print("\n=== Initializing Model ===")
    model = SimpleMCTransformer(
        vis_dim=args.vis_dim,
        aud_dim=args.aud_dim,
        text_dim=args.text_dim,
        d_model=args.d_model,
        self_num_layers=args.self_num_layers,
        text_num_layers=args.text_num_layers,
        cross_num_layers=args.cross_num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff
    ).to(device)
    
    # Simple optimizer with high learning rate for faster overfitting
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    
    print("\n=== Starting Overfitting Test ===")
    print("If the model can learn, loss should decrease significantly over iterations.\n")
    
    model.train()
    
    # Track losses
    iteration_losses = []
    
    # Train for many iterations on the same 4 samples
    for epoch in range(100):
        epoch_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = model(batch)
            masks, out_cls_logits, out_offsets, gt_cls_labels, gt_offsets, feats = outputs
            
            # Compute loss
            losses = model.losses(masks, out_cls_logits, out_offsets, 
                                gt_cls_labels, gt_offsets, feats)
            loss = losses['cls_loss']
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Check gradients
            if epoch == 0 and batch_idx == 0:
                print("=== Gradient Check ===")
                total_norm = 0
                for name, p in model.named_parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                print(f"Total gradient norm: {total_norm:.6f}")
                
                # Check a few specific gradients
                if model.cls_head.weight.grad is not None:
                    print(f"cls_head gradient norm: {model.cls_head.weight.grad.norm():.6f}")
            
            optimizer.step()
            
            epoch_loss += loss.item()
            iteration_losses.append(loss.item())
        
        # Print progress every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Loss = {epoch_loss:.6f}")
            
            # Check predictions
            with torch.no_grad():
                pred_probs = torch.sigmoid(out_cls_logits)
                print(f"  Prediction stats - min: {pred_probs.min():.4f}, "
                      f"max: {pred_probs.max():.4f}, mean: {pred_probs.mean():.4f}")
    
    print("\n=== Overfitting Test Results ===")
    print(f"Initial loss: {iteration_losses[0]:.6f}")
    print(f"Final loss: {iteration_losses[-1]:.6f}")
    print(f"Loss reduction: {(iteration_losses[0] - iteration_losses[-1]) / iteration_losses[0] * 100:.2f}%")
    
    if iteration_losses[-1] < iteration_losses[0] * 0.1:
        print("\n✅ SUCCESS: Model can overfit! Loss decreased by more than 90%.")
    elif iteration_losses[-1] < iteration_losses[0] * 0.5:
        print("\n⚠️  PARTIAL SUCCESS: Model shows some learning but not strong overfitting.")
    else:
        print("\n❌ FAILURE: Model cannot overfit. There's likely a bug in the training pipeline.")
    
    # Save model predictions for inspection
    print("\n=== Final Predictions on Training Data ===")
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            outputs = model(batch)
            masks, out_cls_logits, out_offsets, gt_cls_labels, gt_offsets, feats = outputs
            
            pred_probs = torch.sigmoid(out_cls_logits)
            
            print(f"\nBatch {batch_idx}:")
            print(f"Ground truth positive labels: {(gt_cls_labels > 0.5).sum().item()}")
            print(f"Predicted positive (>0.5): {(pred_probs > 0.5).sum().item()}")
            print(f"Max prediction: {pred_probs.max().item():.4f}")
            print(f"Min prediction: {pred_probs.min().item():.4f}")


def main():
    parser = argparse.ArgumentParser(
        'Test overfitting capability',
        parents=[get_args_parser()],
        add_help=False
    )
    args = parser.parse_args()
    
    # Override some args for testing
    args.batch_size = 4
    args.num_workers = 0
    
    test_overfit(args)


if __name__ == '__main__':
    main()