#!/usr/bin/env python3
"""
Debug script to identify why the full transformer model isn't learning hints
when the simple MLP model learns them perfectly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from compatible_dataset import create_sequence_dataloader
from models.transformer import PositionalEncoding, EncoderLayer
from models.losses import sigmoid_focal_loss
import argparse
from tqdm import tqdm

class MinimalTransformer(nn.Module):
    """Minimal transformer to test if complexity is the issue"""
    def __init__(self, visual_dim=512, d_model=128):
        super().__init__()
        # Just visual path for debugging
        self.proj_v = nn.Linear(visual_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=2000)
        
        # Single self-attention layer
        self.encoder = EncoderLayer(d_model, n_head=4, d_ff=512, dropout=0.1)
        
        # Simple head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )
    
    def forward(self, visual, mask=None):
        # Project visual features
        v = self.proj_v(visual)  # [B, T, d_model]
        
        # Add positional encoding
        v = v.transpose(0, 1)  # [T, B, d_model]
        v = self.pos_encoding(v)
        v = v.transpose(0, 1)  # [B, T, d_model]
        
        # Self-attention
        v = self.encoder(v, mask=mask)
        
        # Classification
        logits = self.head(v).squeeze(-1)  # [B, T]
        
        return logits

def check_gradients(model, loss):
    """Check if gradients are flowing properly"""
    total_norm = 0
    param_count = 0
    zero_grad_params = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
            if param_norm.item() < 1e-8:
                zero_grad_params += 1
        else:
            print(f"No gradient for {name}")
    
    total_norm = total_norm ** (1. / 2)
    print(f"Total gradient norm: {total_norm:.6f}")
    print(f"Params with gradients: {param_count}")
    print(f"Params with near-zero gradients: {zero_grad_params}")
    
    return total_norm

def debug_predictions(model, batch, device):
    """Debug what the model is predicting"""
    visual = batch['features']['visual'].to(device)
    labels = batch['labels'].to(device)
    mask = batch['sequence_masks'].to(device)
    
    # Get predictions
    with torch.no_grad():
        logits = model(visual, mask=mask)
        probs = torch.sigmoid(logits)
        
        # Statistics
        valid_positions = mask.bool()
        valid_probs = probs[valid_positions]
        valid_labels = labels[valid_positions]
        
        print(f"Prediction stats:")
        print(f"  Mean probability: {valid_probs.mean():.4f}")
        print(f"  Min probability: {valid_probs.min():.4f}")
        print(f"  Max probability: {valid_probs.max():.4f}")
        print(f"  Predictions > 0.5: {(valid_probs > 0.5).sum().item()}/{len(valid_probs)}")
        print(f"  Ground truth positives: {valid_labels.sum().item()}/{len(valid_labels)}")
        print(f"  Ground truth ratio: {valid_labels.mean():.4f}")

def train_minimal_debug(args):
    """Train minimal transformer to isolate issues"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataloader
    feature_dirs = {
        'visual': args.visual_dir,
        'audio': args.audio_dir,  # Still need for dataset
        'caption': args.caption_dir
    }
    
    train_loader = create_sequence_dataloader(
        feature_dirs=feature_dirs,
        annotation_file=args.train_json,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2
    )
    
    # Create minimal model
    model = MinimalTransformer(visual_dim=512, d_model=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    
    # Training loop with detailed debugging
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            if batch_idx >= 10:  # Only first 10 batches for debugging
                break
                
            visual = batch['features']['visual'].to(device)
            labels = batch['labels'].to(device)
            mask = batch['sequence_masks'].to(device)
            
            # Forward pass
            logits = model(visual, mask=mask)
            
            # Loss computation
            loss = sigmoid_focal_loss(logits, labels, alpha=0.5, gamma=2.0, reduction='none')
            loss = (loss * mask).sum() / mask.sum()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Check gradients
            if batch_idx == 0 and epoch == 0:
                print(f"\nGradient check for first batch:")
                grad_norm = check_gradients(model, loss)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            # Metrics
            with torch.no_grad():
                valid_positions = mask.bool()
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct = ((preds == labels) * mask).sum().item()
                samples = mask.sum().item()
                
                total_correct += correct
                total_samples += samples
                total_loss += loss.item()
                
                # Debug first batch of first epoch
                if batch_idx == 0 and epoch == 0:
                    print(f"\nFirst batch debugging:")
                    debug_predictions(model, batch, device)
        
        # Epoch summary
        accuracy = total_correct / total_samples if total_samples > 0 else 0
        avg_loss = total_loss / min(len(train_loader), 10)
        
        print(f"\nEpoch {epoch+1}:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
        
        # Early success check
        if accuracy > 0.8:
            print(f"✅ Model learned successfully in {epoch+1} epochs!")
            break
        elif accuracy < 0.55 and epoch >= 2:
            print(f"⚠️  Model not learning after {epoch+1} epochs")
            print("Debugging suggestions:")
            print("1. Check if attention masks are correct")
            print("2. Try simpler architecture")
            print("3. Check learning rate")
            print("4. Verify data loading")

def main():
    parser = argparse.ArgumentParser(description="Debug full model training issues")
    parser.add_argument("--train-json", required=True)
    parser.add_argument("--visual-dir", required=True, help="Visual features with hints")
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--caption-dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    
    args = parser.parse_args()
    
    print("🔍 DEBUGGING FULL MODEL vs SIMPLE MLP")
    print("=" * 50)
    print("Testing if the issue is architectural complexity...")
    print()
    
    train_minimal_debug(args)

if __name__ == "__main__":
    main()