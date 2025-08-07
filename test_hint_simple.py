#!/usr/bin/env python3
"""
Minimal test to verify hint injection with simplest possible model.
If this doesn't work, the problem is in data loading or hint injection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from compatible_dataset import create_sequence_dataloader
import argparse
from tqdm import tqdm
import json


class SimpleHintDetector(nn.Module):
    """Simplest possible model: just visual features -> classification"""
    def __init__(self, visual_dim=512, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, visual_features):
        # visual_features: [batch, seq_len, 512]
        return self.net(visual_features).squeeze(-1)  # [batch, seq_len]


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        # Get visual features only
        visual = batch['features']['visual'].to(device)
        labels = batch['labels'].to(device)
        mask = batch['sequence_masks'].to(device)
        
        # Forward pass
        logits = model(visual)
        
        # Binary cross entropy loss with masking
        loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        loss = (loss * mask).sum() / mask.sum()
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metrics
        with torch.no_grad():
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct = ((preds == labels) * mask).sum().item()
            total_correct += correct
            total_samples += mask.sum().item()
            total_loss += loss.item()
    
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, accuracy


def evaluate(model, dataloader, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    total_positives = 0
    predicted_positives = 0
    true_positives = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            visual = batch['features']['visual'].to(device)
            labels = batch['labels'].to(device)
            mask = batch['sequence_masks'].to(device)
            
            logits = model(visual)
            preds = (torch.sigmoid(logits) > 0.5).float()
            
            # Accuracy
            correct = ((preds == labels) * mask).sum().item()
            total_correct += correct
            total_samples += mask.sum().item()
            
            # Precision/Recall stats
            total_positives += (labels * mask).sum().item()
            predicted_positives += (preds * mask).sum().item()
            true_positives += ((preds == 1) * (labels == 1) * mask).sum().item()
    
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    precision = true_positives / predicted_positives if predicted_positives > 0 else 0
    recall = true_positives / total_positives if total_positives > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return accuracy, precision, recall, f1


def main():
    parser = argparse.ArgumentParser(description="Test hint injection with simplest model")
    parser.add_argument("--train-json", required=True, help="Training JSON file")
    parser.add_argument("--val-json", required=True, help="Validation JSON file")
    parser.add_argument("--visual-dir", required=True, help="Visual features directory (with hints)")
    parser.add_argument("--audio-dir", required=True, help="Audio features directory")
    parser.add_argument("--caption-dir", required=True, help="Caption features directory")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("Creating dataloaders...")
    feature_dirs = {
        'visual': args.visual_dir,
        'audio': args.audio_dir,
        'caption': args.caption_dir
    }
    
    train_loader = create_sequence_dataloader(
        feature_dirs=feature_dirs,
        annotation_file=args.train_json,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = create_sequence_dataloader(
        feature_dirs=feature_dirs,
        annotation_file=args.val_json,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create simple model
    model = SimpleHintDetector(visual_dim=512).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    print("\n" + "="*60)
    print("EXPECTED BEHAVIOR WITH RED DOTS:")
    print("- Should reach >90% accuracy in 1-2 epochs")
    print("- If not, hints are not properly injected or loaded")
    print("="*60 + "\n")
    
    # Training loop
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2%}")
        
        # Evaluate
        val_acc, precision, recall, f1 = evaluate(model, val_loader, device)
        print(f"Val Acc: {val_acc:.2%}, Precision: {precision:.2%}, Recall: {recall:.2%}, F1: {f1:.2%}")
        
        # Early stopping if perfect
        if val_acc > 0.95:
            print(f"\n✅ SUCCESS! Model learned to detect red dots in {epoch+1} epochs!")
            print("Hint injection is working correctly.")
            break
    else:
        if val_acc < 0.7:
            print("\n❌ FAILURE! Model couldn't learn to detect red dots.")
            print("Possible issues:")
            print("1. Red dots not properly injected during feature extraction")
            print("2. Wrong features directory (using non-hint features)")
            print("3. Data loading issue")
            print("\nDebug steps:")
            print("1. Run verify_hint_injection.py to check features")
            print("2. Verify you're using the correct features directory")
            print("3. Check that visual features have hints with:")
            print("   python verify_hint_injection.py --dataset", args.train_json,
                  "--features-dir", args.visual_dir, "--verbose --plot")


if __name__ == "__main__":
    main()