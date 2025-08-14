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
        self.encoder = EncoderLayer(d_model, num_heads=4, d_ff=512, dropout=0.1)

        # Simple head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, 1)
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

    total_norm = total_norm ** (1.0 / 2)
    print(f"Total gradient norm: {total_norm:.6f}")
    print(f"Params with gradients: {param_count}")
    print(f"Params with near-zero gradients: {zero_grad_params}")

    return total_norm


def debug_predictions(model, batch, device):
    """Debug what the model is predicting"""
    visual = batch["features"]["visual"].to(device)
    labels = batch["labels"].to(device)
    mask = batch["sequence_masks"].to(device)

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
        print(
            f"  Predictions > 0.5: {(valid_probs > 0.5).sum().item()}/{len(valid_probs)}"
        )
        print(
            f"  Ground truth positives: {valid_labels.sum().item()}/{len(valid_labels)}"
        )
        print(f"  Ground truth ratio: {valid_labels.mean():.4f}")


def evaluate_model(model, dataloader, device, max_batches=5):
    """Evaluate model on validation/test set"""
    model.eval()
    total_correct = 0
    total_samples = 0
    total_positives = 0
    predicted_positives = 0
    true_positives = 0
    total_loss = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break

            visual = batch["features"]["visual"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["sequence_masks"].to(device)

            logits = model(visual, mask=mask)

            # Loss
            loss = sigmoid_focal_loss(
                logits, labels, alpha=0.5, gamma=2.0, reduction="none"
            )
            loss = (loss * mask).sum() / mask.sum()
            total_loss += loss.item()

            # Metrics
            preds = (torch.sigmoid(logits) > 0.5).float()
            valid_positions = mask.bool()

            correct = ((preds == labels) * mask).sum().item()
            samples = mask.sum().item()
            positives = (labels * mask).sum().item()
            pred_pos = (preds * mask).sum().item()
            true_pos = ((preds == 1) * (labels == 1) * mask).sum().item()

            total_correct += correct
            total_samples += samples
            total_positives += positives
            predicted_positives += pred_pos
            true_positives += true_pos

    accuracy = total_correct / total_samples if total_samples > 0 else 0
    precision = true_positives / predicted_positives if predicted_positives > 0 else 0
    recall = true_positives / total_positives if total_positives > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )
    avg_loss = total_loss / min(len(dataloader), max_batches)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "loss": avg_loss,
    }


def train_minimal_debug(args):
    """Train minimal transformer to isolate issues"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create dataloaders
    feature_dirs = {
        "visual": args.visual_dir,
        "audio": args.audio_dir,  # Still need for dataset
        "caption": args.caption_dir,
    }

    train_loader = create_sequence_dataloader(
        feature_dirs=feature_dirs,
        annotation_file=args.train_json,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
    )

    val_loader = None
    if args.val_json:
        val_loader = create_sequence_dataloader(
            feature_dirs=feature_dirs,
            annotation_file=args.val_json,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
        )
        print(f"Created validation dataloader with {len(val_loader)} batches")

    # Create minimal model
    model = MinimalTransformer(visual_dim=512, d_model=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")

    # Training loop with detailed debugging
    max_train_batches = 20  # Process more batches per epoch

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            if batch_idx >= max_train_batches:
                break

            visual = batch["features"]["visual"].to(device)
            labels = batch["labels"].to(device)
            mask = batch["sequence_masks"].to(device)

            # Forward pass
            logits = model(visual, mask=mask)

            # Loss computation
            loss = sigmoid_focal_loss(
                logits, labels, alpha=0.5, gamma=2.0, reduction="none"
            )
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
        train_accuracy = total_correct / total_samples if total_samples > 0 else 0
        avg_train_loss = total_loss / min(len(train_loader), max_train_batches)

        print(f"\nEpoch {epoch+1}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Train Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.1f}%)")

        # Validation evaluation
        if val_loader:
            val_metrics = evaluate_model(model, val_loader, device, max_batches=10)
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(
                f"  Val Accuracy: {val_metrics['accuracy']:.4f} ({val_metrics['accuracy']*100:.1f}%)"
            )
            print(
                f"  Val Precision: {val_metrics['precision']:.4f} ({val_metrics['precision']*100:.1f}%)"
            )
            print(
                f"  Val Recall: {val_metrics['recall']:.4f} ({val_metrics['recall']*100:.1f}%)"
            )
            print(f"  Val F1: {val_metrics['f1']:.4f} ({val_metrics['f1']*100:.1f}%)")

            # Use validation accuracy for early stopping decisions
            eval_accuracy = val_metrics["accuracy"]
        else:
            eval_accuracy = train_accuracy

        # Early success check
        if eval_accuracy > 0.8:
            print(f"✅ Model learned successfully in {epoch+1} epochs!")
            break
        elif eval_accuracy < 0.55 and epoch >= 3:
            print(f"⚠️  Model not learning after {epoch+1} epochs")
            print("Debugging suggestions:")
            print("1. Check if attention masks are correct")
            print("2. Try simpler architecture")
            print("3. Check learning rate")
            print("4. Verify data loading")
            if val_loader:
                print("5. Check train/val split consistency")


def main():
    parser = argparse.ArgumentParser(description="Debug full model training issues")
    parser.add_argument("--train-json", required=True)
    parser.add_argument("--val-json", help="Validation JSON file")
    parser.add_argument(
        "--visual-dir", required=True, help="Visual features with hints"
    )
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--caption-dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)

    args = parser.parse_args()

    print("🔍 DEBUGGING FULL MODEL vs SIMPLE MLP")
    print("=" * 50)
    print("Testing if the issue is architectural complexity...")
    print()

    train_minimal_debug(args)


if __name__ == "__main__":
    main()
