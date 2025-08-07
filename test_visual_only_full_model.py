#!/usr/bin/env python3
"""
Test if the full RepurposeModel works when using only visual features (like the minimal transformer).
This will help identify if the issue is multi-modal complexity or architectural depth.
"""

import sys
import os

# Add the parent directory to the path so we can import from train_repurpose
sys.path.insert(0, '/Users/yosub/co/Repurpose')

from train_repurpose import RepurposeModel
import torch
import torch.nn.functional as F
from compatible_dataset import create_sequence_dataloader
from models.losses import sigmoid_focal_loss, ctr_diou_loss_1d
import argparse
from tqdm import tqdm

class VisualOnlyRepurposeModel(RepurposeModel):
    """Modified RepurposeModel that only uses visual features"""
    
    def forward(self, audio: torch.Tensor, visual: torch.Tensor, caption: torch.Tensor, mask: torch.Tensor = None):
        """Forward pass using only visual features, ignoring audio and caption"""
        # Only process visual features - ignore audio and caption
        v = self.proj_v(visual)
        
        # Add positional encoding
        v = v.transpose(0, 1)
        v = self.pos_encoding(v)
        v = v.transpose(0, 1)
        
        # Self-attention encoding for visual only
        for layer in self.enc_v:
            v = layer(v, mask=mask)
        
        # Skip all cross-attention and fusion - just use visual
        # Use visual features as if they were fused
        f = v
        
        # Classification heads - only use visual and "fused" (which is just visual)
        logit_v = self.head_v(v).squeeze(-1)
        logit_f = self.head_f(f).squeeze(-1)
        
        # Regression head
        offset_f = self.reg_head_f(f)
        
        # Return dummy audio logits (won't be used in loss)
        logit_a = torch.zeros_like(logit_v)
        
        return logit_a, logit_v, logit_f, offset_f

def train_visual_only_full_model(args):
    """Train full model architecture but with only visual features"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataloader
    feature_dirs = {
        'visual': args.visual_dir,
        'audio': args.audio_dir,  # Still need for dataset compatibility
        'caption': args.caption_dir
    }
    
    train_loader = create_sequence_dataloader(
        feature_dirs=feature_dirs,
        annotation_file=args.train_json,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2
    )
    
    val_loader = None
    if args.val_json:
        val_loader = create_sequence_dataloader(
            feature_dirs=feature_dirs,
            annotation_file=args.val_json,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2
        )
        print(f"Created validation dataloader")
    
    # Get feature dimensions
    sample_batch = next(iter(train_loader))
    dim_audio = sample_batch['features']['audio'].shape[-1]
    dim_visual = sample_batch['features']['visual'].shape[-1]
    dim_caption = sample_batch['features']['caption'].shape[-1]
    
    # Create visual-only version of full model
    model = VisualOnlyRepurposeModel(
        dim_audio=dim_audio,
        dim_visual=dim_visual,
        dim_caption=dim_caption,
        d_model=args.d_model,
        n_head=args.n_head,
        n_self_attn_layers=args.n_self_attn_layers,
        n_cross_attn_layers=0,  # Skip cross-attention
        n_fusion_layers=0,  # Skip fusion
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    print(f"Visual-only full model has {sum(p.numel() for p in model.parameters())} parameters")
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            if batch_idx >= 20:  # Limit for debugging
                break
                
            # Extract features
            audio = batch['features']['audio'].to(device)
            visual = batch['features']['visual'].to(device)
            caption = batch['features']['caption'].to(device)
            labels = batch['labels'].to(device)
            offsets = batch['offsets'].to(device)  # Ground truth regression offsets
            seq_mask = batch['sequence_masks'].to(device)
            
            # Forward pass - model will ignore audio/caption
            logit_a, logit_v, logit_f, offset_f = model(audio, visual, caption, mask=seq_mask)
            
            # Classification loss
            cls_loss = sigmoid_focal_loss(logit_f, labels, alpha=0.5, gamma=2.0, reduction='none')
            cls_loss = (cls_loss * seq_mask).sum() / seq_mask.sum()
            
            # Regression loss - only on positive positions
            reg_loss_all = ctr_diou_loss_1d(offset_f, offsets, reduction='none')  # [B, T]
            cls_mask = (labels > 0.5).float()
            combined_mask = seq_mask * cls_mask
            
            # Debug: Check if we have any positive examples
            num_positives = combined_mask.sum().item()
            if num_positives > 0:
                reg_loss = (reg_loss_all * combined_mask).sum() / num_positives
            else:
                reg_loss = torch.tensor(0.0, device=device)
            
            # Debug logging for first batch  
            if batch_idx == 0 and epoch < 3:  # Only show first few epochs
                print(f"  Debug - Batch {batch_idx}: cls_loss={cls_loss:.4f}, reg_loss={reg_loss:.4f} (weighted: {args.regression_weight * reg_loss:.4f}), positives={num_positives}")
            
            # Combined loss
            loss = cls_loss + args.regression_weight * reg_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Metrics
            with torch.no_grad():
                preds = (torch.sigmoid(logit_f) > 0.5).float()
                correct = ((preds == labels) * seq_mask).sum().item()
                samples = seq_mask.sum().item()
                
                total_correct += correct
                total_samples += samples
                total_loss += loss.item()
        
        # Epoch summary
        train_accuracy = total_correct / total_samples if total_samples > 0 else 0
        avg_loss = total_loss / min(len(train_loader), 20)
        
        print(f"\nEpoch {epoch+1}:")
        loss_desc = f"(classification + {args.regression_weight}*regression)"
        print(f"  Train Loss: {avg_loss:.4f} {loss_desc}")
        print(f"  Train Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.1f}%)")
        
        # Validation
        if val_loader:
            model.eval()
            val_correct = 0
            val_samples = 0
            val_loss = 0
            val_positives = 0
            val_predicted_positives = 0
            val_true_positives = 0
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    if batch_idx >= 10:
                        break
                        
                    audio = batch['features']['audio'].to(device)
                    visual = batch['features']['visual'].to(device)
                    caption = batch['features']['caption'].to(device)
                    labels = batch['labels'].to(device)
                    offsets = batch['offsets'].to(device)
                    seq_mask = batch['sequence_masks'].to(device)
                    
                    logit_a, logit_v, logit_f, offset_f = model(audio, visual, caption, mask=seq_mask)
                    
                    # Classification loss for validation
                    cls_loss = sigmoid_focal_loss(logit_f, labels, alpha=0.5, gamma=2.0, reduction='none')
                    cls_loss = (cls_loss * seq_mask).sum() / seq_mask.sum()
                    
                    # Regression loss for validation
                    reg_loss_all = ctr_diou_loss_1d(offset_f, offsets, reduction='none')
                    cls_mask = (labels > 0.5).float()
                    combined_mask = seq_mask * cls_mask
                    
                    num_positives = combined_mask.sum().item()
                    if num_positives > 0:
                        reg_loss = (reg_loss_all * combined_mask).sum() / num_positives
                    else:
                        reg_loss = torch.tensor(0.0, device=device)
                    
                    loss = cls_loss + args.regression_weight * reg_loss
                    val_loss += loss.item()
                    
                    preds = (torch.sigmoid(logit_f) > 0.5).float()
                    correct = ((preds == labels) * seq_mask).sum().item()
                    samples = seq_mask.sum().item()
                    positives = (labels * seq_mask).sum().item()
                    pred_pos = (preds * seq_mask).sum().item()
                    true_pos = ((preds == 1) * (labels == 1) * seq_mask).sum().item()
                    
                    val_correct += correct
                    val_samples += samples
                    val_positives += positives
                    val_predicted_positives += pred_pos
                    val_true_positives += true_pos
            
            val_accuracy = val_correct / val_samples if val_samples > 0 else 0
            val_precision = val_true_positives / val_predicted_positives if val_predicted_positives > 0 else 0
            val_recall = val_true_positives / val_positives if val_positives > 0 else 0
            val_f1 = 2 * val_precision * val_recall / (val_precision + val_recall) if (val_precision + val_recall) > 0 else 0
            avg_val_loss = val_loss / min(len(val_loader), 10)
            
            val_loss_desc = f"(classification + {args.regression_weight}*regression)"
            print(f"  Val Loss: {avg_val_loss:.4f} {val_loss_desc}")
            print(f"  Val Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.1f}%)")
            print(f"  Val Precision: {val_precision:.4f} ({val_precision*100:.1f}%)")
            print(f"  Val Recall: {val_recall:.4f} ({val_recall*100:.1f}%)")
            print(f"  Val F1: {val_f1:.4f} ({val_f1*100:.1f}%)")
            
            if val_accuracy > 0.8:
                print(f"✅ Visual-only full model learned successfully in {epoch+1} epochs!")
                break
        
        if train_accuracy > 0.8 and not val_loader:
            print(f"✅ Visual-only full model learned successfully in {epoch+1} epochs!")
            break

def main():
    parser = argparse.ArgumentParser(description="Test visual-only full model")
    parser.add_argument("--train-json", required=True)
    parser.add_argument("--val-json", help="Validation JSON file")
    parser.add_argument("--visual-dir", required=True, help="Visual features with hints")
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--caption-dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--n-self-attn-layers", type=int, default=3)
    parser.add_argument("--regression-weight", type=float, default=0.01,
                      help="Weight for regression loss (default: 0.01)")
    
    args = parser.parse_args()
    
    print("🔍 TESTING VISUAL-ONLY FULL MODEL")
    print("=" * 50)
    print(f"Mode: Classification + Regression (weight={args.regression_weight})")
    print(f"Configuration: {args.n_self_attn_layers} layers, d_model={args.d_model}, {args.n_head} heads")
    print("Strategy: Small regression weight to avoid interference with classification learning")
    print()
    
    train_visual_only_full_model(args)
    
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print("RESULTS COMPARISON:")
    print("• Minimal Transformer (1 layer, d_model=128): 86.5% val accuracy, 65.4% recall ✅")
    print("• Visual-only Full Model (3 layers, NO hints): 65.8% accuracy, 0.7% recall ❌") 
    print("• Visual-only Full Model (3 layers, WITH hints): 92.2% accuracy, 77.5% recall ✅")
    print("• Visual-only Full Model + Regression: [Your result above]")
    print()
    print("Expected results with small regression weight (0.01):")
    print("• Classification should still achieve ~90% accuracy")  
    print("• Regression loss contributes minimally (~0.01 * 0.47 = 0.005 vs 0.08 classification)")
    print("• Model learns both classification and boundary regression simultaneously")
    print()
    print("If accuracy is still low, try:")
    print("python test_visual_only_full_model.py --train-json ... --regression-weight 0.001  # Even smaller")

if __name__ == "__main__":
    main()