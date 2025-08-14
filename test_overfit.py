#!/usr/bin/env python3
"""Test if model can overfit on 4 samples."""

import torch
import torch.nn as nn
from compatible_dataset import create_sequence_dataloader
from train_repurpose import RepurposeModel
import matplotlib.pyplot as plt
import numpy as np


def test_overfit():
    # Create dataloader with same 4 samples
    dataloader = create_sequence_dataloader(
        feature_dirs={
            "audio": "audio_pann_features",
            "visual": "video_clip_features",
            "caption": "caption_features",
        },
        annotation_file="debug_4samples.json",
        batch_size=4,
        num_workers=0,
        shuffle=False,
        min_modalities=3,
    )

    # Get feature dimensions
    batch = next(iter(dataloader))
    dim_audio = batch["features"]["audio"].shape[-1]
    dim_visual = batch["features"]["visual"].shape[-1]
    dim_caption = batch["features"]["caption"].shape[-1]

    print(
        f"Feature dims: audio={dim_audio}, visual={dim_visual}, caption={dim_caption}"
    )

    # Create model with small dimensions for faster training
    model = RepurposeModel(
        dim_audio=dim_audio,
        dim_visual=dim_visual,
        dim_caption=dim_caption,
        d_model=64,  # Smaller for faster training
        n_head=4,
        n_layers=1,  # Fewer layers
        lr=1e-2,  # Higher learning rate for overfitting
        lambda1=0.1,
        lambda2=0.3,
        lambda3=0.1,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Manual training loop for better control
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    losses = []
    accuracies = []

    print("\nStarting overfitting test...")
    print("=" * 60)

    for epoch in range(100):
        epoch_losses = []
        epoch_accs = []

        for batch in dataloader:
            # Move to device
            audio = batch["features"]["audio"].to(device)
            visual = batch["features"]["visual"].to(device)
            caption = batch["features"]["caption"].to(device)
            labels = batch["labels"].to(device)
            seq_mask = batch["sequence_masks"].to(device)

            # Forward pass
            optimizer.zero_grad()
            logit_a, logit_v, logit_f = model(audio, visual, caption)

            # Apply mask and compute loss
            valid_positions = seq_mask.bool()
            logit_f_valid = logit_f[valid_positions]
            labels_valid = labels[valid_positions]

            loss = model.focal_loss(logit_f_valid, labels_valid)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            with torch.no_grad():
                pred_probs = torch.sigmoid(logit_f_valid)
                pred_binary = (pred_probs > 0.5).float()
                accuracy = (pred_binary == labels_valid).float().mean()

            epoch_losses.append(loss.item())
            epoch_accs.append(accuracy.item())

        avg_loss = np.mean(epoch_losses)
        avg_acc = np.mean(epoch_accs)
        losses.append(avg_loss)
        accuracies.append(avg_acc)

        if epoch % 10 == 0 or epoch == 99:
            print(f"Epoch {epoch:3d}: Loss = {avg_loss:.4f}, Accuracy = {avg_acc:.4f}")

    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(losses)
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True)

    ax2.plot(accuracies)
    ax2.set_title("Training Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig("overfit_test.png", dpi=150)
    print(f"\nSaved training curves to overfit_test.png")

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final evaluation on training data:")
    print("=" * 60)

    model.eval()
    with torch.no_grad():
        all_preds = []
        all_labels = []

        for i, batch in enumerate(dataloader):
            audio = batch["features"]["audio"].to(device)
            visual = batch["features"]["visual"].to(device)
            caption = batch["features"]["caption"].to(device)
            labels = batch["labels"].to(device)
            seq_mask = batch["sequence_masks"].to(device)

            _, _, logit_f = model(audio, visual, caption)

            # Process each sequence individually
            batch_size = logit_f.shape[0]
            for seq_idx in range(batch_size):
                valid_length = int(seq_mask[seq_idx].sum().item())

                pred_probs = (
                    torch.sigmoid(logit_f[seq_idx, :valid_length]).cpu().numpy()
                )
                labels_np = labels[seq_idx, :valid_length].cpu().numpy()

                all_preds.extend(pred_probs.tolist())
                all_labels.extend(labels_np.tolist())

                # Per-video accuracy
                pred_binary = (pred_probs > 0.5).astype(float)
                acc = (pred_binary == labels_np).mean()
                pos_ratio = labels_np.mean()
                pred_pos_ratio = pred_binary.mean()

                print(f"\nVideo {seq_idx} ({batch['video_ids'][seq_idx]}):")
                print(f"  Accuracy: {acc:.4f}")
                print(f"  True positive ratio: {pos_ratio:.4f}")
                print(f"  Predicted positive ratio: {pred_pos_ratio:.4f}")

    # Overall metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_pred_binary = (all_preds > 0.5).astype(float)

    overall_acc = (all_pred_binary == all_labels).mean()
    print(f"\nOverall accuracy: {overall_acc:.4f}")
    print(f"Overall true positive ratio: {all_labels.mean():.4f}")
    print(f"Overall predicted positive ratio: {all_pred_binary.mean():.4f}")

    # Check if model learned anything
    unique_preds = np.unique(all_pred_binary)
    if len(unique_preds) == 1:
        print(f"\nWARNING: Model predicts only {unique_preds[0]} for all frames!")
    else:
        print(f"\nModel predicts both 0 and 1 (good)")


if __name__ == "__main__":
    test_overfit()
