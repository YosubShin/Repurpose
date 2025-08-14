#!/usr/bin/env python3
"""Test stable training with different configurations."""

import torch
import torch.nn as nn
from compatible_dataset import create_sequence_dataloader
from train_repurpose import RepurposeModel, FocalLoss
import matplotlib.pyplot as plt
import numpy as np


def test_stable_training():
    # Create dataloader
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

    batch = next(iter(dataloader))

    # Test different configurations
    configs = [
        {
            "name": "BCE Loss + Low LR",
            "loss_fn": nn.BCEWithLogitsLoss(),
            "lr": 1e-4,
            "lambda1": 0.0,
            "lambda2": 1.0,
            "lambda3": 0.0,
        },
        {
            "name": "Focal Loss + Very Low LR",
            "loss_fn": FocalLoss(alpha=0.25, gamma=1.0),  # Reduced gamma
            "lr": 1e-5,
            "lambda1": 0.0,
            "lambda2": 1.0,
            "lambda3": 0.0,
        },
        {
            "name": "BCE Loss + Gradient Clipping",
            "loss_fn": nn.BCEWithLogitsLoss(),
            "lr": 1e-3,
            "lambda1": 0.0,
            "lambda2": 1.0,
            "lambda3": 0.0,
            "grad_clip": 0.1,
        },
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {}

    for config in configs:
        print(f"\n{'='*60}")
        print(f"Testing: {config['name']}")
        print(f"{'='*60}")

        # Create model
        model = RepurposeModel(
            dim_audio=2048,
            dim_visual=512,
            dim_caption=384,
            d_model=32,  # Even smaller
            n_head=2,
            n_layers=1,
            lr=config["lr"],
            lambda1=config["lambda1"],
            lambda2=config["lambda2"],
            lambda3=config["lambda3"],
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
        loss_fn = config["loss_fn"]

        losses = []
        accuracies = []
        pred_pos_ratios = []

        # Prepare data
        audio = batch["features"]["audio"].to(device)
        visual = batch["features"]["visual"].to(device)
        caption = batch["features"]["caption"].to(device)
        labels = batch["labels"].to(device)
        seq_mask = batch["sequence_masks"].to(device)

        valid_positions = seq_mask.bool()
        labels_valid = labels[valid_positions]

        # Training loop
        for epoch in range(50):
            optimizer.zero_grad()

            # Forward pass
            logit_a, logit_v, logit_f = model(audio, visual, caption)
            logit_f_valid = logit_f[valid_positions]

            # Compute loss
            loss = loss_fn(logit_f_valid, labels_valid)

            # Backward pass
            loss.backward()

            # Gradient clipping if specified
            if "grad_clip" in config:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])

            optimizer.step()

            # Metrics
            with torch.no_grad():
                pred_probs = torch.sigmoid(logit_f_valid)
                pred_binary = (pred_probs > 0.5).float()
                accuracy = (pred_binary == labels_valid).float().mean()
                pred_pos_ratio = pred_binary.mean()

                losses.append(loss.item())
                accuracies.append(accuracy.item())
                pred_pos_ratios.append(pred_pos_ratio.item())

            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch:2d}: Loss={loss:.4f}, Acc={accuracy:.4f}, PredPos={pred_pos_ratio:.4f}"
                )

        results[config["name"]] = {
            "losses": losses,
            "accuracies": accuracies,
            "pred_pos_ratios": pred_pos_ratios,
        }

        # Final evaluation
        print(
            f"Final: Loss={losses[-1]:.4f}, Acc={accuracies[-1]:.4f}, PredPos={pred_pos_ratios[-1]:.4f}"
        )

    # Plot results
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    for name, data in results.items():
        epochs = range(len(data["losses"]))

        axes[0].plot(epochs, data["losses"], label=name, marker="o", markersize=2)
        axes[1].plot(epochs, data["accuracies"], label=name, marker="o", markersize=2)
        axes[2].plot(
            epochs, data["pred_pos_ratios"], label=name, marker="o", markersize=2
        )

    # Ground truth positive ratio line
    true_pos_ratio = labels_valid.float().mean().item()
    axes[2].axhline(
        y=true_pos_ratio,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"True Pos Ratio ({true_pos_ratio:.3f})",
    )

    axes[0].set_title("Training Loss")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Training Accuracy")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)

    axes[2].set_title("Predicted Positive Ratio")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Positive Ratio")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig("stable_training_test.png", dpi=150)
    print(f"\nSaved comparison plot to stable_training_test.png")


if __name__ == "__main__":
    test_stable_training()
