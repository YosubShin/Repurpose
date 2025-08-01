#!/usr/bin/env python3
"""
Standalone training script for RepurposeModel with comprehensive logging and wandb integration.
Incorporates memory optimizations and visualization capabilities.
"""

import os
import gc
import sys
import time
import argparse
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
from pytorch_lightning.loggers import WandbLogger
import wandb
import matplotlib.pyplot as plt
import psutil

# For visualization
import matplotlib.patches as patches

# Import the sequence dataset
from compatible_dataset import create_sequence_dataloader

# Configure logging


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup comprehensive logging configuration."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )

    # Set specific logger levels
    logging.getLogger('pytorch_lightning').setLevel(logging.INFO)
    logging.getLogger('wandb').setLevel(logging.INFO)

    return logging.getLogger(__name__)


def log_memory_usage(logger, stage: str):
    """Log detailed CPU and GPU memory usage."""
    try:
        # CPU memory
        cpu_mem = psutil.virtual_memory()
        cpu_used_gb = cpu_mem.used / (1024**3)
        cpu_total_gb = cpu_mem.total / (1024**3)
        cpu_percent = cpu_mem.percent

        # GPU memory
        if torch.cuda.is_available():
            gpu_allocated = torch.cuda.memory_allocated() / (1024**3)
            gpu_reserved = torch.cuda.memory_reserved() / (1024**3)
            gpu_max_allocated = torch.cuda.max_memory_allocated() / (1024**3)
            logger.info(f"{stage} | CPU: {cpu_used_gb:.1f}/{cpu_total_gb:.1f}GB ({cpu_percent:.1f}%) | GPU: {gpu_allocated:.1f}GB alloc, {gpu_reserved:.1f}GB reserved, {gpu_max_allocated:.1f}GB max")
        else:
            logger.info(
                f"{stage} | CPU: {cpu_used_gb:.1f}/{cpu_total_gb:.1f}GB ({cpu_percent:.1f}%) | GPU: N/A")
    except Exception as e:
        logger.warning(f"Failed to log memory usage at {stage}: {e}")


# ==================== Loss Functions ====================
class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean', eps: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        probs = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_term = (1 - p_t) ** self.gamma
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * focal_term * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


@torch.no_grad()
def _kl_div_bernoulli(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-6):
    """Element-wise KL divergence for Bernoulli distributions."""
    p = p.clamp(eps, 1 - eps)
    q = q.clamp(eps, 1 - eps)
    return p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))


def kl_div_bernoulli(p: torch.Tensor, q: torch.Tensor):
    """KL divergence between Bernoulli distributions."""
    return _kl_div_bernoulli(p, q).mean()


# ==================== Model Definition ====================
class RepurposeModel(pl.LightningModule):
    """
    Repurpose model with multi-modal fusion and alignment losses.
    Includes memory optimizations and comprehensive logging.
    """

    def __init__(
        self,
        dim_audio: int,
        dim_visual: int,
        dim_caption: int,
        d_model: int = 128,
        n_head: int = 4,
        n_layers: int = 2,
        lr: float = 1e-3,
        lambda1: float = 0.1,
        lambda2: float = 0.3,
        lambda3: float = 0.1,
        log_interval: int = 10
    ):
        super().__init__()
        self.save_hyperparameters()

        # Logging
        self.logger_instance = logging.getLogger(self.__class__.__name__)
        self.log_interval = log_interval
        self.step_count = 0

        # Projections to shared dimension
        self.proj_a = nn.Linear(dim_audio, d_model)
        self.proj_v = nn.Linear(dim_visual, d_model)
        self.proj_c = nn.Linear(dim_caption, d_model)

        # Self-attention encoders (per modality)
        def _encoder():
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_head,
                dim_feedforward=d_model * 4,
                batch_first=True
            )
            return nn.TransformerEncoder(layer, num_layers=n_layers)

        self.enc_a = _encoder()
        self.enc_v = _encoder()
        self.enc_c = _encoder()

        # Fusion encoder
        layer_f = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_model * 4,
            batch_first=True
        )
        self.enc_fusion = nn.TransformerEncoder(layer_f, num_layers=1)

        # Classification heads
        def _head():
            return nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, 1)
            )

        self.head_a = _head()
        self.head_v = _head()
        self.head_f = _head()  # Multi-modal fused head

        # Use Focal Loss as in the original paper
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        self.lr = lr

        # Loss weights
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

        # Metrics tracking
        self.training_metrics = []
        self.validation_metrics = []

    def _caption_enhance(self, src: torch.Tensor, cap: torch.Tensor) -> torch.Tensor:
        """Simple caption guidance by adding caption context."""
        return src + cap

    def forward(self, audio: torch.Tensor, visual: torch.Tensor, caption: torch.Tensor):
        """Forward pass returning logits for each modality and fused predictions."""
        # Project to shared dimension
        a = self.proj_a(audio)
        v = self.proj_v(visual)
        c = self.proj_c(caption)

        # Self-attention encoding
        a = self.enc_a(a)
        v = self.enc_v(v)
        c = self.enc_c(c)

        # Caption enhancement
        a = self._caption_enhance(a, c)
        v = self._caption_enhance(v, c)

        # Fusion
        f = (a + v) / 2.0
        f = self.enc_fusion(f)

        # Per-frame logits
        logit_a = self.head_a(a).squeeze(-1)
        logit_v = self.head_v(v).squeeze(-1)
        logit_f = self.head_f(f).squeeze(-1)

        return logit_a, logit_v, logit_f

    def training_step(self, batch, batch_idx):
        """Training step with comprehensive logging."""
        start_time = time.time()

        # Extract features from dict batch format
        audio = batch['features']['audio']
        visual = batch['features']['visual']
        caption = batch['features']['caption']
        labels = batch['labels']
        seq_mask = batch['sequence_masks']

        # Forward pass
        logit_a, logit_v, logit_f = self(audio, visual, caption)

        # Apply sequence mask to get valid positions
        valid_positions = seq_mask.bool()
        logit_a_valid = logit_a[valid_positions]
        logit_v_valid = logit_v[valid_positions]
        logit_f_valid = logit_f[valid_positions]
        labels_valid = labels[valid_positions]

        # Compute losses using Focal Loss
        loss_mul = self.focal_loss(logit_f_valid, labels_valid)
        loss_a = self.focal_loss(logit_a_valid, labels_valid)
        loss_v = self.focal_loss(logit_v_valid, labels_valid)
        loss_uni = loss_a + loss_v

        # Alignment losses (KL divergence)
        prob_a = torch.sigmoid(logit_a_valid).detach()
        prob_v = torch.sigmoid(logit_v_valid).detach()
        prob_f = torch.sigmoid(logit_f_valid)
        loss_kl = kl_div_bernoulli(prob_v, prob_f) + \
            kl_div_bernoulli(prob_a, prob_f)

        # Total loss
        total_loss = self.lambda1 * loss_uni + \
            self.lambda2 * loss_mul + self.lambda3 * loss_kl

        # Compute metrics
        with torch.no_grad():
            pred_binary = (prob_f > 0.5).float()
            accuracy = (pred_binary == labels_valid).float().mean()

            # Positive predictions
            n_positive_preds = pred_binary.sum().item()
            n_positive_labels = labels_valid.sum().item()
            n_total = len(labels_valid)

        # Log metrics
        metrics = {
            'train/loss_total': total_loss,
            'train/loss_uni': loss_uni,
            'train/loss_mul': loss_mul,
            'train/loss_kl': loss_kl,
            'train/loss_audio': loss_a,
            'train/loss_visual': loss_v,
            'train/accuracy': accuracy,
            'train/positive_pred_ratio': n_positive_preds / n_total if n_total > 0 else 0,
            'train/positive_label_ratio': n_positive_labels / n_total if n_total > 0 else 0,
            'train/step_time': time.time() - start_time,
            'epoch': self.current_epoch,
            'global_step': self.global_step
        }

        # Log to PyTorch Lightning (which should sync to wandb)
        self.log_dict(metrics, prog_bar=True, logger=True,
                      on_step=True, on_epoch=True)

        # Log unique metrics not covered by PyTorch Lightning
        if hasattr(self.logger, 'experiment') and hasattr(self.logger.experiment, 'log'):
            try:
                # Only log learning rate - other metrics are handled by PyTorch Lightning
                unique_metrics = {
                    'batch/learning_rate': self.trainer.optimizers[0].param_groups[0]['lr'] if self.trainer else 1e-3
                }
                self.logger.experiment.log(unique_metrics)
            except Exception as e:
                self.logger_instance.warning(
                    f"Failed to log unique metrics to wandb: {e}")

        # Detailed logging at intervals
        self.step_count += 1
        if self.step_count % self.log_interval == 0:
            step_time = time.time() - start_time
            self.logger_instance.info(
                f"Step {self.step_count} | "
                f"Loss: {total_loss:.4f} | "
                f"Acc: {accuracy:.4f} | "
                f"Pos preds: {n_positive_preds}/{n_total} | "
                f"Time: {step_time:.3f}s"
            )

        return total_loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        # Extract features from dict batch format
        audio = batch['features']['audio']
        visual = batch['features']['visual']
        caption = batch['features']['caption']
        labels = batch['labels']
        seq_mask = batch['sequence_masks']

        # Forward pass
        logit_a, logit_v, logit_f = self(audio, visual, caption)

        # Apply sequence mask
        valid_positions = seq_mask.bool()
        logit_f_valid = logit_f[valid_positions]
        labels_valid = labels[valid_positions]

        val_loss = self.focal_loss(logit_f_valid, labels_valid)

        # Metrics
        prob_f = torch.sigmoid(logit_f_valid)
        pred_binary = (prob_f > 0.5).float()
        accuracy = (pred_binary == labels_valid).float().mean()

        val_metrics = {
            'val/loss': val_loss,
            'val/accuracy': accuracy,
            'epoch': self.current_epoch,
            'global_step': self.global_step
        }

        self.log_dict(val_metrics, prog_bar=True, logger=True,
                      on_step=False, on_epoch=True)

        # Validation metrics are automatically logged by PyTorch Lightning

        return val_loss

    def configure_optimizers(self):
        """Configure optimizer with optional scheduling."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        # Optional: Add learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'train/loss_total',
                'frequency': 1
            }
        }

    def on_before_backward(self, loss):
        """Apply gradient clipping before backward pass."""
        # This is called automatically by PyTorch Lightning when gradient_clip_val is set
        pass


# ==================== Memory Management Callback ====================
class MemoryClearCallback(Callback):
    """Callback to clear memory at regular intervals."""

    def __init__(self, clear_every_n_epochs: int = 1):
        self.clear_every_n_epochs = clear_every_n_epochs
        self.logger = logging.getLogger(self.__class__.__name__)

    def on_train_epoch_end(self, trainer, pl_module):
        """Clear memory at end of epoch."""
        if trainer.current_epoch % self.clear_every_n_epochs == 0:
            self.logger.info(
                f"Clearing memory at epoch {trainer.current_epoch}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Log memory usage if available
            try:
                import psutil
                process = psutil.Process()
                mem_info = process.memory_info()
                self.logger.info(
                    f"Memory usage: {mem_info.rss / 1024**3:.2f} GB")
            except ImportError:
                pass


# ==================== End-of-Epoch Visualization Callback ====================
class EndOfEpochVisualizationCallback(Callback):
    """Callback to create visualizations at end of each epoch for both train and val sets."""

    def __init__(self, train_dataloader, val_dataloader=None, num_samples: int = 10, save_dir: str = "visualizations"):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.num_samples = num_samples
        self.save_dir = save_dir
        self.logger = logging.getLogger(self.__class__.__name__)
        os.makedirs(save_dir, exist_ok=True)

    def on_train_epoch_end(self, trainer, pl_module):
        """Create visualizations at end of epoch."""
        epoch = trainer.current_epoch

        # For debugging, only visualize every 2 epochs to reduce load
        if epoch % 2 != 0:
            self.logger.info(
                f"Skipping visualization for epoch {epoch} (debugging mode - every 2 epochs)")
            return

        self.logger.info(
            f"Creating end-of-epoch visualizations for epoch {epoch}")

        # Set model to eval mode
        pl_module.eval()
        device = next(pl_module.parameters()).device

        with torch.no_grad():
            # Process both train and validation datasets
            datasets_to_viz = []
            if self.train_dataloader:
                datasets_to_viz.append(('train', self.train_dataloader))
            if self.val_dataloader:
                datasets_to_viz.append(('val', self.val_dataloader))

            self.logger.info(
                f"Will process {len(datasets_to_viz)} datasets for visualization")

            for dataset_idx, (dataset_name, dataloader) in enumerate(datasets_to_viz):
                self.logger.info(
                    f"Starting visualization for dataset {dataset_idx+1}/{len(datasets_to_viz)}: {dataset_name}")
                log_memory_usage(
                    self.logger, f"Before {dataset_name} visualization")

                try:
                    sample_count = 0

                    self.logger.info(
                        f"Processing samples from {dataset_name} dataloader with immediate visualization...")

                    for batch_idx, batch in enumerate(dataloader):
                        if sample_count >= self.num_samples:
                            self.logger.info(
                                f"Reached target of {self.num_samples} samples, stopping batch processing")
                            break

                        self.logger.debug(
                            f"Processing batch {batch_idx} for {dataset_name} set")

                        log_memory_usage(
                            self.logger, f"{dataset_name} batch {batch_idx} start")

                        try:
                            # Extract from dict batch format
                            audio = batch['features']['audio'].to(device)
                            visual = batch['features']['visual'].to(device)
                            caption = batch['features']['caption'].to(device)
                            labels = batch['labels'].to(device)
                            seq_mask = batch['sequence_masks']

                            log_memory_usage(
                                self.logger, f"{dataset_name} batch {batch_idx} after data load")

                            # Get predictions
                            logit_a, logit_v, logit_f = pl_module(
                                audio, visual, caption)

                            log_memory_usage(
                                self.logger, f"{dataset_name} batch {batch_idx} after inference")

                        except Exception as e:
                            self.logger.error(
                                f"Error during model inference for {dataset_name} batch {batch_idx}: {e}")
                            log_memory_usage(
                                self.logger, f"{dataset_name} batch {batch_idx} error state")
                            continue

                        # Process each sequence in the batch individually
                        batch_size = logit_f.shape[0]
                        self.logger.debug(
                            f"Processing {batch_size} sequences in batch {batch_idx}")

                        for seq_idx in range(batch_size):
                            if sample_count >= self.num_samples:
                                break

                            try:
                                # Get sequence mask for this specific sequence
                                seq_mask_single = seq_mask[seq_idx]
                                valid_length = int(
                                    seq_mask_single.sum().item())

                                if valid_length == 0:
                                    self.logger.warning(
                                        f"Sequence {seq_idx} has zero valid length, skipping")
                                    continue

                                # Extract predictions and labels for this sequence only
                                logit_f_seq = logit_f[seq_idx, :valid_length]
                                labels_seq = labels[seq_idx, :valid_length]

                                # Convert to numpy for visualization
                                pred_probs = torch.sigmoid(
                                    logit_f_seq).cpu().numpy()
                                labels_np = labels_seq.cpu().numpy()

                                # Extract video ID for this sequence
                                video_id = batch['video_ids'][seq_idx]

                                self.logger.debug(
                                    f"Processing sample {sample_count}: {video_id} ({valid_length} frames)")

                                # Create visualization immediately instead of accumulating data
                                if hasattr(trainer.logger, 'experiment'):
                                    try:
                                        # Create plot
                                        fig, axes = plt.subplots(
                                            2, 1, figsize=(12, 8))
                                        time_points = np.arange(
                                            len(pred_probs))

                                        # Plot 1: Predictions vs Ground Truth
                                        axes[0].plot(time_points, pred_probs, 'b-',
                                                     label='Predicted Probability', alpha=0.7)
                                        positive_idx = labels_np > 0.5
                                        if np.any(positive_idx):
                                            axes[0].scatter(time_points[positive_idx],
                                                            np.ones(
                                                                np.sum(positive_idx)),
                                                            color='red', s=30, label='Ground Truth', zorder=5)
                                        axes[0].set_ylabel('Probability')
                                        dataset_type = dataset_name.upper()
                                        axes[0].set_title(
                                            f'Epoch {epoch} - {dataset_type} Sample {sample_count} - Predictions vs Ground Truth')
                                        axes[0].legend()
                                        axes[0].grid(True, alpha=0.3)
                                        axes[0].set_ylim(-0.1, 1.1)

                                        # Plot 2: Prediction confidence
                                        confidence = np.abs(
                                            pred_probs - 0.5) * 2
                                        axes[1].plot(time_points, confidence, 'g-',
                                                     label='Confidence', alpha=0.7)
                                        axes[1].set_ylabel('Confidence')
                                        axes[1].set_xlabel('Time Steps')
                                        axes[1].set_title(
                                            'Prediction Confidence')
                                        axes[1].legend()
                                        axes[1].grid(True, alpha=0.3)
                                        axes[1].set_ylim(0, 1)

                                        plt.tight_layout()

                                        # Save to file
                                        viz_path = os.path.join(
                                            self.save_dir, f'epoch_{epoch}_{dataset_name}_{video_id}.png')

                                        self.logger.debug(
                                            f"Saving visualization to {viz_path}")
                                        plt.savefig(
                                            viz_path, dpi=120, bbox_inches='tight')

                                        # Create caption
                                        caption = f'Epoch {epoch}, {dataset_type} set, Video {video_id}'

                                        # Log to wandb
                                        try:
                                            trainer.logger.experiment.log({
                                                f"visualizations/{dataset_name}/{video_id}": wandb.Image(viz_path, caption=caption),
                                            })
                                        except Exception as e:
                                            self.logger.error(
                                                f"Error logging wandb image for {video_id}: {e}")

                                        # Close plot and cleanup
                                        plt.close(fig)
                                        del pred_probs, labels_np, fig, axes

                                        self.logger.debug(
                                            f"Completed immediate visualization for {video_id}")

                                    except Exception as e:
                                        self.logger.error(
                                            f"Error creating immediate visualization for {video_id}: {e}")
                                        log_memory_usage(
                                            self.logger, f"After viz {sample_count} error")

                                sample_count += 1

                                # Explicit cleanup of large tensors
                                del logit_f_seq, labels_seq

                            except Exception as e:
                                self.logger.error(
                                    f"Error processing sequence {seq_idx} in {dataset_name}: {e}")
                                continue

                        # Cleanup after each batch
                        try:
                            del audio, visual, caption, labels, seq_mask, logit_f
                        except Exception as cleanup_e:
                            self.logger.warning(
                                f"Error during batch cleanup: {cleanup_e}")
                        log_memory_usage(
                            self.logger, f"{dataset_name} batch {batch_idx} cleanup")

                    self.logger.info(
                        f"Completed immediate processing of {sample_count} samples from {dataset_name} set")

                except Exception as e:
                    self.logger.error(
                        f"Error processing {dataset_name} dataset: {e}")
                    log_memory_usage(
                        self.logger, f"After {dataset_name} dataset error")
                    continue

                finally:
                    # Cleanup between datasets
                    try:
                        plt.close('all')
                        import gc
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        log_memory_usage(
                            self.logger, f"After {dataset_name} dataset cleanup")
                        self.logger.debug(
                            f"Cleaned up resources after {dataset_name} visualization")
                    except Exception as cleanup_error:
                        self.logger.warning(
                            f"Error during cleanup after {dataset_name}: {cleanup_error}")

                self.logger.info(
                    f"Completed visualization for {dataset_name} set")

        # Final cleanup and switch back to training mode
        try:
            plt.close('all')
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            self.logger.warning(f"Error during final cleanup: {e}")

        pl_module.train()
        self.logger.info(f"Visualization callback completed for epoch {epoch}")


# ==================== Visualization Functions ====================
def visualize_predictions(model, dataloader, save_dir: str, num_samples: int = 5, device='cpu'):
    """Visualize model predictions vs ground truth."""
    logger = logging.getLogger("Visualizer")
    logger.info(f"Creating visualizations for {num_samples} samples")

    model.eval()

    # Handle device conversion properly
    if isinstance(device, str):
        if device == 'auto':
            # Get device from model parameters
            device = next(model.parameters()).device
        else:
            device = torch.device(device)

    logger.info(f"Visualization will use device: {device}")
    
    # Only move model if it's not already on the target device
    model_device = next(model.parameters()).device
    if model_device != device:
        logger.info(f"Moving model from {model_device} to {device}")
        model = model.to(device)
    else:
        logger.info(f"Model already on target device {device}")

    os.makedirs(save_dir, exist_ok=True)
    saved_paths = []

    with torch.no_grad():
        sample_count = 0
        logger.info(f"Starting iteration over dataloader...")
        
        for batch_idx, batch in enumerate(dataloader):
            if sample_count >= num_samples:
                logger.info(f"Reached target of {num_samples} samples, stopping")
                break

            try:
                logger.info(f"Processing visualization batch {batch_idx}")
                
                # Get predictions from dict batch format
                audio = batch['features']['audio'].to(device)
                visual = batch['features']['visual'].to(device)
                caption = batch['features']['caption'].to(device)
                labels = batch['labels'].to(device)
                seq_mask = batch['sequence_masks']

                logger.info(f"Batch shapes - audio: {audio.shape}, visual: {visual.shape}, caption: {caption.shape}")
                
                _, _, logit_f = model(audio, visual, caption)
                logger.info(f"Model inference completed, logit_f shape: {logit_f.shape}")
                
            except Exception as batch_error:
                logger.error(f"Error processing batch {batch_idx} in post-training visualization: {batch_error}")
                logger.error(f"Batch error traceback:", exc_info=True)
                continue

            # Process each sequence in the batch individually
            batch_size = logit_f.shape[0]
            for seq_idx in range(batch_size):
                if sample_count >= num_samples:
                    break

                # Get valid length for this sequence
                valid_length = int(seq_mask[seq_idx].sum().item())

                # Extract predictions and labels for this sequence only
                pred_probs = torch.sigmoid(
                    logit_f[seq_idx, :valid_length]).cpu().numpy()
                labels_np = labels[seq_idx, :valid_length].cpu().numpy()

                # Create visualization
                fig, axes = plt.subplots(3, 1, figsize=(15, 10))
                seq_len = len(pred_probs)
                time_points = np.arange(seq_len)

                # Plot 1: Classification scores
                ax1 = axes[0]
                ax1.plot(time_points, pred_probs, 'b-',
                         label='Predicted Prob', alpha=0.7)
                positive_idx = labels_np > 0.5
                if np.any(positive_idx):
                    ax1.scatter(time_points[positive_idx],
                                np.ones(np.sum(positive_idx)),
                                color='red', s=50, label='GT Positive', zorder=5)
                ax1.set_ylabel('Classification Score')
                ax1.set_title(
                    f'Sample {sample_count} - Classification Predictions')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                ax1.set_ylim(-0.1, 1.1)

                # Plot 2: Confidence over time
                ax2 = axes[1]
                confidence = np.abs(pred_probs - 0.5) * 2
                ax2.plot(time_points, confidence, 'g-',
                         label='Confidence', alpha=0.7)
                ax2.set_ylabel('Confidence')
                ax2.set_title('Prediction Confidence')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                ax2.set_ylim(0, 1)

                # Plot 3: Segments visualization
                ax3 = axes[2]

                # Draw predicted segments
                threshold = 0.5
                in_segment = False
                segment_start = 0

                for t in range(seq_len):
                    if pred_probs[t] > threshold and not in_segment:
                        in_segment = True
                        segment_start = t
                    elif pred_probs[t] <= threshold and in_segment:
                        in_segment = False
                        ax3.add_patch(patches.Rectangle(
                            (segment_start, 0.6), t - segment_start, 0.3,
                            facecolor='blue', alpha=0.5
                        ))

                if in_segment:
                    ax3.add_patch(patches.Rectangle(
                        (segment_start, 0.6), seq_len - segment_start, 0.3,
                        facecolor='blue', alpha=0.5
                    ))

                # Draw GT segments
                in_gt_segment = False
                gt_segment_start = 0

                for t in range(seq_len):
                    if labels_np[t] > 0.5 and not in_gt_segment:
                        in_gt_segment = True
                        gt_segment_start = t
                    elif labels_np[t] <= 0.5 and in_gt_segment:
                        in_gt_segment = False
                        ax3.add_patch(patches.Rectangle(
                            (gt_segment_start, 0.1), t - gt_segment_start, 0.3,
                            facecolor='red', alpha=0.5
                        ))

                if in_gt_segment:
                    ax3.add_patch(patches.Rectangle(
                        (gt_segment_start, 0.1), seq_len - gt_segment_start, 0.3,
                        facecolor='red', alpha=0.5
                    ))

                ax3.set_xlim(0, seq_len)
                ax3.set_ylim(0, 1)
                ax3.set_xlabel('Time Steps')
                ax3.set_title('Segments (Blue: Predicted, Red: Ground Truth)')
                ax3.grid(True, alpha=0.3, axis='x')

                blue_patch = patches.Patch(
                    color='blue', alpha=0.5, label='Predicted')
                red_patch = patches.Patch(
                    color='red', alpha=0.5, label='Ground Truth')
                ax3.legend(handles=[blue_patch, red_patch])

                plt.tight_layout()

                # Save figure
                path = os.path.join(
                    save_dir, f'visualization_sample_{sample_count}.png')
                plt.savefig(path, dpi=150, bbox_inches='tight')
                saved_paths.append(path)
                logger.info(f"Saved visualization to {path}")

                plt.close()

                sample_count += 1

    return saved_paths


# ==================== Main Training Function ====================
def main(args):
    """Main training function with comprehensive setup."""
    # Enable Tensor Cores for faster training on compatible GPUs
    torch.set_float32_matmul_precision('medium')

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"train_repurpose_{timestamp}.log"
    logger = setup_logging(args.log_level, log_file)

    logger.info("Starting RepurposeModel training")
    logger.info(f"Arguments: {vars(args)}")

    # Initialize wandb if requested
    wandb_logger = None
    if args.use_wandb:
        try:
            wandb_logger = WandbLogger(
                project=args.wandb_project,
                name=f"repurpose_{timestamp}",
                config=vars(args),
                log_model=False,  # Don't auto-log model checkpoints to save space
                save_dir="./wandb_logs"
            )
            logger.info(f"✓ Initialized wandb project: {args.wandb_project}")
            logger.info(f"✓ Wandb run name: repurpose_{timestamp}")
            logger.info(f"✓ Wandb will log metrics automatically")
        except Exception as e:
            logger.error(f"Failed to initialize wandb: {e}")
            logger.info("Training will continue without wandb logging")
            wandb_logger = None

    # Create data loaders
    logger.info("Creating data loaders...")
    train_dataloader = create_sequence_dataloader(
        feature_dirs={
            'audio': args.audio_dir,
            'visual': args.visual_dir,
            'caption': args.caption_dir
        },
        annotation_file=args.train_annotation,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        min_modalities=3,
        max_seq_length=args.max_seq_len if args.max_seq_len and args.max_seq_len > 0 else None
    )

    val_dataloader = None
    if args.val_annotation:
        val_dataloader = create_sequence_dataloader(
            feature_dirs={
                'audio': args.audio_dir,
                'visual': args.visual_dir,
                'caption': args.caption_dir
            },
            annotation_file=args.val_annotation,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,  # No shuffling for validation
            min_modalities=3,
            max_seq_length=args.max_seq_len if args.max_seq_len and args.max_seq_len > 0 else None
        )

    # Get dimensions from a sample batch
    logger.info("Determining feature dimensions...")
    sample_batch = next(iter(train_dataloader))
    dim_audio = sample_batch['features']['audio'].shape[-1]
    dim_visual = sample_batch['features']['visual'].shape[-1]
    dim_caption = sample_batch['features']['caption'].shape[-1]

    logger.info(
        f"Feature dimensions - Audio: {dim_audio}, Visual: {dim_visual}, Caption: {dim_caption}")

    # Create model
    model = RepurposeModel(
        dim_audio=dim_audio,
        dim_visual=dim_visual,
        dim_caption=dim_caption,
        d_model=args.d_model,
        n_head=args.n_head,
        n_layers=args.n_layers,
        lr=args.learning_rate,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        lambda3=args.lambda3,
        log_interval=args.log_interval
    )

    logger.info(
        f"Created model with {sum(p.numel() for p in model.parameters())} parameters")

    # Setup callbacks
    callbacks = []

    # Memory management
    callbacks.append(MemoryClearCallback(clear_every_n_epochs=1))

    # End-of-epoch visualization for both train and val sets (reduced for debugging)
    viz_callback = EndOfEpochVisualizationCallback(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_samples=args.num_viz_samples,
        save_dir=os.path.join(args.checkpoint_dir, "epoch_visualizations")
    )
    callbacks.append(viz_callback)

    # Model checkpointing (only if enabled)
    if args.enable_checkpointing:
        checkpoint_callback = ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename='repurpose-{epoch:02d}-{train/loss_total:.4f}',
            monitor='train/loss_total',
            mode='min',
            save_top_k=3,
            save_last=True
        )
        callbacks.append(checkpoint_callback)
        logger.info(
            "Checkpointing enabled - models will be saved to: " + args.checkpoint_dir)
    else:
        logger.info("Checkpointing disabled - models will NOT be saved")

    # Early stopping
    if val_dataloader and args.early_stopping_patience > 0:
        early_stopping = EarlyStopping(
            monitor='val/loss',
            patience=args.early_stopping_patience,
            mode='min',
            verbose=True
        )
        callbacks.append(early_stopping)

    # Create trainer
    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        gradient_clip_val=args.gradient_clip,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=callbacks,
        logger=wandb_logger,
        enable_checkpointing=args.enable_checkpointing,
        log_every_n_steps=args.log_interval,
        val_check_interval=args.val_check_interval,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        enable_progress_bar=True,
        deterministic=args.deterministic,
        num_sanity_val_steps=0  # Disable sanity checking to avoid early exit
    )

    # Start training
    logger.info("Starting training...")
    start_time = time.time()

    try:
        if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
            logger.info(f"Resuming training from checkpoint: {args.resume_from_checkpoint}")
            trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=args.resume_from_checkpoint)
        else:
            if args.resume_from_checkpoint:
                logger.warning(f"Checkpoint file not found: {args.resume_from_checkpoint}, starting from scratch")
            trainer.fit(model, train_dataloader, val_dataloader)
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time/60:.2f} minutes")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        logger.error("Full traceback:", exc_info=True)
        # Still try to run post-training steps if requested
        training_time = time.time() - start_time
        logger.info(f"Training stopped after {training_time/60:.2f} minutes")

    # Create visualizations (skip if resumed from checkpoint to avoid dataloader issues)
    if args.create_visualizations:
        if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint or ""):
            logger.info("Skipping post-training visualizations when resuming from checkpoint (known dataloader issue)")
        else:
            logger.info("Creating visualizations...")
            viz_dir = os.path.join(args.checkpoint_dir, "visualizations")

            try:
                # Determine actual device from model
                actual_device = next(model.parameters()).device
                logger.info(f"Using device for visualization: {actual_device}")
                
                # Add memory logging before visualization
                log_memory_usage(logger, "Before post-training visualization")
                
                # Use validation dataloader if available, otherwise training dataloader
                viz_dataloader = val_dataloader or train_dataloader
                logger.info(f"Using {'validation' if val_dataloader else 'training'} dataloader for visualization")

                visualize_predictions(
                    model,
                    viz_dataloader,
                    viz_dir,
                    num_samples=args.num_viz_samples,
                    device=actual_device
                )
                
                log_memory_usage(logger, "After post-training visualization")
                logger.info("Post-training visualizations completed successfully")
                
            except Exception as viz_error:
                logger.error(f"Error during post-training visualization: {viz_error}")
                logger.error("Full visualization traceback:", exc_info=True)
                log_memory_usage(logger, "After post-training visualization error")
                logger.info("Training completed successfully despite visualization error")

    # Final cleanup
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Training script completed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train RepurposeModel with logging and wandb")

    # Data arguments
    parser.add_argument("--audio_dir", type=str, required=True,
                        help="Path to audio features directory")
    parser.add_argument("--visual_dir", type=str, required=True,
                        help="Path to visual features directory")
    parser.add_argument("--caption_dir", type=str, required=True,
                        help="Path to caption features directory")
    parser.add_argument("--train_annotation", type=str,
                        required=True, help="Path to training annotation JSON")
    parser.add_argument("--val_annotation", type=str,
                        help="Path to validation annotation JSON")

    # Model arguments
    parser.add_argument("--d_model", type=int, default=128,
                        help="Model dimension")
    parser.add_argument("--n_head", type=int, default=4,
                        help="Number of attention heads")
    parser.add_argument("--n_layers", type=int, default=2,
                        help="Number of transformer layers")
    parser.add_argument("--max_seq_len", type=int, default=None,
                        help="Maximum sequence length (None for full videos)")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--learning_rate", type=float,
                        default=1e-3, help="Learning rate")
    parser.add_argument("--gradient_clip", type=float,
                        default=1.0, help="Gradient clipping value")
    parser.add_argument("--accumulate_grad_batches", type=int,
                        default=1, help="Gradient accumulation steps")

    # Loss weights
    parser.add_argument("--lambda1", type=float, default=0.1,
                        help="Weight for uni-modal loss")
    parser.add_argument("--lambda2", type=float, default=0.3,
                        help="Weight for multi-modal loss")
    parser.add_argument("--lambda3", type=float, default=0.1,
                        help="Weight for KL divergence loss")

    # Hardware arguments
    parser.add_argument("--accelerator", type=str, default="auto",
                        help="Accelerator type (cpu, gpu, auto)")
    parser.add_argument("--devices", type=int, default=1,
                        help="Number of devices")
    parser.add_argument("--precision", type=str, default="32",
                        help="Training precision (16, 32, bf16)")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of data loader workers")

    # Logging arguments
    parser.add_argument("--log_level", type=str,
                        default="INFO", help="Logging level")
    parser.add_argument("--log_interval", type=int,
                        default=10, help="Log every N steps")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Use Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str,
                        default="repurpose", help="W&B project name")

    # Checkpointing
    parser.add_argument("--checkpoint_dir", type=str,
                        default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--enable_checkpointing", action="store_true",
                        help="Enable model checkpointing (default: disabled)")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint file for resuming training")
    parser.add_argument("--early_stopping_patience", type=int,
                        default=5, help="Early stopping patience")

    # Validation
    parser.add_argument("--val_check_interval", type=float,
                        default=1.0, help="Validation check interval")
    parser.add_argument("--limit_train_batches", type=float,
                        default=1.0, help="Limit training batches")
    parser.add_argument("--limit_val_batches", type=float,
                        default=1.0, help="Limit validation batches")

    # Visualization
    parser.add_argument("--create_visualizations", action="store_true",
                        help="Create prediction visualizations")
    parser.add_argument("--num_viz_samples", type=int,
                        default=5, help="Number of samples to visualize")

    # Misc
    parser.add_argument("--deterministic", action="store_true",
                        help="Use deterministic training")

    args = parser.parse_args()

    # Example command line usage:
    # python train_repurpose.py \
    #   --audio_dir /path/to/audio \
    #   --visual_dir /path/to/visual \
    #   --caption_dir /path/to/caption \
    #   --train_annotation /path/to/train.json \
    #   --val_annotation /path/to/val.json \
    #   --use_wandb \
    #   --create_visualizations \
    #   --epochs 20

    main(args)
