#!/usr/bin/env python3
"""
Standalone training script for RepurposeModel with comprehensive logging and wandb integration.
Incorporates memory optimizations and visualization capabilities.
"""

from models.losses import sigmoid_focal_loss, ctr_diou_loss_1d
from models.softnms import soft_nms_intervals_cpu
from models.transformer import (
    PositionalEncoding,
    EncoderLayer,
    CrossAttentionEncoderLayer,
    CrossSelfEncoderLayer,
)
from utils.metrics import calculate_tiou
import os
import gc
import sys
import time
import math
import argparse
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
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

# Global inference settings - used in validation and visualization
INFERENCE_SETTINGS = {
    "pre_nms_topk": 1000,
    "pre_nms_thresh": 0.5,
    "duration_thresh": 10,
    "duration_thresh_max": 90,
    "max_seg_per_min": 0.3,
    "nms_sigma": 0.5,
    "min_score": 0.01,
}

# Configure logging


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup comprehensive logging configuration."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, log_level.upper()), format=log_format, handlers=handlers
    )

    # Set specific logger levels
    logging.getLogger("pytorch_lightning").setLevel(logging.INFO)
    logging.getLogger("wandb").setLevel(logging.INFO)

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
            logger.info(
                f"{stage} | CPU: {cpu_used_gb:.1f}/{cpu_total_gb:.1f}GB ({cpu_percent:.1f}%) | GPU: {gpu_allocated:.1f}GB alloc, {gpu_reserved:.1f}GB reserved, {gpu_max_allocated:.1f}GB max"
            )
        else:
            logger.info(
                f"{stage} | CPU: {cpu_used_gb:.1f}/{cpu_total_gb:.1f}GB ({cpu_percent:.1f}%) | GPU: N/A"
            )
    except Exception as e:
        logger.warning(f"Failed to log memory usage at {stage}: {e}")


# ==================== Loss Functions ====================
# Import focal loss from existing models/losses.py


@torch.no_grad()
def _kl_div_bernoulli(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-6):
    """Element-wise KL divergence for Bernoulli distributions."""
    p = p.clamp(eps, 1 - eps)
    q = q.clamp(eps, 1 - eps)
    return p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))


def kl_div_bernoulli(p: torch.Tensor, q: torch.Tensor):
    """KL divergence between Bernoulli distributions - using sum to match other losses."""
    return _kl_div_bernoulli(p, q).sum()


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
        d_model: int = 512,
        n_head: int = 8,
        n_self_attn_layers: int = 3,  # Self-attention layers per modality
        n_cross_attn_layers: int = 3,  # Cross-attention layers for A-C and V-C
        n_fusion_layers: int = 3,  # Audio-Visual fusion layers
        lr: float = 1e-4,
        weight_decay: float = 1e-4,  # Weight decay for optimizer
        beta1: float = 0.9,  # Beta1 for AdamW
        beta2: float = 0.98,  # Beta2 for AdamW (lower than default for faster adaptation)
        warmup_epochs: int = 1,
        lambda1: float = 0.1,
        lambda2: float = 0.3,
        lambda3: float = 0.1,
        lambda4: float = 0.7,  # Weight for regression loss
        log_interval: int = 10,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Logging
        self.logger_instance = logging.getLogger(self.__class__.__name__)
        self.log_interval = log_interval
        self.step_count = 0

        # Initialize LR scheduler parameters with defaults (will be updated in on_train_start)
        self.warmup_steps = 1000  # Default fallback
        self.total_steps = 10000  # Default fallback

        # Projections to shared dimension - MLPs as per paper with 2048 hidden dim
        # "We then use three distinct MLP layers to map these features to a unified dimension d"
        def _projection_mlp(input_dim):
            return nn.Sequential(
                nn.Linear(input_dim, 2048), nn.ReLU(), nn.Linear(2048, d_model)
            )

        self.proj_a = _projection_mlp(dim_audio)
        self.proj_v = _projection_mlp(dim_visual)
        self.proj_c = _projection_mlp(dim_caption)

        # Add positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len=5000)

        # Self-attention encoders (per modality) - using Pre-LN architecture
        def _encoder():
            return nn.ModuleList(
                [
                    EncoderLayer(d_model, n_head, d_ff=2048, dropout=0.1)
                    for _ in range(n_self_attn_layers)
                ]
            )

        self.enc_a = _encoder()
        self.enc_v = _encoder()
        self.enc_c = _encoder()

        # Multi-layer cross-attention for A-C and V-C modalities using CrossSelfEncoderLayer
        # This includes self-attention followed by cross-attention as in the original
        self.cross_attn_ac_layers = nn.ModuleList(
            [
                CrossSelfEncoderLayer(d_model, n_head, d_ff=2048, dropout=0.1)
                for _ in range(n_cross_attn_layers)
            ]
        )
        self.cross_attn_vc_layers = nn.ModuleList(
            [
                CrossSelfEncoderLayer(d_model, n_head, d_ff=2048, dropout=0.1)
                for _ in range(n_cross_attn_layers)
            ]
        )

        # Multi-layer Audio-Visual fusion cross-attention using CrossAttentionEncoderLayer
        self.vis_aud_cross_att = nn.ModuleList(
            [
                CrossAttentionEncoderLayer(d_model, n_head, d_ff=2048, dropout=0.1)
                for _ in range(n_fusion_layers)
            ]
        )
        self.aud_vis_cross_att = nn.ModuleList(
            [
                CrossAttentionEncoderLayer(d_model, n_head, d_ff=2048, dropout=0.1)
                for _ in range(n_fusion_layers)
            ]
        )

        # Fusion projection to map concatenated features to lower dimension
        # As per paper: "concatenated, mapped to lower dimensions"
        self.fusion_projection = nn.Linear(d_model * 2, d_model)

        # Classification heads - 3-layer MLP as per paper
        # Note: sigmoid will be applied in loss function, not here
        def _head():
            return nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, d_model // 4),
                nn.ReLU(),
                nn.Linear(d_model // 4, 1),
                # No manual bias initialization - let PyTorch handle defaults
            )

        self.head_a = _head()
        self.head_v = _head()
        self.head_f = _head()  # Multi-modal fused head

        # Regression heads - 3-layer MLP with ReLU at the end (as per paper)
        # "incorporating a ReLU activation layer at its final stage"
        def _regression_head():
            return nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, d_model // 4),
                nn.ReLU(),
                # 2 outputs: left and right offsets
                nn.Linear(d_model // 4, 2),
                nn.ReLU(),  # Final ReLU as specified in paper
            )

        self.reg_head_f = _regression_head()  # Multi-modal fused regression head

        # Use less conservative focal loss parameters that worked well before
        # alpha=0.5 (balanced), gamma=1.0 (moderate down-weighting)
        # These should work even better now with aligned data
        self.focal_alpha = 0.7
        self.focal_gamma = 2.0
        self.lr = lr

        # Loss weights
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda4 = lambda4  # Regression loss weight

        # Metrics tracking
        self.training_metrics = []
        self.validation_metrics = []

        # Storage for validation outputs (PyTorch Lightning v2.0+ compatibility)
        self.validation_outputs = []

        # SIMPLE TRANSFORMER FOR TESTING - minimal architecture
        simple_d_model = d_model
        simple_d_ff = (
            simple_d_model  # Instead of 4:1 ratio, use same dimension for feedforward
        )
        simple_nhead = n_head
        simple_num_layers = n_self_attn_layers

        # Read feature dimension from environment variable (for dimension testing)
        import os

        feature_dim_override = os.environ.get("FEATURE_DIM", None)
        if feature_dim_override:
            input_dim = int(feature_dim_override)
            self.logger_instance.info(f"Using custom feature dimension: {input_dim}")
        else:
            input_dim = dim_visual

        self.simple_v_proj = nn.Linear(input_dim, simple_d_model)

        # Use smaller initialization for higher dimensional inputs to prevent gradient issues
        if feature_dim_override and input_dim > 1:
            with torch.no_grad():
                # Scale down initialization based on input dimension
                scale_factor = 1.0 / np.sqrt(input_dim)
                self.simple_v_proj.weight.data *= scale_factor
                self.logger_instance.info(
                    f"Scaled input projection weights by {scale_factor:.4f}"
                )

        self.simple_pos_embed = nn.Parameter(
            torch.randn(1, 2000, simple_d_model) * 0.01
        )
        self.simple_v_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=simple_d_model,
                nhead=simple_nhead,
                dim_feedforward=simple_d_ff,
                batch_first=True,
            ),
            num_layers=simple_num_layers,
        )

        # Caption processing components (matching visual pattern)
        self.simple_c_proj = nn.Linear(dim_caption, simple_d_model)
        self.simple_c_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=simple_d_model,
                nhead=simple_nhead,
                dim_feedforward=simple_d_ff,
                batch_first=True,
            ),
            num_layers=simple_num_layers,
        )

        # Audio processing components
        self.simple_a_proj = nn.Linear(dim_audio, simple_d_model)
        self.simple_a_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=simple_d_model,
                nhead=simple_nhead,
                dim_feedforward=simple_d_ff,
                batch_first=True,
            ),
            num_layers=simple_num_layers,
        )

        # Visual-Caption cross-attention layers (using existing CrossSelfEncoderLayer)
        self.simple_cross_attn_vc = nn.ModuleList(
            [
                CrossSelfEncoderLayer(
                    simple_d_model, simple_nhead, d_ff=simple_d_ff, dropout=0.1
                )
                for _ in range(n_cross_attn_layers)
            ]
        )

        # Audio-Caption cross-attention layers
        self.simple_cross_attn_ac = nn.ModuleList(
            [
                CrossSelfEncoderLayer(
                    simple_d_model, simple_nhead, d_ff=simple_d_ff, dropout=0.1
                )
                for _ in range(n_cross_attn_layers)
            ]
        )

        # Visual-Audio fusion layers (cross-attention between enhanced V and enhanced A)
        self.simple_fusion_va = nn.ModuleList(
            [
                CrossAttentionEncoderLayer(
                    simple_d_model, simple_nhead, d_ff=simple_d_ff, dropout=0.1
                )
                for _ in range(n_fusion_layers)
            ]
        )

        self.simple_fusion_av = nn.ModuleList(
            [
                CrossAttentionEncoderLayer(
                    simple_d_model, simple_nhead, d_ff=simple_d_ff, dropout=0.1
                )
                for _ in range(n_fusion_layers)
            ]
        )

        self.simple_output = nn.Sequential(
            nn.Linear(simple_d_model, simple_d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(simple_d_model // 2, 2),
        )

        # Add simple classification head
        self.simple_classifier = nn.Sequential(
            nn.Linear(simple_d_model, simple_d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(
                simple_d_model // 2, 1
            ),  # Single output for binary classification
        )

        # Add visual unimodal classification head
        self.simple_visual_classifier = nn.Sequential(
            nn.Linear(simple_d_model, simple_d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(
                simple_d_model // 2, 1
            ),  # Single output for binary classification
        )

        # Add audio unimodal classification head
        self.simple_audio_classifier = nn.Sequential(
            nn.Linear(simple_d_model, simple_d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(
                simple_d_model // 2, 1
            ),  # Single output for binary classification
        )

    def forward(
        self,
        audio: torch.Tensor,
        visual: torch.Tensor,
        caption: torch.Tensor,
        mask: torch.Tensor,
    ):
        """SIMPLE TRANSFORMER - Full multi-modal fusion with V-C, A-C, and V-A cross-attention."""
        batch_size, seq_len = visual.shape[:2]

        # Process visual features
        v = self.simple_v_proj(visual)
        v = v + self.simple_pos_embed[:, :seq_len, :]

        # Convert mask to attention mask (True = ignore)
        attn_mask = ~mask.bool()
        v = self.simple_v_encoder(v, src_key_padding_mask=attn_mask)

        # Process caption features
        c = self.simple_c_proj(caption)
        c = c + self.simple_pos_embed[:, :seq_len, :]  # Reuse pos embeddings
        c = self.simple_c_encoder(c, src_key_padding_mask=attn_mask)

        # Process audio features
        a = self.simple_a_proj(audio)
        a = a + self.simple_pos_embed[:, :seq_len, :]  # Reuse pos embeddings
        a = self.simple_a_encoder(a, src_key_padding_mask=attn_mask)

        # Visual-Caption cross-attention
        v_enhanced = v
        for layer in self.simple_cross_attn_vc:
            v_enhanced = layer(v_enhanced, c, mask=mask)

        # Audio-Caption cross-attention
        a_enhanced = a
        for layer in self.simple_cross_attn_ac:
            a_enhanced = layer(a_enhanced, c, mask=mask)

        # Visual-Audio fusion (cross-attention between enhanced features)
        v_fused_features = v_enhanced
        for layer in self.simple_fusion_va:
            v_fused_features = layer(v_fused_features, a_enhanced, mask=mask)

        a_fused_features = a_enhanced
        for layer in self.simple_fusion_av:
            a_fused_features = layer(a_fused_features, v_enhanced, mask=mask)

        # Concatenate bidirectional cross-attended features as per paper
        fused_features = torch.cat(
            [v_fused_features, a_fused_features], dim=-1
        )  # [B, T, 2*d_model]

        fused_features = self.fusion_projection(fused_features)

        # Get offset predictions using fused features
        # Clamp to ensure positive offsets, avoiding gradient stopping issues of ReLU
        offset_f = self.simple_output(fused_features).clamp(min=1e-4)  # [B, T, 2]

        # Get classification predictions using fused features
        logit_f = self.simple_classifier(fused_features).squeeze(-1)  # [B, T]

        # Get visual unimodal classification predictions
        logit_v = self.simple_visual_classifier(v_enhanced).squeeze(-1)  # [B, T]

        # Get audio unimodal classification predictions
        logit_a = self.simple_audio_classifier(a_enhanced).squeeze(-1)  # [B, T]

        return logit_a, logit_v, logit_f, offset_f

        """
        # ORIGINAL COMPLEX FORWARD PASS - COMMENTED OUT FOR TESTING
        # Forward pass with cross-attention between modalities.
        # TEMPORARY: Zero out audio and caption for testing
        audio = torch.zeros_like(audio)
        caption = torch.zeros_like(caption)
        
        # Project to shared dimension
        a = self.proj_a(audio)
        v = self.proj_v(visual)
        c = self.proj_c(caption)

        # Add positional encoding
        # The PositionalEncoding expects [seq_len, batch, d_model] but we have [batch, seq_len, d_model]
        a = a.transpose(0, 1)
        v = v.transpose(0, 1)
        c = c.transpose(0, 1)

        a = self.pos_encoding(a)
        v = self.pos_encoding(v)
        c = self.pos_encoding(c)

        # Transpose back to [batch, seq_len, d_model]
        a = a.transpose(0, 1)
        v = v.transpose(0, 1)
        c = c.transpose(0, 1)

        # Self-attention encoding for each modality
        for layer in self.enc_a:
            a = layer(a, mask=mask)
        for layer in self.enc_v:
            v = layer(v, mask=mask)
        for layer in self.enc_c:
            c = layer(c, mask=mask)

        # Multi-layer Cross-attention: Audio-Caption
        # CrossSelfEncoderLayer does self-attention + cross-attention internally
        a_enhanced = a
        for layer in self.cross_attn_ac_layers:
            a_enhanced = layer(a_enhanced, c, mask=mask)

        aud_feats = a_enhanced

        for idx, layer in enumerate(self.vis_aud_cross_att):
            vis_feats = layer(vis_feats, aud_feats, mask=mask)

        for idx, layer in enumerate(self.aud_vis_cross_att):
            aud_feats = layer(aud_feats, vis_feats, mask=mask)

        # Concatenate bidirectional cross-attended features as per paper
        f = torch.cat([vis_feats, aud_feats], dim=-1)  # [B, T, 2*d_model]

        # Map to lower dimensions as specified in paper
        # This projection replaces the self-attention encoder
        f = self.fusion_projection(f)  # [B, T, d_model]

        # Per-frame classification logits
        # Use cross-attended features for unimodal heads as per paper
        logit_a = self.head_a(aud_feats).squeeze(-1)
        logit_v = self.head_v(vis_feats).squeeze(-1)
        logit_f = self.head_f(f).squeeze(-1)

        # Per-frame regression offsets (left_offset, right_offset) - only multimodal
        offset_f = self.reg_head_f(f)  # [B, T, 2]

        return logit_a, logit_v, logit_f, offset_f
        """

    def training_step(self, batch, batch_idx):
        """Training step with comprehensive logging."""
        start_time = time.time()

        # Extract features from dict batch format
        audio = batch["features"]["audio"]
        visual = batch["features"]["visual"]
        caption = batch["features"]["caption"]
        labels = batch["labels"]
        offsets = batch["offsets"]  # Ground truth regression offsets [B, T, 2]
        seq_mask = batch["sequence_masks"]

        # Debug: Save first batch data to CSV
        if batch_idx == 0 and self.current_epoch == 0:
            self._save_batch_debug_csv(batch, labels, offsets, seq_mask)

        # Forward pass - now returns both classification and regression outputs
        logit_a, logit_v, logit_f, offset_f = self(
            audio, visual, caption, mask=seq_mask
        )

        # Compute losses following original paper implementation exactly
        # 1. Classification losses - compute for all positions first, then mask
        loss_mul_all = sigmoid_focal_loss(
            logit_f,
            labels,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            reduction="none",
        )

        # Visual unimodal loss
        loss_v_all = sigmoid_focal_loss(
            logit_v,
            labels,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            reduction="none",
        )

        # Audio unimodal loss
        loss_a_all = sigmoid_focal_loss(
            logit_a,
            labels,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            reduction="none",
        )

        # Apply sequence mask and normalize by valid positions
        num_valid = seq_mask.sum().clamp(min=1)  # Avoid division by zero
        loss_mul = (loss_mul_all * seq_mask).sum() / num_valid
        loss_v = (loss_v_all * seq_mask).sum() / num_valid
        loss_a = (loss_a_all * seq_mask).sum() / num_valid
        loss_uni = loss_a + loss_v

        # 2. Regression loss - following original paper implementation exactly
        # Compute regression loss for all positions
        reg_loss_f_all = ctr_diou_loss_1d(offset_f, offsets, reduction="none")  # [B, T]

        # Create combined mask: sequence mask AND positive label mask
        cls_mask = (labels > 0.5).float()
        combined_mask = seq_mask * cls_mask

        # Apply combined mask and normalize by number of positive positions
        num_positive = combined_mask.sum().clamp(min=1)  # Avoid division by zero
        reg_loss_f = (reg_loss_f_all * combined_mask).sum() / num_positive

        # Debug: Save loss details for first batch of each epoch
        if batch_idx == 0:
            self._save_loss_debug_csv(
                reg_loss_f_all,
                cls_mask,
                combined_mask,
                reg_loss_f,
                offset_f,
                offsets,
                epoch=self.current_epoch,
            )

        # Alignment losses (KL divergence) - apply to valid positions only
        valid_positions = seq_mask.bool()
        prob_v = torch.sigmoid(
            logit_v[valid_positions]
        ).detach()  # Detach to prevent gradient flow
        prob_a = torch.sigmoid(
            logit_a[valid_positions]
        ).detach()  # Detach to prevent gradient flow
        prob_f = torch.sigmoid(logit_f[valid_positions])

        # KL divergence from unimodal to fusion (teacher-student style)
        # Normalize by number of valid positions for consistency with other losses
        num_valid_kl = valid_positions.sum().clamp(min=1)
        loss_kl_v = kl_div_bernoulli(prob_v, prob_f) / num_valid_kl
        loss_kl_a = kl_div_bernoulli(prob_a, prob_f) / num_valid_kl

        # Total KL loss
        loss_kl = loss_kl_v + loss_kl_a

        # Total loss - includes classification and regression components
        total_loss = (
            self.lambda1 * loss_uni
            + self.lambda2 * loss_mul
            + self.lambda3 * loss_kl
            + self.lambda4 * reg_loss_f
        )

        # Compute metrics
        with torch.no_grad():
            valid_positions = seq_mask.bool()
            labels_valid = labels[valid_positions]
            logit_f_valid = logit_f[valid_positions]
            prob_f = torch.sigmoid(logit_f_valid)
            pred_binary = (prob_f > 0.5).float()
            accuracy = (pred_binary == labels_valid).float().mean()

            # Positive predictions
            n_positive_preds = pred_binary.sum().item()
            n_positive_labels = labels_valid.sum().item()
            n_total = len(labels_valid)

            # Debug offset statistics
            offset_min = offset_f.min().item()
            offset_max = offset_f.max().item()
            offset_mean = offset_f.mean().item()

        # Log metrics
        metrics = {
            "train/loss_total": total_loss,
            "train/loss_uni": loss_uni,
            "train/loss_mul": loss_mul,
            "train/loss_kl": loss_kl,
            "train/loss_audio": loss_a,
            "train/loss_visual": loss_v,
            "train/reg_loss_f": reg_loss_f,
            "train/accuracy": accuracy,
            "train/positive_pred_ratio": (
                n_positive_preds / n_total if n_total > 0 else 0
            ),
            "train/positive_label_ratio": (
                n_positive_labels / n_total if n_total > 0 else 0
            ),
            "train/offset_min": offset_min,
            "train/offset_max": offset_max,
            "train/offset_mean": offset_mean,
            "train/step_time": time.time() - start_time,
            # Log current LR
            "train/learning_rate": self.optimizers().param_groups[0]["lr"],
            "epoch": self.current_epoch,
            "global_step": self.global_step,
        }

        # Log to PyTorch Lightning (which should sync to wandb)
        self.log_dict(metrics, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        # Log unique metrics not covered by PyTorch Lightning
        if hasattr(self.logger, "experiment") and hasattr(
            self.logger.experiment, "log"
        ):
            try:
                # Only log learning rate - other metrics are handled by PyTorch Lightning
                unique_metrics = {
                    "batch/learning_rate": (
                        self.trainer.optimizers[0].param_groups[0]["lr"]
                        if self.trainer
                        else 1e-3
                    )
                }
                self.logger.experiment.log(unique_metrics)
            except Exception as e:
                self.logger_instance.warning(
                    f"Failed to log unique metrics to wandb: {e}"
                )

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
        audio = batch["features"]["audio"]
        visual = batch["features"]["visual"]
        caption = batch["features"]["caption"]
        labels = batch["labels"]
        offsets = batch["offsets"]  # Ground truth regression offsets [B, T, 2]
        # Ground truth segments from annotations
        gt_segments = batch["gt_segments"]
        seq_mask = batch["sequence_masks"]

        # Forward pass - now returns both classification and regression outputs
        logit_a, logit_v, logit_f, offset_f = self(
            audio, visual, caption, mask=seq_mask
        )

        # Classification losses - multimodal
        val_loss_mul_all = sigmoid_focal_loss(
            logit_f,
            labels,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            reduction="none",
        )

        # Visual unimodal loss
        val_loss_v_all = sigmoid_focal_loss(
            logit_v,
            labels,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            reduction="none",
        )

        # Audio unimodal loss
        val_loss_a_all = sigmoid_focal_loss(
            logit_a,
            labels,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            reduction="none",
        )

        # Apply sequence mask and normalize by valid positions
        num_valid = seq_mask.sum().clamp(min=1)
        val_loss_mul = (val_loss_mul_all * seq_mask).sum() / num_valid
        val_loss_v = (val_loss_v_all * seq_mask).sum() / num_valid
        val_loss_a = (val_loss_a_all * seq_mask).sum() / num_valid
        val_loss_uni = val_loss_a + val_loss_v

        # Regression loss - following original paper implementation exactly
        val_loss_reg_all = ctr_diou_loss_1d(
            offset_f, offsets, reduction="none"
        )  # [B, T]
        cls_mask = (labels > 0.5).float()
        combined_mask = seq_mask * cls_mask
        # Normalize by positive positions
        num_positive = combined_mask.sum().clamp(min=1)
        val_loss_reg = (val_loss_reg_all * combined_mask).sum() / num_positive

        # Alignment losses (KL divergence) - apply to valid positions only
        valid_positions = seq_mask.bool()
        prob_v = torch.sigmoid(
            logit_v[valid_positions]
        ).detach()  # Detach to prevent gradient flow
        prob_a = torch.sigmoid(
            logit_a[valid_positions]
        ).detach()  # Detach to prevent gradient flow
        prob_f = torch.sigmoid(logit_f[valid_positions])

        # KL divergence from unimodal to fusion (teacher-student style)
        # Normalize by number of valid positions for consistency with other losses
        num_valid_kl = valid_positions.sum().clamp(min=1)
        val_loss_kl_v = kl_div_bernoulli(prob_v, prob_f) / num_valid_kl
        val_loss_kl_a = kl_div_bernoulli(prob_a, prob_f) / num_valid_kl
        val_loss_kl = val_loss_kl_v + val_loss_kl_a

        # Total validation loss - matching training loss calculation
        val_loss = (
            self.lambda1 * val_loss_uni
            + self.lambda2 * val_loss_mul
            + self.lambda3 * val_loss_kl
            + self.lambda4 * val_loss_reg
        )

        # Metrics - compute for valid positions only (reuse valid_positions from above)
        logit_f_valid = logit_f[valid_positions]
        labels_valid = labels[valid_positions]
        prob_f = torch.sigmoid(logit_f_valid)
        pred_binary = (prob_f > 0.5).float()
        accuracy = (pred_binary == labels_valid).float().mean()

        val_metrics = {
            "val/loss": val_loss,
            "val/loss_uni": val_loss_uni,
            "val/loss_mul": val_loss_mul,
            "val/loss_kl": val_loss_kl,
            "val/loss_audio": val_loss_a,
            "val/loss_visual": val_loss_v,
            "val/reg_loss_f": val_loss_reg,
            "val/accuracy": accuracy,
            "epoch": self.current_epoch,
            "global_step": self.global_step,
        }

        self.log_dict(
            val_metrics, prog_bar=True, logger=True, on_step=False, on_epoch=True
        )

        # Calculate tIoU metrics using proper inference with soft NMS
        batch_tiou_data = []

        # Define inference settings similar to main.py
        # Get predictions using inference method
        predictions = self.inference_(batch, INFERENCE_SETTINGS)

        # Log detailed segment statistics
        batch_pred_count = 0
        batch_pred_durations = []
        batch_gt_count = 0
        batch_gt_durations = []

        for sample_idx in range(len(predictions)):
            pred_data = predictions[sample_idx]
            sample_gt_segments = gt_segments[sample_idx]

            # Count ground truth segments and durations
            if sample_gt_segments:
                batch_gt_count += len(sample_gt_segments)
                for gt_seg in sample_gt_segments:
                    batch_gt_durations.append(float(gt_seg[1] - gt_seg[0]))

            # Count predicted segments and durations
            if "segments" in pred_data and len(pred_data["segments"]) > 0:
                predicted_segments = pred_data["segments"].cpu().numpy().tolist()
                batch_pred_count += len(predicted_segments)
                for pred_seg in predicted_segments:
                    batch_pred_durations.append(float(pred_seg[1] - pred_seg[0]))

        # Calculate statistics
        pred_mean_duration = (
            sum(batch_pred_durations) / len(batch_pred_durations)
            if batch_pred_durations
            else 0.0
        )
        gt_mean_duration = (
            sum(batch_gt_durations) / len(batch_gt_durations)
            if batch_gt_durations
            else 0.0
        )

        pred_total_duration = sum(batch_pred_durations)
        gt_total_duration = sum(batch_gt_durations)

        duration_ratio = (
            pred_total_duration / gt_total_duration if gt_total_duration > 0 else 0.0
        )
        count_ratio = batch_pred_count / batch_gt_count if batch_gt_count > 0 else 0.0

        # Log segment statistics
        segment_stats = {
            "val/pred_segments_count": float(batch_pred_count),
            "val/gt_segments_count": float(batch_gt_count),
            "val/pred_mean_duration": pred_mean_duration,
            "val/gt_mean_duration": gt_mean_duration,
            "val/duration_ratio_pred_vs_gt": duration_ratio,
            "val/count_ratio_pred_vs_gt": count_ratio,
        }

        # Log to both logger and validation metrics
        self.log_dict(segment_stats, prog_bar=False, logger=True)

        # Calculate tIoU for each sample
        batch_size = len(predictions)
        for sample_idx in range(batch_size):
            pred_data = predictions[sample_idx]
            sample_gt_segments = gt_segments[sample_idx]

            # Extract predicted segments
            if "segments" in pred_data and len(pred_data["segments"]) > 0:
                predicted_segments = pred_data["segments"].cpu().numpy().tolist()

                # Calculate tIoU if we have both predictions and ground truth
                if predicted_segments and sample_gt_segments:
                    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
                    precision_per_threshold = calculate_tiou(
                        sample_gt_segments, predicted_segments, thresholds
                    )
                    batch_tiou_data.append(precision_per_threshold)

        # Store validation outputs for epoch-end aggregation (PyTorch Lightning v2.0+ style)
        validation_output = {"val_loss": val_loss, "tiou_data": batch_tiou_data}
        self.validation_outputs.append(validation_output)

        return val_loss

    def on_validation_epoch_start(self):
        """Clear validation outputs at start of epoch."""
        self.validation_outputs = []

    def on_validation_epoch_end(self):
        """Aggregate validation metrics across all batches."""
        # Collect all tIoU data from validation steps
        all_tiou_data = []
        for output in self.validation_outputs:
            if "tiou_data" in output:
                all_tiou_data.extend(output["tiou_data"])

        # Calculate average tIoU metrics if we have data
        if all_tiou_data:
            thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
            tiou_metrics = {}

            for threshold in thresholds:
                # Average precision across all samples
                threshold_precisions = [
                    sample_data[threshold]
                    for sample_data in all_tiou_data
                    if threshold in sample_data
                ]
                if threshold_precisions:
                    avg_precision = sum(threshold_precisions) / len(
                        threshold_precisions
                    )
                    tiou_metrics[f"val/tIoU@{threshold}"] = avg_precision

            # Calculate Average tIoU (AtIoU)
            if tiou_metrics:
                atiou = sum(tiou_metrics.values()) / len(tiou_metrics)
                tiou_metrics["val/AtIoU"] = atiou

                # Log tIoU metrics
                self.log_dict(tiou_metrics, prog_bar=True, logger=True)

                # Log to console
                self.logger_instance.info(f"Validation tIoU metrics: {tiou_metrics}")

        # Clear outputs after processing
        self.validation_outputs = []

    @torch.no_grad()
    def inference_single_video(self, logit_f, offset_f, seq_mask, inference_settings):
        """Inference on a single video using soft NMS - adapted from MMCTransformer."""
        segs_all = []
        scores_all = []
        cls_idxs_all = []

        # Get valid length for this sequence
        valid_length = int(seq_mask.sum().item())

        # Apply sigmoid normalization and mask
        pred_prob = torch.sigmoid(logit_f[:valid_length]).flatten()

        # Apply filtering to make NMS faster
        # 1. Keep seg with confidence score > a threshold
        keep_idxs = pred_prob > inference_settings["pre_nms_thresh"]
        pred_prob = pred_prob[keep_idxs]
        topk_idxs = keep_idxs.nonzero(as_tuple=True)[0]

        # 2. Keep top k top scoring boxes only
        num_topk = min(inference_settings["pre_nms_topk"], topk_idxs.size(0))
        pred_prob, idxs = pred_prob.sort(descending=True)
        pred_prob = pred_prob[:num_topk].clone()
        topk_idxs = topk_idxs[idxs[:num_topk]].clone()
        offsets = offset_f[:valid_length][topk_idxs]

        # 3. compute predicted segments
        seg_left = topk_idxs - offsets[:, 0]
        seg_right = topk_idxs + offsets[:, 1]
        pred_segs = torch.stack((seg_left, seg_right), -1)

        # 4. Keep seg with duration > a threshold
        seg_durations = seg_right - seg_left
        keep_idxs2 = seg_durations > inference_settings["duration_thresh"]
        keep_idxs3 = seg_durations < inference_settings.get("duration_thresh_max", 1000)
        keep_idxs2 = keep_idxs2 & keep_idxs3

        # *_all : N (filtered # of segments) x 2 / 1
        segs_all.append(pred_segs[keep_idxs2])
        scores_all.append(pred_prob[keep_idxs2])
        cls_idxs_all.append(topk_idxs[keep_idxs2])

        # cat along the seq_len
        segs_all, scores_all, cls_idxs_all = [
            torch.cat(x) for x in [segs_all, scores_all, cls_idxs_all]
        ]

        results = {"segments": segs_all, "scores": scores_all, "labels": cls_idxs_all}
        return results

    @torch.no_grad()
    def inference_(self, batch, inference_settings):
        """Inference with soft NMS - adapted from original paper implementation."""
        # Forward pass
        logit_a, logit_v, logit_f, offset_f = self(
            batch["features"]["audio"],
            batch["features"]["visual"],
            batch["features"]["caption"],
            mask=batch["sequence_masks"],
        )

        # Gather video meta information
        vid_idxs = batch.get(
            "video_ids",
            batch.get("video_id", [f"video_{i}" for i in range(logit_f.shape[0])]),
        )
        vid_lens = batch.get("duration", [0] * logit_f.shape[0])

        # Ensure vid_lens is a list/tensor we can iterate over
        if isinstance(vid_lens, (int, float)):
            vid_lens = [vid_lens] * logit_f.shape[0]
        elif torch.is_tensor(vid_lens) and vid_lens.dim() == 0:
            vid_lens = [vid_lens.item()] * logit_f.shape[0]
        elif torch.is_tensor(vid_lens):
            vid_lens = vid_lens.tolist()

        results = []
        seq_masks = batch["sequence_masks"]

        # Inference on each single video and gather the results
        for idx, (vidx, vlen) in enumerate(zip(vid_idxs, vid_lens)):
            # Gather per-video outputs
            cls_logits_per_vid = logit_f[idx]
            offsets_per_vid = offset_f[idx]
            masks_per_vid = seq_masks[idx]

            # Calculate max_seg_num based on video duration and max_seg_per_min
            mins = max(1, vlen // 60)  # Avoid division by zero, minimum 1 minute
            max_seg_num = mins * inference_settings["max_seg_per_min"]
            max_seg_num = int(np.ceil(max_seg_num))
            max_seg_num = max(1, max_seg_num)  # Ensure at least 1 segment allowed

            # Inference on a single video
            results_per_vid = self.inference_single_video(
                cls_logits_per_vid,
                offsets_per_vid,
                masks_per_vid,
                inference_settings,
            )

            # Apply soft NMS with proper max_seg_num constraint
            if len(results_per_vid["segments"]) > 0:
                results_per_vid_nms_idx = soft_nms_intervals_cpu(
                    results_per_vid["scores"],
                    results_per_vid["segments"],
                    sigma=inference_settings["nms_sigma"],
                    thresh=inference_settings["min_score"],
                    max_seg_num=max_seg_num,
                )
                results_per_vid["segments"] = results_per_vid["segments"][
                    results_per_vid_nms_idx
                ]
                results_per_vid["scores"] = results_per_vid["scores"][
                    results_per_vid_nms_idx
                ]
                results_per_vid["labels"] = results_per_vid["labels"][
                    results_per_vid_nms_idx
                ]

            # Pass through video meta info
            results_per_vid["video_id"] = vidx
            results_per_vid["duration"] = vlen
            results.append(results_per_vid)

        return results

    def configure_optimizers(self):
        """Configure optimizer with linear warmup and cosine decay scheduling."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            betas=(self.hparams.beta1, self.hparams.beta2),
        )

        def lr_lambda(current_step):
            # Linear warmup
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))

            # Cosine decay after warmup
            progress = float(current_step - self.warmup_steps) / float(
                max(1, self.total_steps - self.warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # Update every training step
                "frequency": 1,
            },
        }

    def _save_batch_debug_csv(self, batch, labels, offsets, seq_mask):
        """Save first batch data to CSV for debugging."""
        import pandas as pd
        import os

        debug_dir = "debug_regression"
        os.makedirs(debug_dir, exist_ok=True)

        # Process first sample in batch
        sample_idx = 0
        valid_len = int(seq_mask[sample_idx].sum().item())

        # Prepare data for CSV
        data = {
            "time_step": list(range(valid_len)),
            "label": labels[sample_idx, :valid_len].cpu().numpy(),
            "gt_left_offset": offsets[sample_idx, :valid_len, 0].cpu().numpy(),
            "gt_right_offset": offsets[sample_idx, :valid_len, 1].cpu().numpy(),
            "seq_mask": seq_mask[sample_idx, :valid_len].cpu().numpy(),
        }

        # Add video ID if available
        if "video_ids" in batch:
            video_id = batch["video_ids"][sample_idx]
            data["video_id"] = [video_id] * valid_len

        df = pd.DataFrame(data)
        csv_path = os.path.join(debug_dir, "batch_0_sample_0_data.csv")
        df.to_csv(csv_path, index=False)
        self.logger_instance.info(f"Saved batch debug data to {csv_path}")

        # Also save summary statistics
        stats = {
            "metric": [
                "total_frames",
                "positive_frames",
                "positive_ratio",
                "min_left_offset",
                "max_left_offset",
                "mean_left_offset",
                "min_right_offset",
                "max_right_offset",
                "mean_right_offset",
            ],
            "value": [
                valid_len,
                int((data["label"] > 0.5).sum()),
                float((data["label"] > 0.5).mean()),
                (
                    float(data["gt_left_offset"][data["label"] > 0.5].min())
                    if (data["label"] > 0.5).any()
                    else 0
                ),
                (
                    float(data["gt_left_offset"][data["label"] > 0.5].max())
                    if (data["label"] > 0.5).any()
                    else 0
                ),
                (
                    float(data["gt_left_offset"][data["label"] > 0.5].mean())
                    if (data["label"] > 0.5).any()
                    else 0
                ),
                (
                    float(data["gt_right_offset"][data["label"] > 0.5].min())
                    if (data["label"] > 0.5).any()
                    else 0
                ),
                (
                    float(data["gt_right_offset"][data["label"] > 0.5].max())
                    if (data["label"] > 0.5).any()
                    else 0
                ),
                (
                    float(data["gt_right_offset"][data["label"] > 0.5].mean())
                    if (data["label"] > 0.5).any()
                    else 0
                ),
            ],
        }
        stats_df = pd.DataFrame(stats)
        stats_path = os.path.join(debug_dir, "batch_0_sample_0_stats.csv")
        stats_df.to_csv(stats_path, index=False)
        self.logger_instance.info(f"Saved batch statistics to {stats_path}")

    def _save_loss_debug_csv(
        self,
        reg_loss_f_all,
        cls_mask,
        combined_mask,
        reg_loss_f,
        offset_f,
        offsets,
        epoch=0,
    ):
        """Save loss calculation details to CSV for debugging."""
        import pandas as pd
        import os

        debug_dir = "debug_regression"
        os.makedirs(debug_dir, exist_ok=True)

        # Process first sample
        sample_idx = 0
        batch_size, seq_len = reg_loss_f_all.shape

        # Get valid length from combined mask
        valid_len = int(
            combined_mask[sample_idx].sum().item()
            + (cls_mask[sample_idx] - combined_mask[sample_idx]).sum().item()
            + 100
        )  # Add some buffer to see non-positive positions too
        valid_len = min(valid_len, seq_len)

        # Prepare loss data - include epoch information
        loss_data = {
            "epoch": [epoch] * valid_len,
            "time_step": list(range(valid_len)),
            "cls_mask": cls_mask[sample_idx, :valid_len].detach().cpu().numpy(),
            "combined_mask": combined_mask[sample_idx, :valid_len]
            .detach()
            .cpu()
            .numpy(),
            "reg_loss_all": reg_loss_f_all[sample_idx, :valid_len]
            .detach()
            .cpu()
            .numpy(),
            "pred_left_offset": offset_f[sample_idx, :valid_len, 0]
            .detach()
            .cpu()
            .numpy(),
            "pred_right_offset": offset_f[sample_idx, :valid_len, 1]
            .detach()
            .cpu()
            .numpy(),
            "gt_left_offset": offsets[sample_idx, :valid_len, 0].detach().cpu().numpy(),
            "gt_right_offset": offsets[sample_idx, :valid_len, 1]
            .detach()
            .cpu()
            .numpy(),
        }

        # Add masked loss
        loss_data["masked_loss"] = (
            loss_data["reg_loss_all"] * loss_data["combined_mask"]
        )

        df = pd.DataFrame(loss_data)
        # Save with epoch in filename
        csv_path = os.path.join(
            debug_dir, f"epoch_{epoch:03d}_batch_0_sample_0_losses.csv"
        )
        df.to_csv(csv_path, index=False, float_format="%.6f")
        self.logger_instance.info(f"Saved loss debug data to {csv_path}")

        # Save aggregate metrics with epoch information
        metrics = {
            "epoch": [epoch] * 7,
            "metric": [
                "total_loss",
                "num_active_positions",
                "mean_loss_at_active",
                "mean_pred_left",
                "mean_pred_right",
                "std_pred_left",
                "std_pred_right",
            ],
            "value": [
                float(reg_loss_f.detach().item()),
                int(combined_mask.sum().item()),
                (
                    float(reg_loss_f_all[combined_mask.bool()].detach().mean().item())
                    if combined_mask.sum() > 0
                    else 0
                ),
                float(offset_f[sample_idx, :valid_len, 0].detach().mean().item()),
                float(offset_f[sample_idx, :valid_len, 1].detach().mean().item()),
                float(offset_f[sample_idx, :valid_len, 0].detach().std().item()),
                float(offset_f[sample_idx, :valid_len, 1].detach().std().item()),
            ],
        }
        metrics_df = pd.DataFrame(metrics)

        # Save with epoch in filename
        metrics_path = os.path.join(
            debug_dir, f"epoch_{epoch:03d}_batch_0_sample_0_metrics.csv"
        )
        metrics_df.to_csv(metrics_path, index=False, float_format="%.6f")
        self.logger_instance.info(f"Saved aggregate metrics to {metrics_path}")

        # Also append to a master CSV that tracks all epochs
        master_csv_path = os.path.join(debug_dir, "all_epochs_metrics.csv")
        if os.path.exists(master_csv_path):
            # Append to existing file
            existing_df = pd.read_csv(master_csv_path)
            updated_df = pd.concat([existing_df, metrics_df], ignore_index=True)
            updated_df.to_csv(master_csv_path, index=False, float_format="%.6f")
        else:
            # Create new file
            metrics_df.to_csv(master_csv_path, index=False, float_format="%.6f")
        self.logger_instance.info(f"Updated master metrics file: {master_csv_path}")

    def on_train_start(self):
        """Set warmup and total steps based on actual training configuration."""
        # Get the actual number of training steps
        if self.trainer.max_epochs:
            try:
                # Try to get the actual dataloader length
                if (
                    hasattr(self.trainer, "train_dataloader")
                    and self.trainer.train_dataloader is not None
                ):
                    train_dataloader = self.trainer.train_dataloader()
                    steps_per_epoch = len(train_dataloader)
                else:
                    # Fallback estimation
                    steps_per_epoch = 1000  # Conservative estimate
            except:
                # Fallback estimation if dataloader not available
                steps_per_epoch = 1000

            self.warmup_steps = self.hparams.warmup_epochs * steps_per_epoch
            self.total_steps = self.trainer.max_epochs * steps_per_epoch

            self.logger_instance.info(
                f"LR Schedule configured: warmup_steps={self.warmup_steps}, "
                f"total_steps={self.total_steps}, steps_per_epoch={steps_per_epoch}"
            )
        else:
            # Default values for safety
            self.warmup_steps = 1000
            self.total_steps = 10000
            self.logger_instance.info(
                f"Using default LR schedule: warmup_steps={self.warmup_steps}, total_steps={self.total_steps}"
            )

    def on_before_backward(self, loss):
        """Apply gradient clipping before backward pass."""
        # This is called automatically by PyTorch Lightning when gradient_clip_val is set
        pass

    def on_before_optimizer_step(self, optimizer):
        """Log gradient norms before optimization step."""
        # Compute gradient norm for monitoring
        total_norm = 0.0
        param_count = 0

        for param in self.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1

        if param_count > 0:
            total_norm = total_norm ** (1.0 / 2)

            # Log gradient norm
            self.log(
                "grad_norm", total_norm, on_step=True, on_epoch=False, prog_bar=True
            )

            # Log gradient norm for different component groups
            # Simple transformer components
            simple_norm = 0.0
            simple_count = 0
            for name, param in self.named_parameters():
                if param.grad is not None and "simple_" in name:
                    param_norm = param.grad.data.norm(2)
                    simple_norm += param_norm.item() ** 2
                    simple_count += 1

            if simple_count > 0:
                simple_norm = simple_norm ** (1.0 / 2)
                self.log("grad_norm_simple", simple_norm, on_step=True, on_epoch=False)

            # Cross-attention components
            cross_norm = 0.0
            cross_count = 0
            for name, param in self.named_parameters():
                if param.grad is not None and "cross_attn" in name:
                    param_norm = param.grad.data.norm(2)
                    cross_norm += param_norm.item() ** 2
                    cross_count += 1

            if cross_count > 0:
                cross_norm = cross_norm ** (1.0 / 2)
                self.log(
                    "grad_norm_cross_attn", cross_norm, on_step=True, on_epoch=False
                )


# ==================== Memory Management Callback ====================
class MemoryClearCallback(Callback):
    """Callback to clear memory at regular intervals."""

    def __init__(self, clear_every_n_epochs: int = 1):
        self.clear_every_n_epochs = clear_every_n_epochs
        self.logger = logging.getLogger(self.__class__.__name__)

    def on_train_epoch_end(self, trainer, pl_module):
        """Clear memory at end of epoch."""
        if trainer.current_epoch % self.clear_every_n_epochs == 0:
            self.logger.info(f"Clearing memory at epoch {trainer.current_epoch}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Log memory usage if available
            try:
                import psutil

                process = psutil.Process()
                mem_info = process.memory_info()
                self.logger.info(f"Memory usage: {mem_info.rss / 1024**3:.2f} GB")
            except ImportError:
                pass


# ==================== End-of-Epoch Visualization Callback ====================
class EndOfEpochVisualizationCallback(Callback):
    """Callback to create visualizations at end of each epoch for both train and val sets."""

    def __init__(
        self,
        train_dataloader,
        val_dataloader=None,
        save_dir: str = "visualizations",
    ):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.save_dir = save_dir
        self.logger = logging.getLogger(self.__class__.__name__)
        os.makedirs(save_dir, exist_ok=True)

    def on_train_epoch_end(self, trainer, pl_module):
        """Create visualizations at end of epoch."""
        epoch = trainer.current_epoch

        # For debugging, only visualize every 2 epochs to reduce load
        if epoch % 2 != 0:
            self.logger.info(
                f"Skipping visualization for epoch {epoch} (debugging mode - every 2 epochs)"
            )
            return

        self.logger.info(f"Creating end-of-epoch visualizations for epoch {epoch}")

        # Set model to eval mode
        pl_module.eval()
        device = next(pl_module.parameters()).device

        with torch.no_grad():
            # Process both train and validation datasets
            datasets_to_viz = []
            if self.train_dataloader:
                datasets_to_viz.append(("train", self.train_dataloader))
            if self.val_dataloader:
                datasets_to_viz.append(("val", self.val_dataloader))

            self.logger.info(
                f"Will process {len(datasets_to_viz)} datasets for visualization"
            )

            for dataset_idx, (dataset_name, dataloader) in enumerate(datasets_to_viz):
                self.logger.info(
                    f"Starting visualization for dataset {dataset_idx+1}/{len(datasets_to_viz)}: {dataset_name}"
                )
                log_memory_usage(self.logger, f"Before {dataset_name} visualization")

                try:
                    self.logger.info(
                        f"Processing first batch from {dataset_name} dataloader for visualization..."
                    )

                    # Process only the first batch
                    for batch_idx, batch in enumerate(dataloader):
                        if batch_idx > 0:  # Only process first batch
                            break

                        self.logger.debug(
                            f"Processing batch {batch_idx} for {dataset_name} set"
                        )

                        log_memory_usage(
                            self.logger, f"{dataset_name} batch {batch_idx} start"
                        )

                        try:
                            # Extract from dict batch format
                            audio = batch["features"]["audio"].to(device)
                            visual = batch["features"]["visual"].to(device)
                            caption = batch["features"]["caption"].to(device)
                            labels = batch["labels"].to(device)
                            offsets = batch["offsets"].to(
                                device
                            )  # Get ground truth offsets
                            seq_mask = batch["sequence_masks"].to(device)

                            log_memory_usage(
                                self.logger,
                                f"{dataset_name} batch {batch_idx} after data load",
                            )

                            # Get predictions - now includes regression output
                            logit_a, logit_v, logit_f, offset_f = pl_module(
                                audio, visual, caption, mask=seq_mask
                            )

                            log_memory_usage(
                                self.logger,
                                f"{dataset_name} batch {batch_idx} after inference",
                            )

                        except Exception as e:
                            self.logger.error(
                                f"Error during model inference for {dataset_name} batch {batch_idx}: {e}"
                            )
                            log_memory_usage(
                                self.logger,
                                f"{dataset_name} batch {batch_idx} error state",
                            )
                            continue

                        # Run inference on the full batch once
                        # Get predictions for all samples in batch at once
                        with torch.no_grad():
                            inference_predictions = pl_module.inference_(
                                batch, INFERENCE_SETTINGS
                            )

                        # Process each sequence in the batch for visualization
                        batch_size = logit_f.shape[0]
                        self.logger.debug(
                            f"Creating visualizations for {batch_size} sequences from batch {batch_idx}"
                        )

                        for seq_idx in range(batch_size):

                            try:
                                # Get sequence mask for this specific sequence
                                seq_mask_single = seq_mask[seq_idx]
                                valid_length = int(seq_mask_single.sum().item())

                                if valid_length == 0:
                                    self.logger.warning(
                                        f"Sequence {seq_idx} has zero valid length, skipping"
                                    )
                                    continue

                                # Extract predictions and labels for this sequence only
                                logit_f_seq = logit_f[seq_idx, :valid_length]
                                labels_seq = labels[seq_idx, :valid_length]
                                # Extract offsets
                                offset_f_seq = offset_f[seq_idx, :valid_length]
                                # Ground truth offsets
                                offsets_seq = offsets[seq_idx, :valid_length]

                                # Convert to numpy for visualization
                                pred_probs = torch.sigmoid(logit_f_seq).cpu().numpy()
                                labels_np = labels_seq.cpu().numpy()
                                pred_offsets_np = offset_f_seq.cpu().numpy()
                                gt_offsets_np = offsets_seq.cpu().numpy()

                                # Extract video ID for this sequence
                                video_id = batch["video_ids"][seq_idx]

                                self.logger.debug(
                                    f"Processing sequence {seq_idx+1}/{batch_size}: {video_id} ({valid_length} frames)"
                                )

                                # Create visualization immediately instead of accumulating data
                                if hasattr(trainer.logger, "experiment"):
                                    try:
                                        # Create plot with 4 subplots (similar to debug_visualizer)
                                        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
                                        time_points = np.arange(len(pred_probs))

                                        # Plot 1: Predictions vs Ground Truth
                                        axes[0].plot(
                                            time_points,
                                            pred_probs,
                                            "b-",
                                            label="Predicted Probability",
                                            alpha=0.7,
                                        )
                                        positive_idx = labels_np > 0.5
                                        if np.any(positive_idx):
                                            axes[0].scatter(
                                                time_points[positive_idx],
                                                np.ones(np.sum(positive_idx)),
                                                color="red",
                                                s=30,
                                                label="Ground Truth",
                                                zorder=5,
                                            )
                                        axes[0].set_ylabel("Probability")
                                        dataset_type = dataset_name.upper()
                                        axes[0].set_title(
                                            f"Epoch {epoch} - {dataset_type} Sequence {seq_idx+1} - Predictions vs Ground Truth"
                                        )
                                        axes[0].legend()
                                        axes[0].grid(True, alpha=0.3)
                                        axes[0].set_ylim(-0.1, 1.1)

                                        # Plot 2: Offset predictions
                                        axes[1].plot(
                                            time_points,
                                            pred_offsets_np[:, 0],
                                            "b-",
                                            label="Pred Left Offset",
                                            alpha=0.7,
                                        )
                                        axes[1].plot(
                                            time_points,
                                            pred_offsets_np[:, 1],
                                            "b--",
                                            label="Pred Right Offset",
                                            alpha=0.7,
                                        )
                                        # Plot GT offsets only at positive positions
                                        if np.any(positive_idx):
                                            axes[1].scatter(
                                                time_points[positive_idx],
                                                gt_offsets_np[positive_idx, 0],
                                                color="red",
                                                s=30,
                                                label="GT Left Offset",
                                                marker="o",
                                            )
                                            axes[1].scatter(
                                                time_points[positive_idx],
                                                gt_offsets_np[positive_idx, 1],
                                                color="darkred",
                                                s=30,
                                                label="GT Right Offset",
                                                marker="s",
                                            )
                                        axes[1].set_ylabel("Offset Value")
                                        axes[1].set_xlabel("Time Steps")
                                        axes[1].set_title("Offset Predictions")
                                        axes[1].legend()
                                        axes[1].grid(True, alpha=0.3)

                                        # Plot 3: Segments visualization from offsets
                                        ax3 = axes[2]

                                        # Use the pre-computed inference results for this sequence
                                        # (inference was already run on the full batch above)

                                        # Draw predicted segments from the batch inference results
                                        if (
                                            inference_predictions
                                            and seq_idx < len(inference_predictions)
                                            and "segments"
                                            in inference_predictions[seq_idx]
                                        ):
                                            pred_segments = (
                                                inference_predictions[seq_idx][
                                                    "segments"
                                                ]
                                                .cpu()
                                                .numpy()
                                            )
                                            for seg in pred_segments:
                                                start, end = seg[0], seg[1]
                                                ax3.add_patch(
                                                    patches.Rectangle(
                                                        (start, 0.6),
                                                        end - start,
                                                        0.3,
                                                        facecolor="blue",
                                                        alpha=0.5,
                                                    )
                                                )

                                        # Draw GT segments from offsets
                                        for t_idx in np.where(positive_idx)[0]:
                                            left_offset = gt_offsets_np[t_idx, 0]
                                            right_offset = gt_offsets_np[t_idx, 1]
                                            start = max(0, t_idx - left_offset)
                                            end = min(
                                                len(pred_probs), t_idx + right_offset
                                            )
                                            ax3.add_patch(
                                                patches.Rectangle(
                                                    (start, 0.1),
                                                    end - start,
                                                    0.3,
                                                    facecolor="red",
                                                    alpha=0.5,
                                                )
                                            )

                                        ax3.set_xlim(0, len(pred_probs))
                                        ax3.set_ylim(0, 1)
                                        ax3.set_xlabel("Time Steps")
                                        ax3.set_title(
                                            "Segment Visualization with Soft-NMS Inference (Blue: Predicted, Red: Ground Truth)"
                                        )
                                        ax3.grid(True, alpha=0.3, axis="x")

                                        # Plot 4: Prediction confidence
                                        confidence = np.abs(pred_probs - 0.5) * 2
                                        axes[3].plot(
                                            time_points,
                                            confidence,
                                            "g-",
                                            label="Confidence",
                                            alpha=0.7,
                                        )
                                        axes[3].set_ylabel("Confidence")
                                        axes[3].set_xlabel("Time Steps")
                                        axes[3].set_title("Prediction Confidence")
                                        axes[3].legend()
                                        axes[3].grid(True, alpha=0.3)
                                        axes[3].set_ylim(0, 1)

                                        plt.tight_layout()

                                        # Save to file
                                        viz_path = os.path.join(
                                            self.save_dir,
                                            f"epoch_{epoch}_{dataset_name}_{video_id}.png",
                                        )

                                        self.logger.debug(
                                            f"Saving visualization to {viz_path}"
                                        )
                                        plt.savefig(
                                            viz_path, dpi=120, bbox_inches="tight"
                                        )

                                        # Create caption
                                        caption = f"Epoch {epoch}, {dataset_type} set, Video {video_id}"

                                        # Log to wandb
                                        try:
                                            trainer.logger.experiment.log(
                                                {
                                                    f"visualizations/{dataset_name}/{video_id}": wandb.Image(
                                                        viz_path, caption=caption
                                                    ),
                                                }
                                            )
                                        except Exception as e:
                                            self.logger.error(
                                                f"Error logging wandb image for {video_id}: {e}"
                                            )

                                        # Close plot and cleanup
                                        plt.close(fig)
                                        del (
                                            pred_probs,
                                            labels_np,
                                            pred_offsets_np,
                                            gt_offsets_np,
                                            fig,
                                            axes,
                                        )

                                        self.logger.debug(
                                            f"Completed immediate visualization for {video_id}"
                                        )

                                    except Exception as e:
                                        self.logger.error(
                                            f"Error creating immediate visualization for {video_id}: {e}"
                                        )
                                        log_memory_usage(
                                            self.logger,
                                            f"After viz seq {seq_idx} error",
                                        )

                                # Visualization completed for this sequence

                                # Explicit cleanup of large tensors
                                del logit_f_seq, labels_seq

                            except Exception as e:
                                self.logger.error(
                                    f"Error processing sequence {seq_idx} in {dataset_name}: {e}"
                                )
                                continue

                        # Cleanup after each batch
                        try:
                            del (
                                audio,
                                visual,
                                caption,
                                labels,
                                offsets,
                                seq_mask,
                                logit_f,
                                offset_f,
                            )
                        except Exception as cleanup_e:
                            self.logger.warning(
                                f"Error during batch cleanup: {cleanup_e}"
                            )
                        log_memory_usage(
                            self.logger, f"{dataset_name} batch {batch_idx} cleanup"
                        )

                    self.logger.info(
                        f"Completed visualization of batch from {dataset_name} set"
                    )

                except Exception as e:
                    self.logger.error(f"Error processing {dataset_name} dataset: {e}")
                    log_memory_usage(self.logger, f"After {dataset_name} dataset error")
                    continue

                finally:
                    # Cleanup between datasets
                    try:
                        plt.close("all")
                        import gc

                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        log_memory_usage(
                            self.logger, f"After {dataset_name} dataset cleanup"
                        )
                        self.logger.debug(
                            f"Cleaned up resources after {dataset_name} visualization"
                        )
                    except Exception as cleanup_error:
                        self.logger.warning(
                            f"Error during cleanup after {dataset_name}: {cleanup_error}"
                        )

                self.logger.info(f"Completed visualization for {dataset_name} set")

        # Final cleanup and switch back to training mode
        try:
            plt.close("all")
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            self.logger.warning(f"Error during final cleanup: {e}")

        pl_module.train()
        self.logger.info(f"Visualization callback completed for epoch {epoch}")


# ==================== Main Training Function ====================
def main(args):
    """Main training function with comprehensive setup."""
    # Enable Tensor Cores for faster training on compatible GPUs
    torch.set_float32_matmul_precision("medium")

    # Set seeds for reproducibility if deterministic mode is enabled
    if args.deterministic:
        seed = 42  # Fixed seed for reproducibility
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        # Additional deterministic settings
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"train_repurpose_{timestamp}.log"
    logger = setup_logging(args.log_level, log_file)

    logger.info("Starting RepurposeModel training")
    logger.info(f"Arguments: {vars(args)}")

    if args.deterministic:
        logger.info("✓ Deterministic mode enabled - training will be reproducible")
        logger.info(f"  Random seed: 42")
        logger.info("  Note: This may reduce training speed by 10-30%")

    # Initialize wandb if requested
    wandb_logger = None
    if args.use_wandb:
        try:
            wandb_logger = WandbLogger(
                project=args.wandb_project,
                name=f"repurpose_{timestamp}",
                config=vars(args),
                log_model=False,  # Don't auto-log model checkpoints to save space
                save_dir="./wandb_logs",
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
            "audio": args.audio_dir,
            "visual": args.visual_dir,
            "caption": args.caption_dir,
        },
        annotation_file=args.train_annotation,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        min_modalities=3,
        max_seq_length=(
            args.max_seq_len if args.max_seq_len and args.max_seq_len > 0 else None
        ),
    )

    val_dataloader = None
    if args.val_annotation:
        val_dataloader = create_sequence_dataloader(
            feature_dirs={
                "audio": args.audio_dir,
                "visual": args.visual_dir,
                "caption": args.caption_dir,
            },
            annotation_file=args.val_annotation,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,  # No shuffling for validation
            min_modalities=3,
            max_seq_length=(
                args.max_seq_len if args.max_seq_len and args.max_seq_len > 0 else None
            ),
        )

    # Get dimensions from a sample batch
    logger.info("Determining feature dimensions...")
    sample_batch = next(iter(train_dataloader))
    dim_audio = sample_batch["features"]["audio"].shape[-1]
    dim_visual = sample_batch["features"]["visual"].shape[-1]
    dim_caption = sample_batch["features"]["caption"].shape[-1]

    logger.info(
        f"Feature dimensions - Audio: {dim_audio}, Visual: {dim_visual}, Caption: {dim_caption}"
    )

    # Create model
    model = RepurposeModel(
        dim_audio=dim_audio,
        dim_visual=dim_visual,
        dim_caption=dim_caption,
        d_model=args.d_model,
        n_head=args.n_head,
        n_self_attn_layers=args.n_self_attn_layers,
        n_cross_attn_layers=args.n_cross_attn_layers,
        n_fusion_layers=args.n_fusion_layers,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        warmup_epochs=args.warmup_epochs,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        lambda3=args.lambda3,
        lambda4=args.lambda4,
        log_interval=args.log_interval,
    )

    logger.info(
        f"Created model with {sum(p.numel() for p in model.parameters())} parameters"
    )
    logger.info(
        f"Architecture: Self-Attn={args.n_self_attn_layers}, Cross-Attn={args.n_cross_attn_layers}, Fusion={args.n_fusion_layers}"
    )

    # Setup callbacks
    callbacks = []

    # Memory management
    callbacks.append(MemoryClearCallback(clear_every_n_epochs=1))

    # End-of-epoch visualization for both train and val sets (reduced for debugging)
    viz_callback = EndOfEpochVisualizationCallback(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        save_dir=os.path.join(args.checkpoint_dir, "epoch_visualizations"),
    )
    callbacks.append(viz_callback)

    # Model checkpointing (only if enabled)
    if args.enable_checkpointing:
        checkpoint_callback = ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename="repurpose-{epoch:02d}-{train/loss_total:.4f}",
            monitor="train/loss_total",
            mode="min",
            save_top_k=3,
            save_last=True,
        )
        callbacks.append(checkpoint_callback)
        logger.info(
            "Checkpointing enabled - models will be saved to: " + args.checkpoint_dir
        )
    else:
        logger.info("Checkpointing disabled - models will NOT be saved")

    # Early stopping
    if val_dataloader and args.early_stopping_patience > 0:
        early_stopping = EarlyStopping(
            monitor="val/loss",
            patience=args.early_stopping_patience,
            mode="min",
            verbose=True,
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
        num_sanity_val_steps=0,  # Disable sanity checking to avoid early exit
    )

    # Start training
    logger.info("Starting training...")
    start_time = time.time()

    try:
        # Run initial validation to establish baseline (only for fresh training)
        if not (
            args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint)
        ):
            if val_dataloader:
                logger.info(
                    "Running initial validation before training to establish baseline..."
                )
                try:
                    trainer.validate(model, val_dataloader)
                    logger.info("Initial validation completed")
                except Exception as val_error:
                    logger.warning(f"Initial validation failed: {val_error}")
                    logger.warning("Proceeding with training anyway...")
                    logger.warning(
                        "This might be due to visualization callbacks expecting training context"
                    )

        if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
            logger.info(
                f"Resuming training from checkpoint: {args.resume_from_checkpoint}"
            )
            trainer.fit(
                model,
                train_dataloader,
                val_dataloader,
                ckpt_path=args.resume_from_checkpoint,
            )
        else:
            if args.resume_from_checkpoint:
                logger.warning(
                    f"Checkpoint file not found: {args.resume_from_checkpoint}, starting from scratch"
                )
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
    # Final cleanup
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Training script completed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train RepurposeModel with logging and wandb"
    )

    # Data arguments
    parser.add_argument(
        "--audio_dir", type=str, required=True, help="Path to audio features directory"
    )
    parser.add_argument(
        "--visual_dir",
        type=str,
        required=True,
        help="Path to visual features directory",
    )
    parser.add_argument(
        "--caption_dir",
        type=str,
        required=True,
        help="Path to caption features directory",
    )
    parser.add_argument(
        "--train_annotation",
        type=str,
        required=True,
        help="Path to training annotation JSON",
    )
    parser.add_argument(
        "--val_annotation", type=str, help="Path to validation annotation JSON"
    )

    # Model arguments
    parser.add_argument("--d_model", type=int, default=128, help="Model dimension")
    parser.add_argument(
        "--n_head", type=int, default=4, help="Number of attention heads"
    )
    parser.add_argument(
        "--n_self_attn_layers",
        type=int,
        default=2,
        help="Number of self-attention layers per modality",
    )
    parser.add_argument(
        "--n_cross_attn_layers",
        type=int,
        default=2,
        help="Number of cross-attention layers for A-C and V-C",
    )
    parser.add_argument(
        "--n_fusion_layers",
        type=int,
        default=2,
        help="Number of Audio-Visual fusion layers",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=None,
        help="Maximum sequence length (None for full videos)",
    )

    # Training arguments
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay for optimizer regularization",
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.9,
        help="Beta1 for AdamW optimizer",
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=0.98,
        help="Beta2 for AdamW optimizer (default 0.98, lower than standard 0.999 for faster adaptation)",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=1,
        help="Number of warmup epochs for learning rate schedule",
    )
    parser.add_argument(
        "--gradient_clip",
        type=float,
        default=None,
        help="Gradient clipping value (None to disable)",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )

    # Loss weights
    parser.add_argument(
        "--lambda1", type=float, default=0.1, help="Weight for uni-modal loss"
    )
    parser.add_argument(
        "--lambda2", type=float, default=0.3, help="Weight for multi-modal loss"
    )
    parser.add_argument(
        "--lambda3", type=float, default=0.1, help="Weight for KL divergence loss"
    )
    parser.add_argument(
        "--lambda4", type=float, default=0.7, help="Weight for regression loss"
    )

    # Hardware arguments
    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        help="Accelerator type (cpu, gpu, auto)",
    )
    parser.add_argument("--devices", type=int, default=1, help="Number of devices")
    parser.add_argument(
        "--precision", type=str, default="32", help="Training precision (16, 32, bf16)"
    )
    parser.add_argument(
        "--num_workers", type=int, default=0, help="Number of data loader workers"
    )

    # Logging arguments
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")
    parser.add_argument(
        "--log_interval", type=int, default=10, help="Log every N steps"
    )
    parser.add_argument(
        "--use_wandb", action="store_true", help="Use Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb_project", type=str, default="repurpose", help="W&B project name"
    )

    # Checkpointing
    parser.add_argument(
        "--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory"
    )
    parser.add_argument(
        "--enable_checkpointing",
        action="store_true",
        help="Enable model checkpointing (default: disabled)",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file for resuming training",
    )
    parser.add_argument(
        "--early_stopping_patience", type=int, default=5, help="Early stopping patience"
    )

    # Validation
    parser.add_argument(
        "--val_check_interval",
        type=float,
        default=1.0,
        help="Validation check interval",
    )
    parser.add_argument(
        "--limit_train_batches", type=float, default=1.0, help="Limit training batches"
    )
    parser.add_argument(
        "--limit_val_batches", type=float, default=1.0, help="Limit validation batches"
    )

    # Misc
    parser.add_argument(
        "--deterministic", action="store_true", help="Use deterministic training"
    )

    args = parser.parse_args()

    # Example command line usage:
    # python train_repurpose.py \
    #   --audio_dir /path/to/audio \
    #   --visual_dir /path/to/visual \
    #   --caption_dir /path/to/caption \
    #   --train_annotation /path/to/train.json \
    #   --val_annotation /path/to/val.json \
    #   --use_wandb \
    #   --epochs 20

    main(args)
