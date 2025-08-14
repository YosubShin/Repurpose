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
import argparse
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

import numpy as np
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
        simple_d_model = 32
        simple_nhead = 4
        simple_num_layers = 2
        self.simple_input_proj = nn.Linear(dim_visual, simple_d_model)
        self.simple_pos_embed = nn.Parameter(
            torch.randn(1, 2000, simple_d_model) * 0.01
        )
        self.simple_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=simple_d_model,
                nhead=simple_nhead,
                dim_feedforward=64,
                batch_first=True,
            ),
            num_layers=simple_num_layers,
        )
        self.simple_output = nn.Sequential(
            # Ensure positive offsets
            nn.Linear(simple_d_model, 2),
            nn.Softplus(),
        )

    def forward(
        self,
        audio: torch.Tensor,
        visual: torch.Tensor,
        caption: torch.Tensor,
        mask: torch.Tensor,
    ):
        """SIMPLE OFFSET TRANSFORMER FOR TESTING - uses only visual features."""
        batch_size, seq_len = visual.shape[:2]

        # Forward pass through simple transformer
        x = self.simple_input_proj(visual)
        x = x + self.simple_pos_embed[:, :seq_len, :]

        # Convert mask to attention mask (True = ignore)
        attn_mask = ~mask.bool()
        x = self.simple_encoder(x, src_key_padding_mask=attn_mask)

        # Get offset predictions
        offset_f = self.simple_output(x)  # [B, T, 2]

        # Create dummy classification outputs for compatibility
        logit_a = torch.zeros(batch_size, seq_len, device=visual.device)
        logit_v = torch.zeros(batch_size, seq_len, device=visual.device)
        logit_f = torch.zeros(batch_size, seq_len, device=visual.device)

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

        # Multi-layer Cross-attention: Visual-Caption
        v_enhanced = v
        for layer in self.cross_attn_vc_layers:
            v_enhanced = layer(v_enhanced, c, mask=mask)

        # Multi-layer Audio-Visual fusion cross-attention
        # Following the original architecture with separate cross-attention layers
        vis_feats = v_enhanced
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
        loss_a_all = sigmoid_focal_loss(
            logit_a,
            labels,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            reduction="none",
        )
        loss_v_all = sigmoid_focal_loss(
            logit_v,
            labels,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            reduction="none",
        )

        # Apply sequence mask and sum (following original paper pattern)
        loss_mul = (loss_mul_all * seq_mask).sum()
        loss_a = (loss_a_all * seq_mask).sum()
        loss_v = (loss_v_all * seq_mask).sum()
        loss_uni = loss_a + loss_v

        # 2. Regression loss - following original paper implementation exactly
        # Compute regression loss for all positions
        reg_loss_f_all = ctr_diou_loss_1d(offset_f, offsets, reduction="none")  # [B, T]

        # Create combined mask: sequence mask AND positive label mask
        cls_mask = (labels > 0.5).float()
        combined_mask = seq_mask * cls_mask

        # Apply combined mask and sum (following original pattern)
        reg_loss_f = (reg_loss_f_all * combined_mask).sum() / combined_mask.sum()

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
        prob_a = torch.sigmoid(logit_a[valid_positions]).detach()
        prob_v = torch.sigmoid(logit_v[valid_positions]).detach()
        prob_f = torch.sigmoid(logit_f[valid_positions])
        loss_kl = kl_div_bernoulli(prob_v, prob_f) + kl_div_bernoulli(prob_a, prob_f)

        # Total loss - includes classification and regression components
        total_loss = (
            self.lambda1 * loss_uni
            + self.lambda2 * loss_mul
            + self.lambda3 * loss_kl
            + self.lambda4 * reg_loss_f
        )

        # Compute metrics
        with torch.no_grad():
            labels_valid = labels[valid_positions]
            pred_binary = (prob_f > 0.5).float()
            accuracy = (pred_binary == labels_valid).float().mean()

            # Positive predictions
            n_positive_preds = pred_binary.sum().item()
            n_positive_labels = labels_valid.sum().item()
            n_total = len(labels_valid)

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

        # Classification loss - following original paper implementation exactly
        val_loss_cls_all = sigmoid_focal_loss(
            logit_f,
            labels,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            reduction="none",
        )
        val_loss_cls = (val_loss_cls_all * seq_mask).sum()

        # Regression loss - following original paper implementation exactly
        val_loss_reg_all = ctr_diou_loss_1d(
            offset_f, offsets, reduction="none"
        )  # [B, T]
        cls_mask = (labels > 0.5).float()
        combined_mask = seq_mask * cls_mask
        val_loss_reg = (val_loss_reg_all * combined_mask).sum()

        # Total validation loss
        val_loss = val_loss_cls + self.lambda4 * val_loss_reg

        # Metrics - compute for valid positions only
        valid_positions = seq_mask.bool()
        logit_f_valid = logit_f[valid_positions]
        labels_valid = labels[valid_positions]
        prob_f = torch.sigmoid(logit_f_valid)
        pred_binary = (prob_f > 0.5).float()
        accuracy = (pred_binary == labels_valid).float().mean()

        val_metrics = {
            "val/loss": val_loss,
            "val/loss_cls": val_loss_cls,
            "val/loss_reg": val_loss_reg,
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
        inference_settings = {
            "pre_nms_thresh": 0.001,
            "pre_nms_topk": 2000,
            "duration_thresh": 0.1,
            "duration_thresh_max": 1000,
            "nms_sigma": 0.75,
            "min_score": 0.001,
        }

        # Get predictions using inference method
        predictions = self.inference_(batch, inference_settings)

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
        """Inference with soft NMS - adapted from MMCTransformer."""
        # Forward pass
        logit_a, logit_v, logit_f, offset_f = self(
            batch["features"]["audio"],
            batch["features"]["visual"],
            batch["features"]["caption"],
            mask=batch["sequence_masks"],
        )

        results = []
        seq_masks = batch["sequence_masks"]
        video_ids = batch.get(
            "video_ids", [f"video_{i}" for i in range(logit_f.shape[0])]
        )

        # Process each video in the batch
        for idx in range(logit_f.shape[0]):
            # Get per-video outputs
            cls_logits_per_vid = logit_f[idx]
            offsets_per_vid = offset_f[idx]
            seq_mask_per_vid = seq_masks[idx]

            # Calculate max segments based on video length (simplified)
            valid_length = int(seq_mask_per_vid.sum().item())
            max_seg_num = max(1, valid_length // 10)  # Simplified heuristic

            # Inference on single video
            results_per_vid = self.inference_single_video(
                cls_logits_per_vid,
                offsets_per_vid,
                seq_mask_per_vid,
                inference_settings,
            )

            # Apply soft NMS if we have segments
            if len(results_per_vid["segments"]) > 0:
                results_per_vid_nms_idx = soft_nms_intervals_cpu(
                    results_per_vid["scores"],
                    results_per_vid["segments"],
                    sigma=inference_settings.get("nms_sigma", 0.5),
                    thresh=inference_settings.get("min_score", 0.001),
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

            # Add video metadata
            results_per_vid["video_id"] = video_ids[idx]
            results.append(results_per_vid)

        return results

    def configure_optimizers(self):
        """Configure optimizer with warmup and cosine decay scheduling."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            # weight_decay=self.hparams.weight_decay,
        )

        # We need to create a custom scheduler that combines warmup and cosine decay
        # The actual warmup_steps will be set in on_train_start when we know the dataloader size
        self.warmup_steps = 100  # Default, will be updated
        self.total_steps = 1000  # Default, will be updated

        # Create a combined scheduler
        def lr_lambda(current_step):
            # Warmup phase
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            # Cosine decay phase
            else:
                progress = float(current_step - self.warmup_steps) / float(
                    max(1, self.total_steps - self.warmup_steps)
                )
                return 0.5 * (1.0 + np.cos(np.pi * progress))

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # Step every batch, not epoch
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
            steps_per_epoch = len(self.trainer.train_dataloader)
            self.warmup_steps = self.hparams.warmup_epochs * steps_per_epoch
            self.total_steps = self.trainer.max_epochs * steps_per_epoch

            self.logger_instance.info(
                f"Learning rate schedule configured: warmup_steps={self.warmup_steps}, "
                f"total_steps={self.total_steps}, steps_per_epoch={steps_per_epoch}"
            )

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
        num_samples: int = 10,
        save_dir: str = "visualizations",
    ):
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
                    sample_count = 0

                    self.logger.info(
                        f"Processing samples from {dataset_name} dataloader with immediate visualization..."
                    )

                    for batch_idx, batch in enumerate(dataloader):
                        if sample_count >= self.num_samples:
                            self.logger.info(
                                f"Reached target of {self.num_samples} samples, stopping batch processing"
                            )
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

                        # Process each sequence in the batch individually
                        batch_size = logit_f.shape[0]
                        self.logger.debug(
                            f"Processing {batch_size} sequences in batch {batch_idx}"
                        )

                        for seq_idx in range(batch_size):
                            if sample_count >= self.num_samples:
                                break

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
                                    f"Processing sample {sample_count}: {video_id} ({valid_length} frames)"
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
                                            f"Epoch {epoch} - {dataset_type} Sample {sample_count} - Predictions vs Ground Truth"
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
                                        # Draw predicted segments at high confidence points
                                        high_conf_idx = pred_probs > 0.5
                                        for t_idx in np.where(high_conf_idx)[0]:
                                            left_offset = pred_offsets_np[t_idx, 0]
                                            right_offset = pred_offsets_np[t_idx, 1]
                                            start = max(0, t_idx - left_offset)
                                            end = min(
                                                len(pred_probs), t_idx + right_offset
                                            )
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
                                            "Segment Visualization from Offsets (Blue: Predicted, Red: Ground Truth)"
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
                                            f"After viz {sample_count} error",
                                        )

                                sample_count += 1

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
                        f"Completed immediate processing of {sample_count} samples from {dataset_name} set"
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


# ==================== Visualization Functions ====================
def visualize_predictions(
    model, dataloader, save_dir: str, num_samples: int = 5, device="cpu"
):
    """Visualize model predictions vs ground truth."""
    logger = logging.getLogger("Visualizer")
    logger.info(f"Creating visualizations for {num_samples} samples")

    model.eval()

    # Handle device conversion properly
    if isinstance(device, str):
        if device == "auto":
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
                audio = batch["features"]["audio"].to(device)
                visual = batch["features"]["visual"].to(device)
                caption = batch["features"]["caption"].to(device)
                labels = batch["labels"].to(device)
                offsets = batch["offsets"].to(device)  # Get ground truth offsets
                seq_mask = batch["sequence_masks"]

                logger.info(
                    f"Batch shapes - audio: {audio.shape}, visual: {visual.shape}, caption: {caption.shape}"
                )

                _, _, logit_f, offset_f = model(audio, visual, caption, mask=seq_mask)
                logger.info(
                    f"Model inference completed, logit_f shape: {logit_f.shape}"
                )

            except Exception as batch_error:
                logger.error(
                    f"Error processing batch {batch_idx} in post-training visualization: {batch_error}"
                )
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
                pred_probs = (
                    torch.sigmoid(logit_f[seq_idx, :valid_length]).cpu().numpy()
                )
                labels_np = labels[seq_idx, :valid_length].cpu().numpy()
                pred_offsets_np = offset_f[seq_idx, :valid_length].cpu().numpy()
                gt_offsets_np = offsets[seq_idx, :valid_length].cpu().numpy()

                # Create visualization with offset plots
                fig, axes = plt.subplots(4, 1, figsize=(15, 12))
                seq_len = len(pred_probs)
                time_points = np.arange(seq_len)

                # Plot 1: Classification scores
                ax1 = axes[0]
                ax1.plot(
                    time_points, pred_probs, "b-", label="Predicted Prob", alpha=0.7
                )
                positive_idx = labels_np > 0.5
                if np.any(positive_idx):
                    ax1.scatter(
                        time_points[positive_idx],
                        np.ones(np.sum(positive_idx)),
                        color="red",
                        s=50,
                        label="GT Positive",
                        zorder=5,
                    )
                ax1.set_ylabel("Classification Score")
                ax1.set_title(f"Sample {sample_count} - Classification Predictions")
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                ax1.set_ylim(-0.1, 1.1)

                # Plot 2: Offset predictions
                ax2 = axes[1]
                ax2.plot(
                    time_points,
                    pred_offsets_np[:, 0],
                    "b-",
                    label="Pred Left Offset",
                    alpha=0.7,
                )
                ax2.plot(
                    time_points,
                    pred_offsets_np[:, 1],
                    "b--",
                    label="Pred Right Offset",
                    alpha=0.7,
                )
                # Plot GT offsets only at positive positions
                if np.any(positive_idx):
                    ax2.scatter(
                        time_points[positive_idx],
                        gt_offsets_np[positive_idx, 0],
                        color="red",
                        s=30,
                        label="GT Left Offset",
                        marker="o",
                    )
                    ax2.scatter(
                        time_points[positive_idx],
                        gt_offsets_np[positive_idx, 1],
                        color="darkred",
                        s=30,
                        label="GT Right Offset",
                        marker="s",
                    )
                ax2.set_ylabel("Offset Value")
                ax2.set_xlabel("Time Steps")
                ax2.set_title("Offset Predictions")
                ax2.legend()
                ax2.grid(True, alpha=0.3)

                # Plot 3: Segments visualization from offsets
                ax3 = axes[2]

                # Draw predicted segments at high confidence points
                high_conf_idx = pred_probs > 0.5
                for t_idx in np.where(high_conf_idx)[0]:
                    left_offset = pred_offsets_np[t_idx, 0]
                    right_offset = pred_offsets_np[t_idx, 1]
                    start = max(0, t_idx - left_offset)
                    end = min(seq_len, t_idx + right_offset)
                    ax3.add_patch(
                        patches.Rectangle(
                            (start, 0.6), end - start, 0.3, facecolor="blue", alpha=0.5
                        )
                    )

                # Draw GT segments from offsets
                for t_idx in np.where(positive_idx)[0]:
                    left_offset = gt_offsets_np[t_idx, 0]
                    right_offset = gt_offsets_np[t_idx, 1]
                    start = max(0, t_idx - left_offset)
                    end = min(seq_len, t_idx + right_offset)
                    ax3.add_patch(
                        patches.Rectangle(
                            (start, 0.1), end - start, 0.3, facecolor="red", alpha=0.5
                        )
                    )

                ax3.set_xlim(0, seq_len)
                ax3.set_ylim(0, 1)
                ax3.set_xlabel("Time Steps")
                ax3.set_title(
                    "Segment Visualization from Offsets (Blue: Predicted, Red: Ground Truth)"
                )
                ax3.grid(True, alpha=0.3, axis="x")

                # Plot 4: Confidence over time
                ax4 = axes[3]
                confidence = np.abs(pred_probs - 0.5) * 2
                ax4.plot(time_points, confidence, "g-", label="Confidence", alpha=0.7)
                ax4.set_ylabel("Confidence")
                ax4.set_xlabel("Time Steps")
                ax4.set_title("Prediction Confidence")
                ax4.legend()
                ax4.grid(True, alpha=0.3)
                ax4.set_ylim(0, 1)

                plt.tight_layout()

                # Save figure
                path = os.path.join(
                    save_dir, f"visualization_sample_{sample_count}.png"
                )
                plt.savefig(path, dpi=150, bbox_inches="tight")
                saved_paths.append(path)
                logger.info(f"Saved visualization to {path}")

                plt.close()

                sample_count += 1

    return saved_paths


# ==================== Main Training Function ====================
def main(args):
    """Main training function with comprehensive setup."""
    # Enable Tensor Cores for faster training on compatible GPUs
    torch.set_float32_matmul_precision("medium")

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
        num_samples=args.num_viz_samples,
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
    if args.create_visualizations:
        if args.resume_from_checkpoint and os.path.exists(
            args.resume_from_checkpoint or ""
        ):
            logger.info(
                "Skipping post-training visualizations when resuming from checkpoint (known dataloader issue)"
            )
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
                logger.info(
                    f"Using {'validation' if val_dataloader else 'training'} dataloader for visualization"
                )

                visualize_predictions(
                    model,
                    viz_dataloader,
                    viz_dir,
                    num_samples=args.num_viz_samples,
                    device=actual_device,
                )

                log_memory_usage(logger, "After post-training visualization")
                logger.info("Post-training visualizations completed successfully")

            except Exception as viz_error:
                logger.error(f"Error during post-training visualization: {viz_error}")
                logger.error("Full visualization traceback:", exc_info=True)
                log_memory_usage(logger, "After post-training visualization error")
                logger.info(
                    "Training completed successfully despite visualization error"
                )

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
        "--warmup_epochs",
        type=int,
        default=1,
        help="Number of warmup epochs for learning rate schedule",
    )
    parser.add_argument(
        "--gradient_clip", type=float, default=1.0, help="Gradient clipping value"
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

    # Visualization
    parser.add_argument(
        "--create_visualizations",
        action="store_true",
        help="Create prediction visualizations",
    )
    parser.add_argument(
        "--num_viz_samples", type=int, default=5, help="Number of samples to visualize"
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
    #   --create_visualizations \
    #   --epochs 20

    main(args)
