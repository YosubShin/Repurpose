import torch
import torch.nn as nn
import math
from .losses import sigmoid_focal_loss
from .softnms import soft_nms_intervals_cpu
import numpy as np


class PositionalEncoding(nn.Module):
    """Minimal positional encoding implementation"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        return x + self.pe[:, :x.size(1)]


class MMCTransformer(nn.Module):
    def __init__(self, vis_dim, aud_dim, text_dim, d_model, self_num_layers, text_num_layers, cross_num_layers, num_heads, d_ff=2048):
        super(MMCTransformer, self).__init__()
        # Concatenated feature dimension
        concat_dim = vis_dim + aud_dim + text_dim
        
        # Linear layer to project concatenated features to d_model
        self.input_projection = nn.Linear(concat_dim, d_model)
        
        # Add input layer normalization to help with gradient flow
        self.input_norm = nn.LayerNorm(d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model)
        
        # Use PyTorch's built-in TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=0.1,  # Add dropout to help with regularization
            activation='relu',
            batch_first=True,  # Important: use batch_first=True for consistency
            norm_first=True  # Pre-LN architecture, helps with gradient flow
        )
        
        self.multimodal_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self_num_layers,
            enable_nested_tensor=False  # Disable for better compatibility
        )
        
        # Additional normalization after encoder
        self.encoder_norm = nn.LayerNorm(d_model)
        
        hidden_dim = 256
        
        # Feature projection with additional normalization
        self.feature_map = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),  # Extra norm layer
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Classification head with layer normalization
        self.cls_head = nn.Sequential(
            nn.LayerNorm(d_model),  # Pre-norm before classification
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )
        
        # Regression head with layer normalization
        self.reg_head = nn.Sequential(
            nn.LayerNorm(d_model),  # Pre-norm before regression
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2),
            nn.ReLU()
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, batch):
        visual_feats = batch['visual_feats']
        audio_feats = batch['audio_feats']
        text_feats = batch['text_feats']
        masks = batch['masks']
        gt_cls_labels = batch['labels']
        gt_offsets = batch['segments']
        
        # Concatenate all three modalities
        concatenated_feats = torch.cat([visual_feats, audio_feats, text_feats], dim=-1)
        
        # Project concatenated features to d_model
        projected_feats = self.input_projection(concatenated_feats)
        
        # Apply input normalization
        projected_feats = self.input_norm(projected_feats)
        
        # Add positional encoding
        projected_feats = self.positional_encoding(projected_feats)
        
        # Create attention mask for PyTorch transformer
        # PyTorch expects True values to be masked (ignored)
        src_key_padding_mask = (masks == 0)
        
        # Encode the features via transformer
        encoded_feats = self.multimodal_encoder(
            projected_feats,
            src_key_padding_mask=src_key_padding_mask
        )
        
        # Apply post-encoder normalization
        encoded_feats = self.encoder_norm(encoded_feats)
        
        # Apply feature mapping
        feats = self.feature_map(encoded_feats)
        
        # out_cls: [B, seq_len, 1]
        out_cls_logits = self.cls_head(feats)
        # out_offset: [B, seq_len, 2]
        out_offsets = self.reg_head(feats)

        return masks, out_cls_logits, out_offsets, gt_cls_labels, gt_offsets, feats

    @property
    def device(self):
        # a hacky way to get the device type
        # will throw an error if parameters are on different devices
        return list(set(p.device for p in self.parameters()))[0]

    def losses(
        self, masks,
        out_cls_logits, out_offsets,
        gt_cls_labels, gt_offsets, feats
    ):
        # mask: batch_size, max_len
        # out_cls_logits: List[B, seq_len, 1]
        # out_offsets: List[B, seq_len, 2]
        # gt_cls_labels: List[B, seq_len, 1]
        # gt_offsets: List[B, seq_len, 2]

        # 1. classification loss
        gt_cls_labels = gt_cls_labels.unsqueeze(-1)
        cls_loss = sigmoid_focal_loss(out_cls_logits, gt_cls_labels)

        masks = masks.transpose(1, 2).contiguous()
        cls_loss = cls_loss * masks

        cls_loss = cls_loss.sum()

        return {'cls_loss': cls_loss}

    @torch.no_grad()
    def inference_single_video(self, masks, out_cls_logits, out_offsets, inference_settings):
        segs_all = []
        scores_all = []
        cls_idxs_all = []

        # sigmoid normalization for output logits
        pred_prob = (out_cls_logits.sigmoid().squeeze() * masks).flatten()

        # Apply filtering to make NMS faster
        # 1. Keep seg with confidence score > a threshold
        keep_idxs = (pred_prob > inference_settings['pre_nms_thresh'])

        pred_prob = pred_prob[keep_idxs]

        topk_idxs = keep_idxs.nonzero(as_tuple=True)[0]

        # 2. Keep top k top scoring boxes only
        num_topk = min(inference_settings['pre_nms_topk'], topk_idxs.size(0))
        pred_prob, idxs = pred_prob.sort(descending=True)
        pred_prob = pred_prob[:num_topk].clone()
        topk_idxs = topk_idxs[idxs[:num_topk]].clone()
        offsets = out_offsets[topk_idxs]

        # 3. compute predicted segments
        seg_left = topk_idxs - offsets[:, 0]
        seg_right = topk_idxs + offsets[:, 1]
        pred_segs = torch.stack((seg_left, seg_right), -1)

        # 4. Keep seg with duration > a threshold
        seg_durations = seg_right - seg_left
        keep_idxs2 = seg_durations > inference_settings['duration_thresh']
        keep_idxs3 = seg_durations < inference_settings['duration_thresh_max']

        keep_idxs2 = keep_idxs2 & keep_idxs3

        # *_all : N (filtered # of segments) x 2 / 1
        segs_all.append(pred_segs[keep_idxs2])
        scores_all.append(pred_prob[keep_idxs2])
        cls_idxs_all.append(topk_idxs[keep_idxs2])

        # cat along the seq_len
        segs_all, scores_all, cls_idxs_all = [
            torch.cat(x) for x in [segs_all, scores_all, cls_idxs_all]
        ]
        results = {'segments': segs_all,
                   'scores': scores_all,
                   'labels': cls_idxs_all}
        return results

    @torch.no_grad()
    def inference_(self, batch, inference_settings):

        masks, out_cls_logits, out_offsets, gt_cls_labels, gt_offsets, feats = self.forward(
            batch)

        # batch seq_len
        pred_prob = out_cls_logits.squeeze(-1)

        results = []

        # 1: gather video meta information
        vid_idxs = batch['video_id']
        vid_lens = batch['duration']

        # 2: inference on each single video and gather the results
        # upto this point, all results use timestamps defined on feature grids
        for idx, (vidx, vlen) in enumerate(
            zip(vid_idxs, vid_lens)
        ):
            # gather per-video outputs
            cls_logits_per_vid = pred_prob[idx]
            offsets_per_vid = out_offsets[idx]
            masks_per_vid = masks[idx]
            mins = vlen // 60
            max_seg_num = mins * inference_settings['max_seg_per_min']
            max_seg_num = int(np.ceil(max_seg_num))

            # inference on a single video (should always be the case)
            results_per_vid = self.inference_single_video(
                masks_per_vid,
                cls_logits_per_vid, offsets_per_vid, inference_settings
            )
            # results_per_vid_nms_idx = soft_nms_intervals(results_per_vid['scores'], results_per_vid['segments'], sigma=inference_settings['nms_sigma'], thresh=inference_settings['min_score'])
            results_per_vid_nms_idx = soft_nms_intervals_cpu(
                results_per_vid['scores'], results_per_vid['segments'], sigma=inference_settings['nms_sigma'], thresh=inference_settings['min_score'], max_seg_num=max_seg_num)
            results_per_vid['segments'] = results_per_vid['segments'][results_per_vid_nms_idx]
            results_per_vid['scores'] = results_per_vid['scores'][results_per_vid_nms_idx]
            results_per_vid['labels'] = results_per_vid['labels'][results_per_vid_nms_idx]
            # pass through video meta info
            results_per_vid['video_id'] = vidx
            results_per_vid['duration'] = vlen
            results.append(results_per_vid)

        return results