import torch
import torch.nn as nn
from .losses import sigmoid_focal_loss
from .softnms import soft_nms_intervals_cpu
import numpy as np


class SimpleMCTransformer(nn.Module):
    """
    Minimal model for debugging - just linear layers, no transformers.
    This should be able to overfit on 4 samples easily.
    """

    def __init__(self, vis_dim, aud_dim, text_dim, d_model, self_num_layers, text_num_layers, cross_num_layers, num_heads, d_ff=2048):
        super(SimpleMCTransformer, self).__init__()

        # Concatenated feature dimension
        concat_dim = vis_dim + aud_dim + text_dim

        # Super simple architecture - just linear layers
        self.layers = nn.Sequential(
            nn.Linear(concat_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # Direct prediction heads - no fancy stuff
        self.cls_head = nn.Linear(128, 1)  # Binary classification
        self.reg_head = nn.Sequential(
            nn.Linear(128, 2),
            nn.ReLU()  # Ensure positive offsets
        )

        # Initialize weights
        self._init_weights()

        print(f"SimpleMCTransformer initialized with input_dim={concat_dim}")

    def _init_weights(self):
        """Simple weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    # Special initialization for classification head
                    if m.out_features == 1:  # This is the cls_head
                        nn.init.constant_(m.bias, 0.5)  # Slight positive bias
                        print("Initialized cls_head with positive bias = 0.5")
                    else:
                        nn.init.constant_(m.bias, 0)

    def forward(self, batch):
        visual_feats = batch['visual_feats']
        audio_feats = batch['audio_feats']
        text_feats = batch['text_feats']
        masks = batch['masks']
        gt_cls_labels = batch['labels']
        gt_offsets = batch['segments']

        # Print shapes for debugging
        if not hasattr(self, '_printed_shapes'):
            print(
                f"Input shapes - visual: {visual_feats.shape}, audio: {audio_feats.shape}, text: {text_feats.shape}")
            print(f"Masks shape: {masks.shape}")
            print(f"Labels shape: {gt_cls_labels.shape}")
            self._printed_shapes = True

        # Concatenate all modalities
        concatenated_feats = torch.cat(
            [visual_feats, audio_feats, text_feats], dim=-1)

        # Apply simple layers
        feats = self.layers(concatenated_feats)

        # Get predictions
        out_cls_logits = self.cls_head(feats)
        out_offsets = self.reg_head(feats)

        return masks, out_cls_logits, out_offsets, gt_cls_labels, gt_offsets, feats

    @property
    def device(self):
        return list(set(p.device for p in self.parameters()))[0]

    def losses(self, masks, out_cls_logits, out_offsets, gt_cls_labels, gt_offsets, feats):
        # Ensure correct shapes
        if gt_cls_labels.dim() == 2:
            gt_cls_labels = gt_cls_labels.unsqueeze(-1)

        # Simple focal loss for classification
        # Try alpha=0.25 to give more weight to positive examples
        cls_loss = sigmoid_focal_loss(
            out_cls_logits, gt_cls_labels, alpha=0.25, reduction='none')

        # Alternative: Try standard BCE if focal loss doesn't work
        # cls_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        #     out_cls_logits, gt_cls_labels, reduction='none'
        # )

        # Apply mask
        if masks.dim() == 3:  # [batch, 1, seq_len]
            masks = masks.transpose(1, 2).contiguous()
        else:  # [batch, seq_len]
            masks = masks.unsqueeze(-1)

        cls_loss = cls_loss * masks

        # Debug prints
        if not hasattr(self, '_printed_loss_info'):
            print(
                f"Loss computation - cls_loss shape: {cls_loss.shape}, masks shape: {masks.shape}")
            print(
                f"Number of positive labels: {(gt_cls_labels > 0.5).sum().item()}")
            print(f"Number of valid positions (mask=1): {masks.sum().item()}")
            print(
                f"Mean cls_loss before masking: {sigmoid_focal_loss(out_cls_logits, gt_cls_labels, reduction='mean').item():.4f}")
            self._printed_loss_info = True

        # Average over all valid positions
        num_valid = masks.sum()
        if num_valid > 0:
            cls_loss = cls_loss.sum() / num_valid
        else:
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
        for idx, (vidx, vlen) in enumerate(zip(vid_idxs, vid_lens)):
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
            results_per_vid_nms_idx = soft_nms_intervals_cpu(
                results_per_vid['scores'], results_per_vid['segments'],
                sigma=inference_settings['nms_sigma'],
                thresh=inference_settings['min_score'],
                max_seg_num=max_seg_num
            )
            results_per_vid['segments'] = results_per_vid['segments'][results_per_vid_nms_idx]
            results_per_vid['scores'] = results_per_vid['scores'][results_per_vid_nms_idx]
            results_per_vid['labels'] = results_per_vid['labels'][results_per_vid_nms_idx]
            # pass through video meta info
            results_per_vid['video_id'] = vidx
            results_per_vid['duration'] = vlen
            results.append(results_per_vid)

        return results
