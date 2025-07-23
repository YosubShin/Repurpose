from .transformer import *
from .losses import ctr_diou_loss_1d, sigmoid_focal_loss
from .softnms import soft_nms_intervals_cpu
import numpy as np


class MMCTransformer(nn.Module):
    def __init__(self, vis_dim, aud_dim, text_dim, d_model, self_num_layers, text_num_layers, cross_num_layers, num_heads, d_ff=2048):
        super(MMCTransformer, self).__init__()
        # Audio-only encoder
        self.aud_encoder = UniModalEncoder(
            aud_dim, d_model, self_num_layers, num_heads, d_ff)

        hidden_dim = 256

        self.feature_map = nn.Linear(d_model, d_model)

        self.cls_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

        self.reg_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, 2),
            nn.ReLU()
        )

    def forward(self, batch):
        aud_feats = batch['audio_feats']
        masks = batch['masks']
        gt_cls_labels = batch['labels']
        gt_offsets = batch['segments']
        # preprocessing
        # aud: [batch_size, seq_len, aud_dim]
        # masks: [batch_size, 1, max_len]

        # encode the features via self-attention
        aud_feats = self.aud_encoder(aud_feats, masks)

        # use audio features directly
        feats = self.feature_map(aud_feats)

        # out_cls: List[B, #cls, seq_len]
        out_cls_logits = self.cls_head(feats)
        # out_offset: List[B, 2, seq_len]
        out_offsets = self.reg_head(feats)

        return masks, out_cls_logits, out_offsets, gt_cls_labels, gt_offsets

    @property
    def device(self):
        # a hacky way to get the device type
        # will throw an error if parameters are on different devices
        return list(set(p.device for p in self.parameters()))[0]

    def losses(
        self, masks,
        out_cls_logits, out_offsets,
        gt_cls_labels, gt_offsets
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

        # 2. regression loss
        cls_mask = (gt_cls_labels != 0).float()

        combined_mask = masks * cls_mask

        reg_loss = ctr_diou_loss_1d(
            out_offsets,
            gt_offsets,
        )

        reg_loss = (reg_loss * combined_mask.squeeze(-1)).sum()

        return {'cls_loss': cls_loss,
                'reg_loss': reg_loss}

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

        masks, out_cls_logits, out_offsets, gt_cls_labels, gt_offsets = self.forward(
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
