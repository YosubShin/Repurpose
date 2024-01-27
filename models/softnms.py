import numpy as np

def soft_nms_intervals_cpu(out_cls_logits, out_offsets, sigma=0.5, thresh=0.001, max_seg_num=20):
    out_cls_logits = out_cls_logits.cpu().numpy()
    out_offsets = out_offsets.cpu().numpy()
    seq_len = out_offsets.shape[0]
    indexes = np.arange(0, seq_len, dtype=np.float32).reshape(seq_len, 1)
    out_offsets = np.concatenate((out_offsets, indexes), axis=1)

    begin = out_offsets[:, 0]
    end = out_offsets[:, 1]
    scores = out_cls_logits
    lengths = end - begin
    
    max_segments = min(max_seg_num, seq_len)
    selected_counter = 0
    for i in range(seq_len):
        tscore = scores[i]
        pos = i + 1
        if i != seq_len - 1:
            maxscore = np.amax(scores[pos:], axis=0)
            maxpos = np.argmax(scores[pos:], axis=0)
            if tscore < maxscore:
                out_offsets[i], out_offsets[maxpos.item() + i + 1] = out_offsets[maxpos.item() + i + 1].copy(), out_offsets[i].copy()
                scores[i], scores[maxpos.item() + i + 1] = scores[maxpos.item() + i + 1].copy(), scores[i].copy()
        if tscore > thresh:
            selected_counter += 1
            if selected_counter >= max_segments:
                break  # Stop if we have selected enough segments
        max_begin = np.maximum(begin[i], begin[pos:])
        min_end = np.minimum(end[i], end[pos:])
        overlap = np.clip(min_end - max_begin, a_min=0, a_max=None)
        total_length = lengths[i] + lengths[pos:] - overlap
        overlap_ratio = overlap / total_length
        weight = np.exp(-(overlap_ratio * overlap_ratio) / sigma)
        scores[pos:] = weight * scores[pos:]
    keep = out_offsets[scores > thresh][:max_segments, 2].astype(int)
    return keep
