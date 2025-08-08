import torch
from torch.nn import functional as F

@torch.jit.script
def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.7,
    gamma: float = 2.0,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Taken from
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py
    # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = 0.25 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    inputs = inputs.float()
    targets = targets.float()

    p = torch.sigmoid(inputs)

    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


@torch.jit.script
def ctr_diou_loss_1d(
    input_offsets: torch.Tensor,
    target_offsets: torch.Tensor,
    reduction: str = 'none',
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Distance-IoU Loss (Zheng et. al) for 1D segments
    https://arxiv.org/abs/1911.08287

    This implementation assumes a 1D segment at position t is represented as:
    [t - left_offset, t + right_offset] where offsets >= 0
    
    The position t serves as the implicit center point for each segment.

    Reference code from
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/giou_loss.py

    Args:
        input/target_offsets (Tensor): 1D offsets of size (B, T, 2) where B is batch size, T is sequence length
                                       [:,:,0] = left offsets, [:,:,1] = right offsets
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
        eps (float): small number to prevent division by zero
    """
    input_offsets = input_offsets.float()
    target_offsets = target_offsets.float()
    # check all 1D events are valid
    assert (input_offsets >= 0.0).all(), "predicted offsets must be non-negative"
    assert (target_offsets >= 0.0).all(), "GT offsets must be non-negative"
        
    # Extract left and right offsets
    pred_left, pred_right = input_offsets[:,:,0], input_offsets[:,:,1]
    gt_left, gt_right = target_offsets[:,:,0], target_offsets[:,:,1]

    # For segments centered at position t:
    # pred segment: [t - pred_left, t + pred_right]
    # gt segment:   [t - gt_left, t + gt_right]
    
    # Intersection calculation
    # Left boundary of intersection = max(t - pred_left, t - gt_left) = t - min(pred_left, gt_left)
    # Right boundary of intersection = min(t + pred_right, t + gt_right) = t + min(pred_right, gt_right)
    # Length of intersection = (t + min_right) - (t - min_left) = min_left + min_right
    
    min_left = torch.min(pred_left, gt_left)
    min_right = torch.min(pred_right, gt_right)
    
    # Intersection length (only positive if segments overlap)
    # For segments to overlap: left_boundary < right_boundary
    # i.e., (t - min_left) < (t + min_right) which is always true if offsets > 0
    intersection = min_left + min_right
    
    # Union calculation
    pred_length = pred_left + pred_right
    gt_length = gt_left + gt_right
    union = pred_length + gt_length - intersection
    
    # IoU
    iou = intersection / union.clamp(min=eps)

    # Smallest enclosing box
    # Left boundary = min(t - pred_left, t - gt_left) = t - max(pred_left, gt_left)
    # Right boundary = max(t + pred_right, t + gt_right) = t + max(pred_right, gt_right)
    # Length = (t + max_right) - (t - max_left) = max_left + max_right
    max_left = torch.max(pred_left, gt_left)
    max_right = torch.max(pred_right, gt_right)
    enclosing_length = max_left + max_right

    # Center offset between predicted and ground truth segments
    # Pred center = t - pred_left/2 + pred_right/2 = t + (pred_right - pred_left)/2
    # GT center = t - gt_left/2 + gt_right/2 = t + (gt_right - gt_left)/2
    # Distance = pred_center - gt_center = (pred_right - pred_left)/2 - (gt_right - gt_left)/2
    center_distance = 0.5 * (pred_right - pred_left - gt_right + gt_left)

    # DIoU loss
    loss = 1.0 - iou + torch.square(center_distance / enclosing_length.clamp(min=eps))

    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


@torch.jit.script
def segment_consistency_loss(
    pred_offsets: torch.Tensor,
    labels: torch.Tensor,
    seq_mask: torch.Tensor,
    reduction: str = 'none',
) -> torch.Tensor:
    """
    Consistency loss that enforces adjacent positive positions to predict similar segment boundaries.
    
    Within a highlight segment, all positions should predict the same absolute boundaries:
    - left_boundary = position - left_offset (should be constant within segment)
    - right_boundary = position + right_offset (should be constant within segment)
    
    Args:
        pred_offsets: Predicted offsets [B, T, 2] where [:,:,0] = left, [:,:,1] = right
        labels: Binary labels [B, T] indicating positive positions
        seq_mask: Sequence mask [B, T] indicating valid positions
        reduction: 'none' | 'mean' | 'sum'
    
    Returns:
        Consistency loss encouraging adjacent positives to predict same boundaries
    """
    batch_size, seq_len = labels.shape
    device = pred_offsets.device
    
    # Create position indices [B, T]
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    positions = positions.unsqueeze(0).expand(batch_size, seq_len)
    
    # Calculate predicted absolute boundaries for each position
    # left_boundary = position - left_offset
    # right_boundary = position + right_offset
    pred_left_boundary = positions - pred_offsets[:, :, 0]
    pred_right_boundary = positions + pred_offsets[:, :, 1]
    
    # Identify positive positions
    positive_mask = (labels > 0.5).float() * seq_mask
    
    # Check if adjacent positions are both positive
    # Shape: [B, T-1]
    adjacent_positives = positive_mask[:, :-1] * positive_mask[:, 1:]
    
    # Calculate boundary differences between adjacent positions
    # If they belong to the same segment, these should be ~0
    left_diff = torch.abs(pred_left_boundary[:, :-1] - pred_left_boundary[:, 1:])
    right_diff = torch.abs(pred_right_boundary[:, :-1] - pred_right_boundary[:, 1:])
    
    # Combined consistency loss
    consistency = (left_diff + right_diff) * adjacent_positives
    
    if reduction == 'none':
        return consistency
    elif reduction == 'mean':
        # Mean over positions that are adjacent positives
        num_adjacent = adjacent_positives.sum()
        if num_adjacent > 0:
            return consistency.sum() / num_adjacent
        else:
            return torch.tensor(0.0, device=device)
    elif reduction == 'sum':
        return consistency.sum()
    else:
        raise ValueError(f"Invalid reduction mode: {reduction}")
