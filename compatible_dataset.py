"""
Sequence-level video dataset for RepurposeModel training.
Only supports sequence mode with dict batch format.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional
import numpy as np
import os
import json
from functools import lru_cache


class SequenceVideoDataset(Dataset):
    """
    Sequence-level video dataset that handles missing features.
    Returns dict format with sequence padding and masking.
    """

    def __init__(
        self,
        feature_dirs: Dict[str, str],
        annotation_file: str,
        cache_size: int = 32,
        use_mmap: bool = False,
        max_seq_length: Optional[int] = None,
        stride: int = 1,
        min_modalities: int = 1,
    ):
        self.feature_dirs = feature_dirs
        self.use_mmap = use_mmap
        self.max_seq_length = max_seq_length
        self.stride = stride
        self.min_modalities = min_modalities

        # Load annotations and filter videos
        self._load_annotations(annotation_file)

        # Set up caching
        if cache_size > 0:
            self._load_video = lru_cache(maxsize=cache_size)(self._load_video_impl)
        else:
            self._load_video = self._load_video_impl

    def _load_annotations(self, annotation_file: str):
        """Load and filter annotations based on feature availability."""
        with open(annotation_file, "r") as f:
            annotations = json.load(f)

        self.annotations = []  # Store annotations by index
        self.feature_status = []  # Store feature status by index

        complete_videos = 0
        partial_videos = 0
        length_mismatch_filtered = 0

        for ann in annotations:
            video_id = ann["youtube_id"]
            time_range = ann.get("timeRangeOffset", [0, 0])
            expected_length = int(time_range[1] - time_range[0])

            # Check feature availability and lengths
            available_features = {}
            feature_lengths = {}
            for modality, feat_dir in self.feature_dirs.items():
                feat_path = os.path.join(feat_dir, f"{video_id}.npy")
                if os.path.exists(feat_path):
                    available_features[modality] = True
                    # Load just the shape to check length
                    feat_shape = np.load(feat_path, mmap_mode="r").shape
                    feature_lengths[modality] = feat_shape[0]
                else:
                    available_features[modality] = False

            # Include if minimum modalities are available
            available_count = sum(available_features.values())
            if available_count >= self.min_modalities:
                # Check if feature lengths are reasonable (within ±1 of expected)
                length_ok = True
                for modality, length in feature_lengths.items():
                    # Apply timeRange slicing to get actual length
                    time_range_full = ann.get("timeRange", [0, 0])
                    start_idx = int(time_range_full[0])
                    end_idx = int(time_range_full[1])
                    sliced_length = min(length, end_idx) - start_idx

                    # Check if sliced length matches expected length (±1 tolerance)
                    if abs(sliced_length - expected_length) > 1:
                        length_ok = False
                        length_mismatch_filtered += 1
                        break

                if length_ok:
                    self.annotations.append(ann)
                    self.feature_status.append(available_features)

                    if available_count == len(self.feature_dirs):
                        complete_videos += 1
                    else:
                        partial_videos += 1

        print(f"Dataset loaded: {len(self.annotations)} entries total")
        print(f"  Complete features: {complete_videos}, Partial: {partial_videos}")
        if length_mismatch_filtered > 0:
            print(
                f"  Filtered out {length_mismatch_filtered} entries due to length mismatch"
            )

        # Print availability stats
        for modality in self.feature_dirs.keys():
            count = sum(1 for status in self.feature_status if status[modality])
            print(f"  {modality}: {count}/{len(self.annotations)} entries")

    def _load_video_impl(self, idx: int) -> Dict[str, Optional[np.ndarray]]:
        """Load video features, handling missing modalities."""
        video_id = self.annotations[idx]["youtube_id"]
        features = {}
        available_status = self.feature_status[idx]

        for modality, feat_dir in self.feature_dirs.items():
            if not available_status[modality]:
                features[modality] = None
                continue

            feat_path = os.path.join(feat_dir, f"{video_id}.npy")
            try:
                if self.use_mmap:
                    features[modality] = np.load(feat_path, mmap_mode="r")
                else:
                    features[modality] = np.load(feat_path)
            except Exception as e:
                print(f"Warning: Failed to load {modality} for {video_id}: {e}")
                features[modality] = None

        return features

    def _get_feature_dim(self, modality: str) -> int:
        """Get feature dimension for a modality."""
        # Try to get dimension from first available feature
        for idx, status in enumerate(self.feature_status):
            if status.get(modality, False):
                features = self._load_video(idx)
                if features[modality] is not None:
                    return features[modality].shape[-1]

        # Defaults
        default_dims = {
            "audio": 2048,
            "visual": 512,
            "caption": 384,
            "video": 512,
            "text": 384,
        }
        return default_dims.get(modality, 512)

    def _get_labels(self, ann: dict, num_frames: int) -> np.ndarray:
        """Generate frame-level labels."""
        segments = ann.get("segmentsOffset", [])

        # Create integer timestamps (frame indices)
        timestamps = np.arange(num_frames, dtype=np.int32)
        labels = np.zeros(num_frames, dtype=np.float32)

        for idx, t in enumerate(timestamps):
            # Round segment boundaries to nearest integer frame
            if any(round(start) <= t < round(end) for start, end in segments):
                labels[idx] = 1.0

        return labels

    def _get_regression_offsets(self, ann: dict, num_frames: int) -> np.ndarray:
        """Generate regression offsets for each frame."""
        segments = ann.get("segmentsOffset", [])

        # Create integer timestamps (frame indices)
        timestamps = np.arange(num_frames, dtype=np.int32)
        # Initialize with zeros - shape: (num_frames, 2) for [left_offset, right_offset]
        offsets = np.zeros((num_frames, 2), dtype=np.float32)

        for idx, t in enumerate(timestamps):
            # Find if this timestamp is inside any segment
            for start, end in segments:
                # Round segment boundaries to nearest integer frame
                start_frame = round(start)
                end_frame = round(end)

                if start_frame <= t < end_frame:
                    # Calculate offsets to segment boundaries (will be integers)
                    left_offset = float(t - start_frame)
                    right_offset = float(end_frame - t - 1)
                    offsets[idx] = [left_offset, right_offset]
                    break  # Use first matching segment

        return offsets

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        return self._get_sequence(idx)

    def _get_sequence(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get full sequence - returns dict for custom collate_fn."""
        features = self._load_video(idx)
        ann = self.annotations[idx]
        video_id = ann["youtube_id"]
        time_range = ann.get("timeRange", [0, 0])

        # Get available features and sequence length
        available_features = {k: v for k, v in features.items() if v is not None}
        if not available_features:
            raise ValueError(f"No features available for {video_id}")

        # Always apply timeRange slicing to cap memory usage (like original paper)
        for modality in available_features.keys():
            start_idx = int(time_range[0])
            end_idx = int(time_range[1])
            available_features[modality] = available_features[modality][
                start_idx:end_idx
            ]

        min_length = min(f.shape[0] for f in available_features.values())
        indices = slice(0, min_length, self.stride)
        if self.max_seq_length:
            indices = slice(0, min(min_length, self.max_seq_length), self.stride)

        # Generate labels and regression offsets FIRST
        time_range = ann.get("timeRangeOffset", [0, 0])
        target_seq_length = int(time_range[1] - time_range[0])
        labels = self._get_labels(ann, target_seq_length)
        offsets = self._get_regression_offsets(ann, target_seq_length)

        # HACK FOR TESTING: Replace visual features with binary label encoding
        # Create a feature vector that's just the label repeated across feature dimension
        USE_TRIVIAL_FEATURES = True  # Toggle this to enable/disable the hack

        # Process features
        output_features = {}
        feature_masks = {}

        for modality in self.feature_dirs.keys():
            if modality == "visual" and USE_TRIVIAL_FEATURES:
                # Replace visual features with trivial encoding
                feat_dim = self._get_feature_dim(modality)
                # Create feature where all dimensions are set to the label value
                # Shape: [target_seq_length, feat_dim]
                trivial_features = np.repeat(labels[:, np.newaxis], feat_dim, axis=1)
                output_features[modality] = torch.from_numpy(
                    trivial_features.astype(np.float32)
                )
                feature_masks[modality] = True
            elif features[modality] is not None:
                # Always apply timeRange slicing to cap memory usage
                feat_data = features[modality]
                start_idx = int(time_range[0])
                end_idx = int(time_range[1])
                feat_data = feat_data[start_idx:end_idx]

                output_features[modality] = torch.from_numpy(
                    feat_data[indices].astype(np.float32)
                )
                feature_masks[modality] = True
            else:
                # Zero placeholder - use reference shape from sliced features
                ref_shape = next(iter(available_features.values()))[indices].shape
                feat_dim = self._get_feature_dim(modality)
                output_features[modality] = torch.zeros(
                    (ref_shape[0], feat_dim), dtype=torch.float32
                )
                feature_masks[modality] = False

        # Adjust features to match target sequence length (handle ±1 duration differences)
        for modality in output_features:
            current_length = output_features[modality].shape[0]
            if current_length > target_seq_length:
                # Trim if longer
                output_features[modality] = output_features[modality][
                    :target_seq_length
                ]
            elif current_length < target_seq_length:
                # Pad if shorter (handle off-by-1 cases)
                feat_dim = output_features[modality].shape[1]
                pad_length = target_seq_length - current_length
                padding = torch.zeros(
                    (pad_length, feat_dim), dtype=output_features[modality].dtype
                )
                output_features[modality] = torch.cat(
                    [output_features[modality], padding], dim=0
                )

        return {
            "video_id": video_id,
            "features": output_features,
            "feature_masks": feature_masks,
            "labels": torch.from_numpy(labels),
            "offsets": torch.from_numpy(offsets),  # Shape: [seq_len, 2]
            # Ground truth segments
            "gt_segments": ann.get("segmentsOffset", []),
            "duration": ann.get("duration", 0),
        }


def create_sequence_dataloader(
    feature_dirs: Dict[str, str],
    annotation_file: str,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: Optional[bool] = None,
    **kwargs,
) -> DataLoader:
    """
    Create a sequence-level DataLoader.

    Args:
        feature_dirs: Dict mapping modality names to directories
        annotation_file: Path to annotations JSON
        batch_size: Batch size
        num_workers: Number of data loading workers (default: 4)
        shuffle: Whether to shuffle data (default: True)
        pin_memory: Whether to pin memory for GPU transfer (default: auto-detect)
        **kwargs: Additional dataset arguments

    Returns:
        DataLoader with sequence batches in dict format
    """
    dataset = SequenceVideoDataset(
        feature_dirs=feature_dirs, annotation_file=annotation_file, **kwargs
    )

    # Custom collation for sequences
    def sequence_collate_fn(batch):
        max_len = max(sample["labels"].shape[0] for sample in batch)

        output = {
            "video_ids": [s["video_id"] for s in batch],
            "features": {},
            "feature_masks": {},
            "labels": [],
            "offsets": [],  # Add regression offsets
            # Ground truth segments
            "gt_segments": [s["gt_segments"] for s in batch],
            "sequence_masks": [],
        }

        modalities = list(batch[0]["features"].keys())
        for modality in modalities:
            output["features"][modality] = []
            output["feature_masks"][modality] = []

        for sample in batch:
            seq_len = sample["labels"].shape[0]

            # Pad features
            for modality in modalities:
                feat = sample["features"][modality]
                if seq_len < max_len:
                    pad_size = (max_len - seq_len, feat.shape[-1])
                    feat = torch.cat([feat, torch.zeros(pad_size)], dim=0)
                output["features"][modality].append(feat)
                output["feature_masks"][modality].append(
                    sample["feature_masks"][modality]
                )

            # Pad labels
            labels = sample["labels"]
            if seq_len < max_len:
                labels = torch.cat([labels, torch.zeros(max_len - seq_len)], dim=0)
            output["labels"].append(labels)

            # Pad offsets - shape: [seq_len, 2]
            offsets = sample["offsets"]
            if seq_len < max_len:
                pad_size = (max_len - seq_len, 2)
                offsets = torch.cat([offsets, torch.zeros(pad_size)], dim=0)
            output["offsets"].append(offsets)

            # Sequence mask
            seq_mask = torch.ones(max_len)
            if seq_len < max_len:
                seq_mask[seq_len:] = 0
            output["sequence_masks"].append(seq_mask)

        # Stack tensors
        for modality in modalities:
            output["features"][modality] = torch.stack(output["features"][modality])
            output["feature_masks"][modality] = torch.tensor(
                output["feature_masks"][modality]
            )
        output["labels"] = torch.stack(output["labels"])
        # Shape: [batch_size, max_len, 2]
        output["offsets"] = torch.stack(output["offsets"])
        output["sequence_masks"] = torch.stack(output["sequence_masks"])

        return output

    collate_fn = sequence_collate_fn

    # Set default pin_memory if not provided
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=True,
        prefetch_factor=2,
    )


if __name__ == "__main__":
    # Test sequence dataset
    feature_dirs = {
        "audio": "audio_pann_features",
        "visual": "video_clip_features",
        "caption": "caption_features",
    }

    print("Testing sequence dataset...")
    seq_loader = create_sequence_dataloader(
        feature_dirs=feature_dirs,
        annotation_file="test.json",
        batch_size=4,
        max_seq_length=500,
    )

    for i, batch in enumerate(seq_loader):
        print(f"Sequence batch {i}:")
        for mod, feat in batch["features"].items():
            print(f"  {mod}: {feat.shape}")
        print(f"  labels: {batch['labels'].shape}")
        if i >= 1:
            break
