"""
Efficient dataset implementation for video repurposing that scales to large datasets.
Features:
- Lazy loading: Only loads features when needed
- Memory-mapped arrays for large files
- Caching of frequently accessed videos
- Support for both frame-level and sequence-level datasets
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)


class EfficientVideoDataset(Dataset):
    """
    Efficient dataset that loads video features on-demand.
    Supports both frame-level and sequence-level access.
    """
    
    def __init__(
        self,
        feature_dirs: Dict[str, str],
        annotation_file: str,
        mode: str = 'sequence',  # 'sequence' or 'frame'
        cache_size: int = 32,    # Number of videos to keep in memory
        use_mmap: bool = True,   # Use memory-mapped arrays
        max_seq_length: Optional[int] = None,  # Truncate long sequences
        stride: int = 1,         # Downsample sequences
    ):
        """
        Args:
            feature_dirs: Dict mapping modality names to feature directories
            annotation_file: Path to JSON file with annotations
            mode: 'sequence' returns full videos, 'frame' returns individual frames
            cache_size: Number of videos to cache in memory
            use_mmap: Whether to use memory-mapped arrays (saves memory)
            max_seq_length: Maximum sequence length (truncate if longer)
            stride: Temporal downsampling factor
        """
        self.feature_dirs = feature_dirs
        self.mode = mode
        self.use_mmap = use_mmap
        self.max_seq_length = max_seq_length
        self.stride = stride
        
        # Load annotations
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)
        
        # Filter videos and track feature availability
        self.video_list = []
        self.video_to_annotation = {}
        self.video_feature_status = {}  # Track which features are available per video
        
        videos_with_complete_features = 0
        videos_with_partial_features = 0
        
        for ann in annotations:
            video_id = ann['youtube_id']
            
            # Check which features are available
            available_features = {}
            for modality, feat_dir in feature_dirs.items():
                feat_path = os.path.join(feat_dir, f"{video_id}.npy")
                available_features[modality] = os.path.exists(feat_path)
            
            # Include video if at least one feature is available
            if any(available_features.values()):
                self.video_list.append(video_id)
                self.video_to_annotation[video_id] = ann
                self.video_feature_status[video_id] = available_features
                
                if all(available_features.values()):
                    videos_with_complete_features += 1
                else:
                    videos_with_partial_features += 1
        
        logger.info(f"Found {len(self.video_list)} videos total:")
        logger.info(f"  - {videos_with_complete_features} with complete features")
        logger.info(f"  - {videos_with_partial_features} with partial features")
        
        # Log missing modality statistics
        modality_counts = {mod: 0 for mod in feature_dirs.keys()}
        for video_id, status in self.video_feature_status.items():
            for mod, available in status.items():
                if available:
                    modality_counts[mod] += 1
        
        logger.info("Feature availability by modality:")
        for mod, count in modality_counts.items():
            logger.info(f"  - {mod}: {count}/{len(self.video_list)} videos")
        
        # Set up caching
        if cache_size > 0:
            self._load_video_cached = lru_cache(maxsize=cache_size)(self._load_video)
        else:
            self._load_video_cached = self._load_video
        
        # For frame mode, pre-compute frame indices
        if mode == 'frame':
            self._build_frame_index()
    
    def _build_frame_index(self):
        """Build index for frame-level access."""
        self.frame_to_video = []  # List of (video_id, frame_idx) tuples
        
        for video_id in self.video_list:
            # Get video length without loading features
            sample_path = os.path.join(
                self.feature_dirs[list(self.feature_dirs.keys())[0]], 
                f"{video_id}.npy"
            )
            if self.use_mmap:
                arr = np.load(sample_path, mmap_mode='r')
                num_frames = len(arr)
            else:
                # Load just to get shape
                arr = np.load(sample_path)
                num_frames = len(arr)
                del arr  # Free memory
            
            # Add frame indices with stride
            for frame_idx in range(0, num_frames, self.stride):
                if self.max_seq_length is None or frame_idx < self.max_seq_length:
                    self.frame_to_video.append((video_id, frame_idx))
        
        logger.info(f"Built frame index with {len(self.frame_to_video)} frames")
    
    def _load_video(self, video_id: str) -> Dict[str, Optional[np.ndarray]]:
        """Load all modality features for a video, handling missing features."""
        features = {}
        available_status = self.video_feature_status[video_id]
        
        for modality, feat_dir in self.feature_dirs.items():
            if not available_status[modality]:
                features[modality] = None
                continue
                
            feat_path = os.path.join(feat_dir, f"{video_id}.npy")
            
            try:
                if self.use_mmap:
                    # Memory-mapped array - doesn't load into RAM until accessed
                    features[modality] = np.load(feat_path, mmap_mode='r')
                else:
                    features[modality] = np.load(feat_path)
            except Exception as e:
                logger.warning(f"Failed to load {modality} features for {video_id}: {e}")
                features[modality] = None
        
        return features
    
    def _get_labels(self, video_id: str, num_frames: int) -> np.ndarray:
        """Generate frame-level labels from annotations."""
        ann = self.video_to_annotation[video_id]
        segments = ann.get('segmentsOffset', [])
        time_start, time_end = ann.get('timeRangeOffset', [0, 0])
        
        # Create timestamps
        timestamps = np.linspace(time_start, time_end, num=num_frames, endpoint=False)
        
        # Generate labels
        labels = np.zeros(num_frames, dtype=np.float32)
        for idx, t in enumerate(timestamps):
            if any(start <= t < end for start, end in segments):
                labels[idx] = 1.0
        
        return labels
    
    def __len__(self):
        if self.mode == 'sequence':
            return len(self.video_list)
        else:  # frame mode
            return len(self.frame_to_video)
    
    def __getitem__(self, idx):
        if self.mode == 'sequence':
            return self._get_sequence(idx)
        else:
            return self._get_frame(idx)
    
    def _get_sequence(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get full sequence for a video, handling missing features."""
        video_id = self.video_list[idx]
        features = self._load_video_cached(video_id)
        
        # Get minimum length across available modalities
        available_features = {k: v for k, v in features.items() if v is not None}
        if not available_features:
            raise ValueError(f"No features available for video {video_id}")
            
        lengths = [feat.shape[0] for feat in available_features.values()]
        min_length = min(lengths)
        
        # Apply stride and max length
        indices = slice(0, min_length, self.stride)
        if self.max_seq_length is not None:
            indices = slice(0, min(min_length, self.max_seq_length), self.stride)
        
        # Extract features
        output = {
            'video_id': video_id,
            'features': {},
            'feature_mask': {}  # Indicates which features are available
        }
        
        for modality, feat in features.items():
            if feat is not None:
                output['features'][modality] = torch.from_numpy(
                    feat[indices].astype(np.float32)
                )
                output['feature_mask'][modality] = True
            else:
                # Create zero tensor as placeholder
                # Use shape from first available feature
                ref_shape = next(iter(available_features.values()))[indices].shape
                feat_dim = self._get_feature_dim(modality)
                placeholder_shape = (ref_shape[0], feat_dim)
                output['features'][modality] = torch.zeros(placeholder_shape, dtype=torch.float32)
                output['feature_mask'][modality] = False
        
        # Get labels
        labels = self._get_labels(video_id, min_length)[indices]
        output['labels'] = torch.from_numpy(labels)
        
        # Add metadata
        output['duration'] = self.video_to_annotation[video_id].get('duration', 0)
        
        return output
    
    def _get_feature_dim(self, modality: str) -> int:
        """Get feature dimension for a modality by checking available videos."""
        for video_id in self.video_list:
            if self.video_feature_status[video_id][modality]:
                features = self._load_video_cached(video_id)
                if features[modality] is not None:
                    return features[modality].shape[-1]
        
        # Default dimensions if not found (common feature sizes)
        default_dims = {
            'audio': 2048,
            'visual': 512, 
            'caption': 384,
            'video': 512,
            'text': 384
        }
        return default_dims.get(modality, 512)
    
    def _get_frame(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get single frame, handling missing features."""
        video_id, frame_idx = self.frame_to_video[idx]
        features = self._load_video_cached(video_id)
        
        # Concatenate available features and create mask
        frame_features = []
        feature_mask = []
        
        for modality in sorted(self.feature_dirs.keys()):
            if features[modality] is not None:
                frame_features.append(features[modality][frame_idx])
                feature_mask.append(1.0)
            else:
                # Use zero padding for missing features
                feat_dim = self._get_feature_dim(modality)
                frame_features.append(np.zeros(feat_dim, dtype=np.float32))
                feature_mask.append(0.0)
        
        x = np.concatenate(frame_features, axis=-1)
        mask = np.array(feature_mask, dtype=np.float32)
        
        # Get label
        available_features = [f for f in features.values() if f is not None]
        min_length = min(feat.shape[0] for feat in available_features)
        labels = self._get_labels(video_id, min_length)
        y = labels[frame_idx]
        
        return (torch.from_numpy(x.astype(np.float32)), 
                torch.tensor(y, dtype=torch.float32),
                torch.from_numpy(mask))


class EfficientCollator:
    """Custom collator for sequence mode that handles variable-length sequences."""
    
    def __init__(self, pad_value: float = 0.0):
        self.pad_value = pad_value
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate sequences with padding, handling missing features."""
        # Find max sequence length in batch
        max_len = max(sample['labels'].shape[0] for sample in batch)
        
        # Prepare output
        output = {
            'video_ids': [s['video_id'] for s in batch],
            'durations': torch.tensor([s['duration'] for s in batch]),
            'features': {},
            'feature_masks': {},  # Indicates which features are available per sample
            'labels': [],
            'masks': []
        }
        
        # Get modality names from first sample
        modalities = list(batch[0]['features'].keys())
        
        # Initialize feature tensors
        for modality in modalities:
            feat_dim = batch[0]['features'][modality].shape[-1]
            output['features'][modality] = []
            output['feature_masks'][modality] = []
        
        # Process each sample
        for sample in batch:
            seq_len = sample['labels'].shape[0]
            
            # Pad features
            for modality in modalities:
                feat = sample['features'][modality]
                if seq_len < max_len:
                    pad_size = (max_len - seq_len, feat.shape[-1])
                    feat = torch.cat([
                        feat,
                        torch.full(pad_size, self.pad_value)
                    ], dim=0)
                output['features'][modality].append(feat)
                
                # Feature availability mask
                feat_available = sample['feature_mask'][modality]
                output['feature_masks'][modality].append(feat_available)
            
            # Pad labels
            labels = sample['labels']
            if seq_len < max_len:
                labels = torch.cat([
                    labels,
                    torch.zeros(max_len - seq_len)
                ], dim=0)
            output['labels'].append(labels)
            
            # Create sequence mask (1 for valid, 0 for padded)
            mask = torch.ones(max_len)
            if seq_len < max_len:
                mask[seq_len:] = 0
            output['masks'].append(mask)
        
        # Stack into tensors
        for modality in modalities:
            output['features'][modality] = torch.stack(output['features'][modality])
            output['feature_masks'][modality] = torch.tensor(output['feature_masks'][modality])
        output['labels'] = torch.stack(output['labels'])
        output['masks'] = torch.stack(output['masks'])
        
        return output


def create_efficient_dataloader(
    feature_dirs: Dict[str, str],
    annotation_file: str,
    batch_size: int = 32,
    mode: str = 'frame',
    num_workers: int = 4,
    **dataset_kwargs
) -> DataLoader:
    """
    Create an efficient dataloader.
    
    Example:
        feature_dirs = {
            'audio': 'path/to/audio_features',
            'visual': 'path/to/visual_features',
            'caption': 'path/to/caption_features'
        }
        
        # Frame-level dataloader
        frame_loader = create_efficient_dataloader(
            feature_dirs, 'annotations.json',
            batch_size=256, mode='frame'
        )
        
        # Sequence-level dataloader
        seq_loader = create_efficient_dataloader(
            feature_dirs, 'annotations.json',
            batch_size=8, mode='sequence',
            max_seq_length=1000, stride=2
        )
    """
    dataset = EfficientVideoDataset(
        feature_dirs=feature_dirs,
        annotation_file=annotation_file,
        mode=mode,
        **dataset_kwargs
    )
    
    if mode == 'sequence':
        collator = EfficientCollator()
        # Smaller batch size for sequences
        actual_batch_size = min(batch_size, 8)
    else:
        collator = None
        actual_batch_size = batch_size
    
    return DataLoader(
        dataset,
        batch_size=actual_batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0
    )


if __name__ == '__main__':
    # Example usage
    import time
    
    feature_dirs = {
        'audio': 'audio_pann_features',
        'visual': 'video_clip_features', 
        'caption': 'caption_features'
    }
    
    # Test frame-level loading
    print("Testing frame-level dataset...")
    frame_dataset = EfficientVideoDataset(
        feature_dirs=feature_dirs,
        annotation_file='test.json',
        mode='frame',
        stride=10  # Downsample by 10x
    )
    
    start = time.time()
    for i in range(min(100, len(frame_dataset))):
        x, y = frame_dataset[i]
        if i == 0:
            print(f"Frame shape: {x.shape}, Label: {y.item()}")
    print(f"Loaded 100 frames in {time.time() - start:.2f}s")
    
    # Test sequence-level loading
    print("\nTesting sequence-level dataset...")
    seq_dataset = EfficientVideoDataset(
        feature_dirs=feature_dirs,
        annotation_file='test.json',
        mode='sequence',
        max_seq_length=500
    )
    
    for i in range(min(2, len(seq_dataset))):
        sample = seq_dataset[i]
        print(f"Video {sample['video_id']}:")
        for mod, feat in sample['features'].items():
            print(f"  {mod}: {feat.shape}")
        print(f"  labels: {sample['labels'].shape}")