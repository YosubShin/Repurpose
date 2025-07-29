"""
DataLoader-compatible wrapper for RobustVideoDataset.
Handles the "too many values to unpack" error by providing proper tuple/dict returns.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Union, Optional
import numpy as np
import os
import json
from functools import lru_cache


class CompatibleVideoDataset(Dataset):
    """
    DataLoader-compatible video dataset that handles missing features.
    
    For frame mode: Returns (x, y) tuples
    For sequence mode: Use with custom collate_fn
    """
    
    def __init__(
        self,
        feature_dirs: Dict[str, str],
        annotation_file: str,
        mode: str = 'frame',  # 'frame' or 'sequence'
        cache_size: int = 32,
        use_mmap: bool = True,
        max_seq_length: Optional[int] = None,
        stride: int = 1,
        min_modalities: int = 1,
    ):
        self.feature_dirs = feature_dirs
        self.mode = mode
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
        
        # Build indices for frame mode
        if mode == 'frame':
            self._build_frame_index()
    
    def _load_annotations(self, annotation_file: str):
        """Load and filter annotations based on feature availability."""
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)
        
        self.video_list = []
        self.video_to_annotation = {}
        self.video_feature_status = {}
        
        complete_videos = 0
        partial_videos = 0
        
        for ann in annotations:
            video_id = ann['youtube_id']
            
            # Check feature availability
            available_features = {}
            for modality, feat_dir in self.feature_dirs.items():
                feat_path = os.path.join(feat_dir, f"{video_id}.npy")
                available_features[modality] = os.path.exists(feat_path)
            
            # Include if minimum modalities are available
            available_count = sum(available_features.values())
            if available_count >= self.min_modalities:
                self.video_list.append(video_id)
                self.video_to_annotation[video_id] = ann
                self.video_feature_status[video_id] = available_features
                
                if available_count == len(self.feature_dirs):
                    complete_videos += 1
                else:
                    partial_videos += 1
        
        print(f"Dataset loaded: {len(self.video_list)} videos total")
        print(f"  Complete features: {complete_videos}, Partial: {partial_videos}")
        
        # Print availability stats
        for modality in self.feature_dirs.keys():
            count = sum(1 for status in self.video_feature_status.values() 
                       if status[modality])
            print(f"  {modality}: {count}/{len(self.video_list)} videos")
    
    def _build_frame_index(self):
        """Build frame-level index for efficient access."""
        self.frame_to_video = []
        
        for video_id in self.video_list:
            # Find first available modality
            available_modality = None
            for modality, available in self.video_feature_status[video_id].items():
                if available:
                    available_modality = modality
                    break
            
            if available_modality is None:
                continue
            
            # Get sequence length
            feat_path = os.path.join(self.feature_dirs[available_modality], f"{video_id}.npy")
            if self.use_mmap:
                arr = np.load(feat_path, mmap_mode='r')
            else:
                arr = np.load(feat_path)
            
            num_frames = len(arr)
            
            # Add frame indices
            for frame_idx in range(0, num_frames, self.stride):
                if self.max_seq_length is None or frame_idx < self.max_seq_length:
                    self.frame_to_video.append((video_id, frame_idx))
        
        print(f"Built frame index: {len(self.frame_to_video)} frames")
    
    def _load_video_impl(self, video_id: str) -> Dict[str, Optional[np.ndarray]]:
        """Load video features, handling missing modalities."""
        features = {}
        available_status = self.video_feature_status[video_id]
        
        for modality, feat_dir in self.feature_dirs.items():
            if not available_status[modality]:
                features[modality] = None
                continue
            
            feat_path = os.path.join(feat_dir, f"{video_id}.npy")
            try:
                if self.use_mmap:
                    features[modality] = np.load(feat_path, mmap_mode='r')
                else:
                    features[modality] = np.load(feat_path)
            except Exception as e:
                print(f"Warning: Failed to load {modality} for {video_id}: {e}")
                features[modality] = None
        
        return features
    
    def _get_feature_dim(self, modality: str) -> int:
        """Get feature dimension for a modality."""
        for video_id in self.video_list:
            if self.video_feature_status[video_id][modality]:
                features = self._load_video(video_id)
                if features[modality] is not None:
                    return features[modality].shape[-1]
        
        # Defaults
        default_dims = {
            'audio': 2048, 'visual': 512, 'caption': 384,
            'video': 512, 'text': 384
        }
        return default_dims.get(modality, 512)
    
    def _get_labels(self, video_id: str, num_frames: int) -> np.ndarray:
        """Generate frame-level labels."""
        ann = self.video_to_annotation[video_id]
        segments = ann.get('segmentsOffset', [])
        time_range = ann.get('timeRangeOffset', [0, 0])
        
        timestamps = np.linspace(time_range[0], time_range[1], num_frames, endpoint=False)
        labels = np.zeros(num_frames, dtype=np.float32)
        
        for idx, t in enumerate(timestamps):
            if any(start <= t < end for start, end in segments):
                labels[idx] = 1.0
        
        return labels
    
    def __len__(self):
        if self.mode == 'frame':
            return len(self.frame_to_video)
        else:
            return len(self.video_list)
    
    def __getitem__(self, idx):
        if self.mode == 'frame':
            return self._get_frame(idx)
        else:
            return self._get_sequence(idx)
    
    def _get_frame(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get single frame - returns (x, y) tuple compatible with DataLoader."""
        video_id, frame_idx = self.frame_to_video[idx]
        features = self._load_video(video_id)
        
        # Concatenate available features, pad missing ones
        frame_features = []
        for modality in sorted(self.feature_dirs.keys()):
            if features[modality] is not None:
                frame_features.append(features[modality][frame_idx])
            else:
                # Zero padding for missing features
                feat_dim = self._get_feature_dim(modality)
                frame_features.append(np.zeros(feat_dim, dtype=np.float32))
        
        x = np.concatenate(frame_features, axis=-1)
        
        # Get label
        available_features = [f for f in features.values() if f is not None]
        min_length = min(f.shape[0] for f in available_features)
        labels = self._get_labels(video_id, min_length)
        y = labels[frame_idx]
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    
    def _get_sequence(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get full sequence - returns dict for custom collate_fn."""
        video_id = self.video_list[idx]
        features = self._load_video(video_id)
        
        # Get available features and sequence length
        available_features = {k: v for k, v in features.items() if v is not None}
        if not available_features:
            raise ValueError(f"No features available for {video_id}")
        
        min_length = min(f.shape[0] for f in available_features.values())
        indices = slice(0, min_length, self.stride)
        if self.max_seq_length:
            indices = slice(0, min(min_length, self.max_seq_length), self.stride)
        
        # Process features
        output_features = {}
        feature_masks = {}
        
        for modality in self.feature_dirs.keys():
            if features[modality] is not None:
                output_features[modality] = torch.from_numpy(
                    features[modality][indices].astype(np.float32)
                )
                feature_masks[modality] = True
            else:
                # Zero placeholder
                ref_shape = next(iter(available_features.values()))[indices].shape
                feat_dim = self._get_feature_dim(modality)
                output_features[modality] = torch.zeros(
                    (ref_shape[0], feat_dim), dtype=torch.float32
                )
                feature_masks[modality] = False
        
        labels = self._get_labels(video_id, min_length)[indices]
        
        return {
            'video_id': video_id,
            'features': output_features,
            'feature_masks': feature_masks,
            'labels': torch.from_numpy(labels),
            'duration': self.video_to_annotation[video_id].get('duration', 0)
        }


def create_compatible_dataloader(
    feature_dirs: Dict[str, str],
    annotation_file: str,
    mode: str = 'frame',
    batch_size: int = 32,
    **kwargs
) -> DataLoader:
    """
    Create a DataLoader-compatible dataset.
    
    Args:
        feature_dirs: Dict mapping modality names to directories
        annotation_file: Path to annotations JSON
        mode: 'frame' for frame-level, 'sequence' for video-level
        batch_size: Batch size
        **kwargs: Additional dataset arguments
    
    Returns:
        DataLoader that works with standard PyTorch training loops
    """
    dataset = CompatibleVideoDataset(
        feature_dirs=feature_dirs,
        annotation_file=annotation_file,
        mode=mode,
        **kwargs
    )
    
    if mode == 'sequence':
        # Need custom collation for sequences
        def sequence_collate_fn(batch):
            max_len = max(sample['labels'].shape[0] for sample in batch)
            
            output = {
                'video_ids': [s['video_id'] for s in batch],
                'features': {},
                'feature_masks': {},
                'labels': [],
                'sequence_masks': []
            }
            
            modalities = list(batch[0]['features'].keys())
            for modality in modalities:
                output['features'][modality] = []
                output['feature_masks'][modality] = []
            
            for sample in batch:
                seq_len = sample['labels'].shape[0]
                
                # Pad features
                for modality in modalities:
                    feat = sample['features'][modality]
                    if seq_len < max_len:
                        pad_size = (max_len - seq_len, feat.shape[-1])
                        feat = torch.cat([feat, torch.zeros(pad_size)], dim=0)
                    output['features'][modality].append(feat)
                    output['feature_masks'][modality].append(sample['feature_masks'][modality])
                
                # Pad labels
                labels = sample['labels']
                if seq_len < max_len:
                    labels = torch.cat([labels, torch.zeros(max_len - seq_len)], dim=0)
                output['labels'].append(labels)
                
                # Sequence mask
                seq_mask = torch.ones(max_len)
                if seq_len < max_len:
                    seq_mask[seq_len:] = 0
                output['sequence_masks'].append(seq_mask)
            
            # Stack tensors
            for modality in modalities:
                output['features'][modality] = torch.stack(output['features'][modality])
                output['feature_masks'][modality] = torch.tensor(output['feature_masks'][modality])
            output['labels'] = torch.stack(output['labels'])
            output['sequence_masks'] = torch.stack(output['sequence_masks'])
            
            return output
        
        collate_fn = sequence_collate_fn
        # Use smaller batch size for sequences
        batch_size = min(batch_size, 8)
    else:
        collate_fn = None
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )


if __name__ == '__main__':
    # Test compatibility
    feature_dirs = {
        'audio': 'audio_pann_features',
        'visual': 'video_clip_features',
        'caption': 'caption_features'
    }
    
    # Test frame-level loading (standard DataLoader compatibility)
    print("Testing frame-level dataset...")
    frame_loader = create_compatible_dataloader(
        feature_dirs=feature_dirs,
        annotation_file='test.json',
        mode='frame',
        batch_size=64,
        stride=10
    )
    
    # This should work without "too many values to unpack" error
    for i, (x, y) in enumerate(frame_loader):
        print(f"Batch {i}: x.shape={x.shape}, y.shape={y.shape}")
        if i >= 2:
            break
    
    # Test sequence-level loading
    print("\nTesting sequence-level dataset...")
    seq_loader = create_compatible_dataloader(
        feature_dirs=feature_dirs,
        annotation_file='test.json',
        mode='sequence',
        batch_size=4,
        max_seq_length=500
    )
    
    for i, batch in enumerate(seq_loader):
        print(f"Sequence batch {i}:")
        for mod, feat in batch['features'].items():
            print(f"  {mod}: {feat.shape}")
        print(f"  labels: {batch['labels'].shape}")
        if i >= 1:
            break