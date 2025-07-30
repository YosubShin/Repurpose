#!/usr/bin/env python3
"""
Test script to validate the training setup before running full training.
"""

import os
import sys
import logging
from datetime import datetime

# Test imports
print("Testing imports...")
try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
except ImportError as e:
    print(f"✗ PyTorch import failed: {e}")
    sys.exit(1)

try:
    import pytorch_lightning as pl
    print(f"✓ PyTorch Lightning {pl.__version__}")
except ImportError as e:
    print(f"✗ PyTorch Lightning import failed: {e}")
    sys.exit(1)

try:
    import wandb
    print(f"✓ Wandb available")
except ImportError as e:
    print(f"⚠ Wandb not available: {e}")

try:
    import matplotlib.pyplot as plt
    print(f"✓ Matplotlib available")
except ImportError as e:
    print(f"✗ Matplotlib import failed: {e}")
    sys.exit(1)

try:
    from compatible_dataset import create_compatible_dataloader
    print(f"✓ Compatible dataset module")
except ImportError as e:
    print(f"✗ Compatible dataset import failed: {e}")
    sys.exit(1)

# Test data paths
print("\nTesting data paths...")
data_paths = {
    'audio': '/home/yosubs/koa_scratch/repurpose/data/audio_pann_features',
    'visual': '/home/yosubs/koa_scratch/repurpose/data/video_clip_features/',
    'caption': '/home/yosubs/koa_scratch/repurpose/data/caption_features',
    'annotation': '/home/yosubs/co/Repurpose/data/test.json'
}

for name, path in data_paths.items():
    if os.path.exists(path):
        print(f"✓ {name}: {path}")
    else:
        print(f"✗ {name}: {path} (not found)")

# Test data loader creation
print("\nTesting data loader creation...")
try:
    dataloader = create_compatible_dataloader(
        feature_dirs={
            'audio': data_paths['audio'],
            'visual': data_paths['visual'],
            'caption': data_paths['caption']
        },
        annotation_file=data_paths['annotation'],
        mode='sequence',
        batch_size=1,
        num_workers=0,
        min_modalities=3,
        max_seq_length=512
    )
    print("✓ Data loader created successfully")
    
    # Test batch loading
    sample_batch = next(iter(dataloader))
    print(f"✓ Sample batch loaded:")
    print(f"  Audio shape: {sample_batch['features']['audio'].shape}")
    print(f"  Visual shape: {sample_batch['features']['visual'].shape}")
    print(f"  Caption shape: {sample_batch['features']['caption'].shape}")
    print(f"  Labels shape: {sample_batch['labels'].shape}")
    
except Exception as e:
    print(f"✗ Data loader test failed: {e}")
    sys.exit(1)

# Test model creation
print("\nTesting model creation...")
try:
    from train_repurpose import RepurposeModel
    
    dim_audio = sample_batch['features']['audio'].shape[-1]
    dim_visual = sample_batch['features']['visual'].shape[-1]
    dim_caption = sample_batch['features']['caption'].shape[-1]
    
    model = RepurposeModel(
        dim_audio=dim_audio,
        dim_visual=dim_visual,
        dim_caption=dim_caption,
        d_model=128,
        n_head=4,
        n_layers=2,
        lr=1e-3
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created with {total_params:,} parameters")
    
    # Test forward pass
    with torch.no_grad():
        audio = sample_batch['features']['audio']
        visual = sample_batch['features']['visual']
        caption = sample_batch['features']['caption']
        
        logit_a, logit_v, logit_f = model(audio, visual, caption)
        print(f"✓ Forward pass successful:")
        print(f"  Audio logits: {logit_a.shape}")
        print(f"  Visual logits: {logit_v.shape}")
        print(f"  Fused logits: {logit_f.shape}")
    
except Exception as e:
    print(f"✗ Model test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test GPU availability
print("\nTesting compute environment...")
if torch.cuda.is_available():
    print(f"✓ CUDA available: {torch.cuda.device_count()} GPU(s)")
    print(f"  Current device: {torch.cuda.current_device()}")
    print(f"  Device name: {torch.cuda.get_device_name()}")
else:
    print("⚠ CUDA not available, will use CPU")

# Test directory creation
print("\nTesting directory setup...")
directories = ['checkpoints', 'logs', 'visualizations']
for directory in directories:
    try:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Directory {directory}/ ready")
    except Exception as e:
        print(f"✗ Failed to create {directory}/: {e}")

print("\n" + "="*50)
print("SETUP VALIDATION COMPLETE")
print("="*50)
print("✓ All tests passed! Ready to run training.")
print("\nTo run training:")
print("  python train_simple.py")
print("  # or #")
print("  ./run_training.sh")
print("="*50)