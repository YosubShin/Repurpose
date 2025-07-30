#!/usr/bin/env python3
"""Test the fixed training on 4 samples."""

import subprocess
import sys
import os

def test_fixed_training():
    """Test training with the fixes applied."""
    
    print("Testing fixed training configuration on 4 samples...")
    
    # Run training with debug dataset
    cmd = [
        sys.executable, "train_repurpose.py",
        "--audio_dir", "audio_pann_features",
        "--visual_dir", "video_clip_features", 
        "--caption_dir", "caption_features",
        "--train_annotation", "debug_4samples.json",
        "--val_annotation", "debug_4samples.json",  # Same for overfitting test
        "--batch_size", "4",
        "--epochs", "20",
        "--learning_rate", "1e-3",
        "--d_model", "64",  # Smaller for faster training
        "--n_head", "4",
        "--n_layers", "1",
        "--lambda1", "0.0",  # Only multi-modal loss
        "--lambda2", "1.0", 
        "--lambda3", "0.0",
        "--log_interval", "2",
        "--checkpoint_dir", "debug_checkpoints",
        "--num_workers", "0",
        "--create_visualizations",
        "--num_viz_samples", "4",
        "--log_level", "INFO",
        "--gradient_clip", "0.1",  # Key fix for stability
        "--enable_checkpointing"
    ]
    
    print("Running command:")
    print(" ".join(cmd))
    print("\n" + "="*60)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("\n" + "="*60)
        print("✓ Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Training failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n✗ Training interrupted by user")
        return False

if __name__ == '__main__':
    success = test_fixed_training()
    sys.exit(0 if success else 1)