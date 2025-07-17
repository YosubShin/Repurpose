#!/usr/bin/env python3
"""
Launch script for Repurpose training with different multi-GPU configurations.

This script provides an easy interface to launch training with various
multi-GPU strategies and handles environment setup automatically.
"""

import os
import sys
import subprocess
import argparse
import yaml
import json
from pathlib import Path


def load_config(config_file):
    """Load YAML configuration file."""
    with open(config_file, "r") as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)
    return config


def detect_environment():
    """Detect if we're running in SLURM environment."""
    is_slurm = 'SLURM_JOB_ID' in os.environ
    return {
        'is_slurm': is_slurm,
        'job_id': os.environ.get('SLURM_JOB_ID'),
        'node_list': os.environ.get('SLURM_JOB_NODELIST'),
        'gpu_devices': os.environ.get('CUDA_VISIBLE_DEVICES')
    }


def run_gpu_analysis():
    """Run GPU detection and analysis."""
    print("🔍 Running GPU analysis...")
    try:
        result = subprocess.run([
            sys.executable, 'detect_gpu_setup.py',
            '--output-json', 'gpu_analysis.json'
        ], capture_output=True, text=True, check=True)

        print("GPU analysis completed successfully")

        # Load and return analysis results
        with open('gpu_analysis.json', 'r') as f:
            analysis = json.load(f)
        return analysis

    except subprocess.CalledProcessError as e:
        print(f"❌ GPU analysis failed: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return None


def test_multi_gpu_setup(config_path, strategy):
    """Test multi-GPU setup before training."""
    print(f"🧪 Testing multi-GPU setup (strategy: {strategy})...")
    try:
        result = subprocess.run([
            sys.executable, 'test_multi_gpu.py',
            '--config-path', config_path,
            '--strategy', strategy
        ], check=True)

        print("✅ Multi-GPU test passed!")
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Multi-GPU test failed with exit code {e.returncode}")
        return False


def launch_training_local(config_path, strategy, resume=None, log_level="INFO"):
    """Launch training locally (non-SLURM environment)."""
    print(f"🚀 Launching local training (strategy: {strategy})...")

    cmd = [sys.executable, 'main.py', '--config_path',
           config_path, '--log-level', log_level]

    if resume:
        cmd.extend(['--resume', resume])

    # Set strategy via environment variable if needed
    env = os.environ.copy()
    if strategy != 'auto':
        env['REPURPOSE_STRATEGY'] = strategy

    try:
        # Launch training process
        process = subprocess.Popen(cmd, env=env)
        return_code = process.wait()

        if return_code == 0:
            print("✅ Training completed successfully!")
        else:
            print(f"❌ Training failed with exit code {return_code}")

        return return_code

    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        process.terminate()
        return 130


def submit_slurm_job(config_path, strategy, resume=None, job_name=None):
    """Submit training job to SLURM."""
    print(f"📤 Submitting SLURM job (strategy: {strategy})...")

    script_path = "slurm_multi_gpu_training.sh"

    if not os.path.exists(script_path):
        print(f"❌ SLURM script not found: {script_path}")
        return None

    cmd = ['sbatch']

    if job_name:
        cmd.extend(['--job-name', job_name])

    cmd.extend([script_path, config_path, strategy])

    if resume:
        cmd.append(resume)

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True)

        # Extract job ID from output
        output = result.stdout.strip()
        job_id = None
        if "Submitted batch job" in output:
            job_id = output.split()[-1]

        print(f"✅ Job submitted successfully!")
        if job_id:
            print(f"Job ID: {job_id}")
            print(f"Monitor with: squeue -j {job_id}")
            print(f"Cancel with: scancel {job_id}")

        return job_id

    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to submit SLURM job: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Launch Repurpose training with multi-GPU support")
    parser.add_argument("--config", type=str, default="configs/Repurpose.yaml",
                        help="Path to configuration file")
    parser.add_argument("--strategy", type=str, choices=['auto', 'single', 'dp', 'ddp'],
                        default='auto', help="Multi-GPU strategy")
    parser.add_argument("--resume", type=str,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level")
    parser.add_argument("--slurm", action="store_true",
                        help="Submit job to SLURM (auto-detected if not specified)")
    parser.add_argument("--local", action="store_true",
                        help="Force local execution (even in SLURM environment)")
    parser.add_argument("--job-name", type=str, help="SLURM job name")
    parser.add_argument("--test-only", action="store_true",
                        help="Only run tests, don't start training")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Only run GPU analysis")
    args = parser.parse_args()

    print("=" * 60)
    print("🎯 REPURPOSE MULTI-GPU TRAINING LAUNCHER")
    print("=" * 60)

    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"❌ Configuration file not found: {args.config}")
        return 1

    # Detect environment
    env_info = detect_environment()
    print(f"Environment: {'SLURM' if env_info['is_slurm'] else 'Local'}")

    if env_info['is_slurm']:
        print(f"SLURM Job ID: {env_info['job_id']}")
        print(f"Node: {env_info['node_list']}")
        print(f"GPUs: {env_info['gpu_devices']}")

    # Run GPU analysis
    analysis = run_gpu_analysis()
    if analysis is None:
        print("⚠️  GPU analysis failed, but continuing...")
        # Use fallback strategy selection
        if args.strategy == 'auto':
            try:
                import torch
                gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
                if gpu_count >= 2:
                    args.strategy = 'dp'  # Default to DataParallel for multi-GPU
                    print(f"🔄 Fallback: Using DataParallel strategy (detected {gpu_count} GPUs)")
                else:
                    args.strategy = 'single'
                    print(f"🔄 Fallback: Using single GPU strategy")
            except:
                args.strategy = 'single'
                print("🔄 Fallback: Using single GPU strategy")
    else:
        recommended_strategy = analysis['recommendations']['strategy']
        print(f"🎯 Recommended strategy: {recommended_strategy}")

        if args.strategy == 'auto':
            args.strategy = recommended_strategy
            print(f"Using recommended strategy: {args.strategy}")

    if args.analyze_only:
        if analysis is not None:
            print("✅ GPU analysis completed. Exiting.")
            return 0
        else:
            print("❌ GPU analysis failed. Exiting.")
            return 1

    # Test multi-GPU setup
    test_success = test_multi_gpu_setup(args.config, args.strategy)
    if not test_success:
        print("❌ Multi-GPU tests failed. Please check your setup.")
        if args.strategy in ['dp', 'ddp']:
            print("🔄 Falling back to single GPU strategy due to test failures")
            args.strategy = 'single'
        else:
            print("⚠️  Continuing with failed tests - training may encounter issues")
            # Don't return 1 here, let the user decide to continue

    if args.test_only:
        print("✅ All tests passed. Exiting.")
        return 0

    # Load config to check training parameters
    try:
        config = load_config(args.config)
        print(f"📊 Training configuration:")
        print(f"  Epochs: {config['train']['epochs']}")
        print(f"  Batch size: {config['train']['batch_size']}")
        print(f"  Learning rate: {config['train']['lr']}")
        print(f"  Strategy: {args.strategy}")
    except Exception as e:
        print(f"⚠️  Could not load config details: {e}")

    # Determine execution method
    use_slurm = args.slurm or (env_info['is_slurm'] and not args.local)

    if use_slurm:
        if env_info['is_slurm']:
            print("🏃 Running in SLURM environment, executing training directly...")
            # We're already in a SLURM job, run training directly
            return launch_training_local(args.config, args.strategy, args.resume, args.log_level)
        else:
            print("📤 Submitting to SLURM...")
            job_id = submit_slurm_job(
                args.config, args.strategy, args.resume, args.job_name)
            return 0 if job_id else 1
    else:
        print("🏠 Running locally...")
        return launch_training_local(args.config, args.strategy, args.resume, args.log_level)


if __name__ == "__main__":
    sys.exit(main())
