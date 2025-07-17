#!/usr/bin/env python3
"""
Test script for multi-GPU training setup validation.

This script tests the multi-GPU infrastructure without running full training.
"""

import os
import sys
import torch
import argparse
import yaml
import logging
from utils.distributed import MultiGPUStrategy, is_main_process, get_rank, get_world_size


def load_config(config_file):
    with open(config_file, "r") as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)
    return config


def test_model_initialization():
    """Test model initialization and wrapping."""
    try:
        from models.MMCTransformer import MMCTransformer

        # Create a simple model config for testing
        model_config = {
            'vis_dim': 512,
            'aud_dim': 2048,
            'text_dim': 384,
            'd_model': 512,
            'self_num_layers': 2,  # Reduced for testing
            'text_num_layers': 2,
            'cross_num_layers': 2,
            'num_heads': 8,
        }

        model = MMCTransformer(**model_config)
        return model, True
    except Exception as e:
        logging.error(f"Model initialization failed: {e}")
        return None, False


def test_data_loading():
    """Test dataset and data loader creation."""
    try:
        from dataset.RepurposeClip import RepurposeClip, collate_fn

        # Check if data paths exist
        data_paths = {
            'label_path': 'data/train.json',
            'video_path': '/home/yosubs/koa_scratch/repurpose/data/video_clip_features',
            'audio_path': '/home/yosubs/koa_scratch/repurpose/data/audio_pann_features',
            'text_path': '/home/yosubs/koa_scratch/repurpose/data/caption_features',
        }

        # Check if paths exist
        missing_paths = []
        for name, path in data_paths.items():
            if not os.path.exists(path):
                missing_paths.append(f"{name}: {path}")

        if missing_paths:
            logging.warning(f"Missing data paths (expected on local machine):")
            for path in missing_paths:
                logging.warning(f"  {path}")
            return None, False

        # Try to create dataset
        dataset = RepurposeClip(**data_paths)
        logging.info(
            f"Dataset created successfully with {len(dataset)} samples")
        return dataset, True

    except Exception as e:
        logging.error(f"Data loading test failed: {e}")
        return None, False


def test_multi_gpu_functionality(multi_gpu, model=None):
    """Test multi-GPU specific functionality."""
    results = {
        'strategy': multi_gpu.strategy,
        'world_size': multi_gpu.world_size,
        'rank': multi_gpu.rank,
        'device': str(multi_gpu.device),
        'model_wrapping': False,
        'tensor_operations': False,
        'loss_reduction': False
    }

    try:
        # Test model wrapping
        if model is not None:
            wrapped_model = multi_gpu.wrap_model(model)
            results['model_wrapping'] = True
            logging.info(
                f"Model wrapped successfully for {multi_gpu.strategy}")

        # Test tensor operations
        test_tensor = torch.randn(4, 8, device=multi_gpu.device)
        result_tensor = test_tensor * 2
        results['tensor_operations'] = True
        logging.info(f"Tensor operations successful on {multi_gpu.device}")

        # Test loss reduction
        dummy_loss = torch.tensor(1.5, device=multi_gpu.device)
        reduced_loss = multi_gpu.reduce_tensor(dummy_loss)
        results['loss_reduction'] = True
        logging.info(
            f"Loss reduction successful: {dummy_loss.item()} -> {reduced_loss.item()}")

        # Test barrier
        multi_gpu.barrier()
        logging.info("Barrier synchronization successful")

    except Exception as e:
        logging.error(f"Multi-GPU functionality test failed: {e}")

    return results


def test_dataloader_creation(multi_gpu, dataset=None):
    """Test DataLoader creation with multi-GPU support."""
    try:
        if dataset is None:
            # Create a dummy dataset for testing
            class DummyDataset(torch.utils.data.Dataset):
                def __len__(self):
                    return 10

                def __getitem__(self, idx):
                    return {'data': torch.randn(4), 'target': idx}

            dataset = DummyDataset()

        # Test DataLoader creation
        dataloader = multi_gpu.create_dataloader(
            dataset,
            batch_size=2,
            shuffle=True,
            num_workers=0  # Use 0 workers for testing
        )

        # Test iteration
        for i, batch in enumerate(dataloader):
            if i >= 2:  # Test only first few batches
                break

        logging.info(f"DataLoader creation and iteration successful")
        return True

    except Exception as e:
        logging.error(f"DataLoader test failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test multi-GPU training setup")
    parser.add_argument("--config-path", type=str, default="configs/Repurpose.yaml",
                        help="Path to configuration file")
    parser.add_argument("--strategy", type=str, choices=['auto', 'single', 'dp', 'ddp'],
                        default='auto', help="Multi-GPU strategy to test")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    args = parser.parse_args()

    # Set up logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logger = logging.getLogger(__name__)

    if is_main_process():
        logger.info("=" * 60)
        logger.info("MULTI-GPU TRAINING SETUP TEST")
        logger.info("=" * 60)

    # Load configuration
    try:
        cfg = load_config(args.config_path)
        logger.info(f"Configuration loaded from {args.config_path}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1

    # Override strategy if specified
    if args.strategy != 'auto':
        if 'distributed' not in cfg:
            cfg['distributed'] = {}
        cfg['distributed']['strategy'] = args.strategy

    # Initialize multi-GPU strategy
    try:
        gpu_strategy = cfg.get('distributed', {}).get('strategy', 'auto')
        multi_gpu = MultiGPUStrategy(strategy=gpu_strategy)

        # Setup distributed training
        if not multi_gpu.setup():
            logger.error("Failed to setup multi-GPU training")
            return 1

        if is_main_process():
            multi_gpu.print_setup_info()

    except Exception as e:
        logger.error(f"Multi-GPU initialization failed: {e}")
        return 1

    # Run tests
    test_results = {
        'multi_gpu_setup': True,
        'model_init': False,
        'data_loading': False,
        'multi_gpu_ops': False,
        'dataloader': False
    }

    # Test model initialization
    if is_main_process():
        logger.info("\n" + "=" * 40)
        logger.info("Testing model initialization...")

    model, model_success = test_model_initialization()
    test_results['model_init'] = model_success

    # Test multi-GPU functionality
    if is_main_process():
        logger.info("\n" + "=" * 40)
        logger.info("Testing multi-GPU functionality...")

    gpu_results = test_multi_gpu_functionality(multi_gpu, model)
    test_results['multi_gpu_ops'] = gpu_results.get('model_wrapping', False)

    # Test data loading
    if is_main_process():
        logger.info("\n" + "=" * 40)
        logger.info("Testing data loading...")

    dataset, data_success = test_data_loading()
    test_results['data_loading'] = data_success

    # Test DataLoader creation
    if is_main_process():
        logger.info("\n" + "=" * 40)
        logger.info("Testing DataLoader creation...")

    dataloader_success = test_dataloader_creation(multi_gpu, dataset)
    test_results['dataloader'] = dataloader_success

    # Print summary
    if is_main_process():
        logger.info("\n" + "=" * 60)
        logger.info("TEST SUMMARY")
        logger.info("=" * 60)

        for test_name, success in test_results.items():
            status = "✓ PASS" if success else "✗ FAIL"
            logger.info(f"{test_name:20}: {status}")

        all_passed = all(test_results.values())
        logger.info("\n" + "=" * 60)

        if all_passed:
            logger.info(
                "🎉 ALL TESTS PASSED! Multi-GPU setup is ready for training.")
            logger.info(f"Strategy: {multi_gpu.strategy.upper()}")
            logger.info(f"World Size: {multi_gpu.world_size}")
            logger.info(f"Device: {multi_gpu.device}")
        else:
            failed_tests = [name for name,
                            success in test_results.items() if not success]
            logger.warning(f"⚠️  Some tests failed: {', '.join(failed_tests)}")
            logger.warning("Please check the error messages above.")

        logger.info("=" * 60)

    # Cleanup
    multi_gpu.cleanup()

    return 0 if all(test_results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
