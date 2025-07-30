#!/usr/bin/env python3
"""
Simple training script for RepurposeModel - same configuration as notebook but with logging and wandb.
This script has hardcoded paths to match the notebook configuration.
"""

import os
import gc
import time
import logging
from datetime import datetime

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb

# Import our training script components
from train_repurpose import RepurposeModel, MemoryClearCallback, visualize_predictions, setup_logging
from compatible_dataset import create_compatible_dataloader


def main():
    """Main function with hardcoded notebook configuration."""
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"train_simple_{timestamp}.log"
    logger = setup_logging("INFO", log_file)
    
    logger.info("=" * 60)
    logger.info("Starting RepurposeModel training (simple configuration)")
    logger.info("=" * 60)
    
    # Configuration matching the notebook
    CONFIG = {
        # Data paths (from notebook)
        'audio_dir': '/home/yosubs/koa_scratch/repurpose/data/audio_pann_features',
        'visual_dir': '/home/yosubs/koa_scratch/repurpose/data/video_clip_features/',
        'caption_dir': '/home/yosubs/koa_scratch/repurpose/data/caption_features',
        'train_annotation': '/home/yosubs/co/Repurpose/data/test.json',
        
        # Model hyperparameters (from notebook)
        'd_model': 128,
        'n_head': 4,
        'n_layers': 2,
        'batch_size': 1,
        'epochs': 10,
        'learning_rate': 1e-3,
        
        # Loss weights (from notebook)
        'lambda1': 0.1,  # uni-modal weight  
        'lambda2': 0.3,  # multi-modal weight
        'lambda3': 0.1,  # KL divergence weight
        
        # Training settings
        'num_workers': 0,  # Memory-friendly
        'log_interval': 5,  # Log every 5 steps
        'max_seq_len': None,  # Use full video sequences for higher memory usage
        
        # Wandb settings
        'use_wandb': True,
        'wandb_project': 'repurpose-simple',
        
        # Visualization
        'create_visualizations': True,
        'num_viz_samples': 3
    }
    
    logger.info("Configuration:")
    for key, value in CONFIG.items():
        logger.info(f"  {key}: {value}")
    
    # Initialize wandb
    wandb_logger = None
    if CONFIG['use_wandb']:
        try:
            wandb_logger = WandbLogger(
                project=CONFIG['wandb_project'],
                name=f"repurpose_simple_{timestamp}",
                config=CONFIG
            )
            logger.info(f"✓ Initialized wandb project: {CONFIG['wandb_project']}")
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")
    
    # Create directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("visualizations", exist_ok=True)
    
    # Create data loader
    logger.info("Creating data loader...")
    start_time = time.time()
    
    try:
        dataloader = create_compatible_dataloader(
            feature_dirs={
                'audio': CONFIG['audio_dir'],
                'visual': CONFIG['visual_dir'], 
                'caption': CONFIG['caption_dir']
            },
            annotation_file=CONFIG['train_annotation'],
            mode='sequence',
            batch_size=CONFIG['batch_size'],
            num_workers=CONFIG['num_workers'],
            min_modalities=3,
            max_seq_length=CONFIG['max_seq_len']
        )
        
        logger.info(f"✓ Data loader created in {time.time() - start_time:.2f}s")
    except Exception as e:
        logger.error(f"Failed to create data loader: {e}")
        return
    
    # Get feature dimensions
    logger.info("Determining feature dimensions...")
    try:
        sample_batch = next(iter(dataloader))
        dim_audio = sample_batch['features']['audio'].shape[-1]
        dim_visual = sample_batch['features']['visual'].shape[-1] 
        dim_caption = sample_batch['features']['caption'].shape[-1]
        
        logger.info(f"✓ Feature dimensions:")
        logger.info(f"  Audio: {dim_audio}")
        logger.info(f"  Visual: {dim_visual}")
        logger.info(f"  Caption: {dim_caption}")
        
        # Log sample batch info
        logger.info(f"✓ Sample batch info:")
        logger.info(f"  Audio shape: {sample_batch['features']['audio'].shape}")
        logger.info(f"  Visual shape: {sample_batch['features']['visual'].shape}")
        logger.info(f"  Caption shape: {sample_batch['features']['caption'].shape}")
        logger.info(f"  Labels shape: {sample_batch['labels'].shape}")
        logger.info(f"  Sequence masks shape: {sample_batch['sequence_masks'].shape}")
        
    except Exception as e:
        logger.error(f"Failed to get feature dimensions: {e}")
        return
    
    # Create model
    logger.info("Creating model...")
    try:
        model = RepurposeModel(
            dim_audio=dim_audio,
            dim_visual=dim_visual,
            dim_caption=dim_caption,
            d_model=CONFIG['d_model'],
            n_head=CONFIG['n_head'],
            n_layers=CONFIG['n_layers'],
            lr=CONFIG['learning_rate'],
            lambda1=CONFIG['lambda1'],
            lambda2=CONFIG['lambda2'],
            lambda3=CONFIG['lambda3'],
            log_interval=CONFIG['log_interval']
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"✓ Model created:")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        logger.info(f"  Model size: ~{total_params * 4 / 1024**2:.1f} MB")
        
    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        return
    
    # Setup callbacks
    callbacks = []
    
    # Memory management callback
    callbacks.append(MemoryClearCallback(clear_every_n_epochs=1))
    
    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename=f'repurpose-simple-{timestamp}-{{epoch:02d}}-{{loss_total:.4f}}',
        monitor='loss_total',
        mode='min',
        save_top_k=3,
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # Create trainer
    logger.info("Setting up trainer...")
    trainer = Trainer(
        max_epochs=CONFIG['epochs'],
        accelerator='auto',
        devices=1 if torch.cuda.is_available() else None,
        precision='32',
        gradient_clip_val=1.0,
        callbacks=callbacks,
        logger=wandb_logger,
        enable_checkpointing=True,
        log_every_n_steps=CONFIG['log_interval'],
        enable_progress_bar=True,
        num_sanity_val_steps=0,  # Skip validation sanity check
    )
    
    # Log training info
    logger.info("Training configuration:")
    logger.info(f"  Device: {trainer.strategy.root_device}")
    logger.info(f"  Precision: {trainer.precision}")
    logger.info(f"  Max epochs: {CONFIG['epochs']}")
    logger.info(f"  Log interval: {CONFIG['log_interval']}")
    logger.info(f"  Checkpoints will be saved to: checkpoints/")
    
    # Start training
    logger.info("=" * 60)
    logger.info("STARTING TRAINING")
    logger.info("=" * 60)
    
    training_start_time = time.time()
    
    try:
        trainer.fit(model, dataloader)
        
        training_time = time.time() - training_start_time
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Total training time: {training_time/60:.2f} minutes")
        logger.info(f"Average time per epoch: {training_time/CONFIG['epochs']:.2f} seconds")
        
        # Get final metrics
        if hasattr(trainer, 'callback_metrics'):
            final_metrics = trainer.callback_metrics
            logger.info("Final metrics:")
            for key, value in final_metrics.items():
                if torch.is_tensor(value):
                    logger.info(f"  {key}: {value.item():.4f}")
                else:
                    logger.info(f"  {key}: {value}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.error("Exception details:", exc_info=True)
        return
    
    # Create visualizations
    if CONFIG['create_visualizations']:
        logger.info("=" * 60)
        logger.info("CREATING VISUALIZATIONS")
        logger.info("=" * 60)
        
        try:
            device = next(model.parameters()).device
            viz_paths = visualize_predictions(
                model=model,
                dataloader=dataloader,
                save_dir="visualizations",
                num_samples=CONFIG['num_viz_samples'],
                device=device
            )
            
            logger.info(f"✓ Created {len(viz_paths)} visualizations:")
            for path in viz_paths:
                logger.info(f"  {path}")
        
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
    
    # Final cleanup
    logger.info("Performing final cleanup...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Close wandb
    if CONFIG['use_wandb']:
        try:
            wandb.finish()
            logger.info("✓ Wandb run finished")
        except:
            pass
    
    logger.info("=" * 60)
    logger.info("SCRIPT COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)
    logger.info(f"Log file: {log_file}")
    logger.info("Check the following directories:")
    logger.info("  - checkpoints/ for saved models")
    logger.info("  - visualizations/ for prediction plots")
    if CONFIG['use_wandb']:
        logger.info(f"  - wandb dashboard for experiment tracking")


if __name__ == "__main__":
    main()