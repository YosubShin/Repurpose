from dataset.RepurposeClip import RepurposeClip, RepurposeClipTest
from dataset.RepurposeClip import collate_fn, collate_fn_test
from models.MMCTransformer import MMCTransformer
from utils.metrics import *
from utils.distributed import MultiGPUStrategy, is_main_process, get_rank, get_world_size
from utils.debug_visualizer import ValidationDebugger
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from tqdm import tqdm
import numpy as np
import random
import time
import yaml
import os
import wandb
import logging


def load_config(config_file):
    with open(config_file, "r") as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)
    return config


def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_debug_samples(dataset, num_samples=5, seed=42):
    """Get a fixed set of samples for debugging visualization"""
    # Use a fixed seed to always get the same samples
    rng = np.random.RandomState(seed)
    total_samples = len(dataset)
    sample_indices = rng.choice(total_samples, size=min(
        num_samples, total_samples), replace=False)

    samples = []
    for idx in sample_indices:
        sample = dataset[idx]
        samples.append({
            'idx': idx,
            'video_id': sample['video_id'],
            'data': sample
        })
    return samples


def main(args):
    # Set up logger for this function
    logger = logging.getLogger(__name__)

    logger.debug(f"Starting main() function - PID: {os.getpid()}")
    logger.debug(f"Arguments: {args}")

    # load config
    logger.debug(f"Loading config from: {args.config_path}")
    cfg = load_config(args.config_path)
    logger.debug("Config loaded successfully")

    # Initialize multi-GPU strategy
    logger.debug("Initializing multi-GPU strategy")
    distributed_cfg = cfg.get('distributed', {})
    logger.debug(f"Distributed config: {distributed_cfg}")
    multi_gpu = MultiGPUStrategy(
        strategy=distributed_cfg.get('strategy', 'auto'),
        backend=distributed_cfg.get('backend', 'nccl'),
        timeout=distributed_cfg.get('timeout', 1800),
        find_unused_parameters=distributed_cfg.get(
            'find_unused_parameters', False)
    )
    logger.debug(
        f"MultiGPUStrategy created - strategy: {multi_gpu.strategy}, rank: {multi_gpu.rank}, world_size: {multi_gpu.world_size}")

    # Setup distributed training if needed
    logger.debug("Setting up distributed training...")
    if not multi_gpu.setup():
        raise RuntimeError("Failed to setup multi-GPU training")
    logger.debug("Distributed training setup completed")

    # Print setup info (only from main process)
    if is_main_process():
        logger.debug("Main process - printing setup info")
        multi_gpu.print_setup_info()
    else:
        logger.debug(f"Worker process - rank {multi_gpu.rank}")

    logger.debug("About to call seed_everything")

    seed_everything(cfg['train']['seed'])
    logger.debug(f"Seed set to {cfg['train']['seed']}")

    checkpoint_path = time.strftime('saved_model/ckpt_%Y%m%d_%H%M%S')
    logger.debug(f"Checkpoint path: {checkpoint_path}")
    if is_main_process():
        logger.debug("Main process - creating checkpoint directory")
        os.makedirs(checkpoint_path, exist_ok=True)
        with open(os.path.join(checkpoint_path, 'config.yaml'), 'w') as f:
            yaml.dump(cfg, f)
        print('The checkpoint path is %s' % checkpoint_path)
    else:
        logger.debug(
            "Worker process - waiting for main process to create checkpoint path")

    # Synchronize all processes
    logger.debug(f"About to call multi_gpu.barrier() - rank {multi_gpu.rank}")
    multi_gpu.barrier()
    logger.debug(f"Barrier completed - rank {multi_gpu.rank}")

    logger.debug(f"Creating train dataset - rank {multi_gpu.rank}")
    train_dataset = RepurposeClip(**cfg['train_dataset'])
    logger.debug(
        f"Train dataset created with {len(train_dataset)} samples - rank {multi_gpu.rank}")

    logger.debug(f"Creating validation dataset - rank {multi_gpu.rank}")
    val_dataset = RepurposeClipTest(**cfg['val_dataset'])
    logger.debug(
        f"Validation dataset created with {len(val_dataset)} samples - rank {multi_gpu.rank}")

    logger.debug(f"Creating test dataset - rank {multi_gpu.rank}")
    test_dataset = RepurposeClipTest(**cfg['test_dataset'])
    logger.debug(
        f"Test dataset created with {len(test_dataset)} samples - rank {multi_gpu.rank}")

    # Get fixed training samples for debugging visualization
    logger.debug(f"Getting debug samples - rank {multi_gpu.rank}")
    debug_train_samples = get_debug_samples(
        train_dataset, num_samples=5, seed=42)
    if is_main_process():
        print(
            f"Selected training samples for debug visualization: {[s['video_id'] for s in debug_train_samples]}")
    logger.debug(f"Debug samples obtained - rank {multi_gpu.rank}")

    logger.debug(f"Creating model - rank {multi_gpu.rank}")
    model = MMCTransformer(**cfg['model'])
    logger.debug(f"Model created - rank {multi_gpu.rank}")

    logger.debug(
        f"Wrapping model with multi-GPU strategy - rank {multi_gpu.rank}")
    model = multi_gpu.wrap_model(model)
    logger.debug(f"Model wrapped - rank {multi_gpu.rank}")

    learning_rate = cfg['train']['lr']
    weight_decay = cfg['train']['weight_decay']
    batch_size = cfg['train']['batch_size']
    num_epochs = cfg['train']['epochs']
    warmup_epochs = cfg['train']['warmup_epochs']
    logger.debug(
        f"Training params - lr: {learning_rate}, batch_size: {batch_size}, epochs: {num_epochs} - rank {multi_gpu.rank}")

    # Create data loaders with multi-GPU support
    logger.debug(f"Creating train data loader - rank {multi_gpu.rank}")
    train_data_loader = multi_gpu.create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=min(24, 4)  # Reduce workers for multi-GPU
    )
    logger.debug(
        f"Train data loader created with {len(train_data_loader)} batches - rank {multi_gpu.rank}")

    logger.debug(f"Creating validation data loader - rank {multi_gpu.rank}")
    val_data_loader = multi_gpu.create_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_test,
        num_workers=min(24, 4)
    )
    logger.debug(
        f"Validation data loader created with {len(val_data_loader)} batches - rank {multi_gpu.rank}")

    logger.debug(f"Creating test data loader - rank {multi_gpu.rank}")
    test_data_loader = multi_gpu.create_dataloader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_test,
        num_workers=min(24, 4)
    )
    logger.debug(
        f"Test data loader created with {len(test_data_loader)} batches - rank {multi_gpu.rank}")

    logger.debug(f"Creating optimizer - rank {multi_gpu.rank}")
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    logger.debug(f"Optimizer created - rank {multi_gpu.rank}")

    num_iters = len(train_data_loader)
    warmup_steps = warmup_epochs * num_iters
    total_steps = num_epochs * num_iters
    logger.debug(
        f"Scheduler params - num_iters: {num_iters}, warmup_steps: {warmup_steps}, total_steps: {total_steps} - rank {multi_gpu.rank}")

    def warmup_lambda(global_step):
        return (global_step + 1) / warmup_steps if (global_step + 1) <= warmup_steps else 1

    logger.debug(f"Creating schedulers - rank {multi_gpu.rank}")
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
    logger.debug(f"Schedulers created - rank {multi_gpu.rank}")

    start_epoch = 0
    global_step = 0
    logger.debug(
        f"Initial values - start_epoch: {start_epoch}, global_step: {global_step} - rank {multi_gpu.rank}")

    if args.resume:
        logger.debug(
            f"Loading checkpoint from {args.resume} - rank {multi_gpu.rank}")
        checkpoint = torch.load(args.resume, map_location=multi_gpu.device)
        # Handle DDP module loading
        if hasattr(model, 'module'):
            model.module.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        warmup_scheduler.load_state_dict(checkpoint['warmup_scheduler'])
        cosine_scheduler.load_state_dict(checkpoint['cosine_scheduler'])
        start_epoch = checkpoint['epoch']
        global_step = start_epoch * num_iters
        logger.debug(
            f"Checkpoint loaded - start_epoch: {start_epoch}, global_step: {global_step} - rank {multi_gpu.rank}")
    else:
        logger.debug(f"No checkpoint to resume from - rank {multi_gpu.rank}")

    # Initialize wandb (only on main process)
    if is_main_process():
        logger.debug("Initializing wandb - main process")
        effective_batch_size = multi_gpu.get_effective_batch_size(batch_size)
        run_name = f"mmc_{multi_gpu.strategy}_{multi_gpu.world_size}gpu_{time.strftime('%Y%m%d_%H%M%S')}"
        logger.debug(f"WandB run name: {run_name}")
        wandb.init(
            project="repurpose-video",
            name=run_name,
            config={
                "architecture": "MMCTransformer",
                "dataset": "Repurpose",
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "effective_batch_size": effective_batch_size,
                "epochs": num_epochs,
                "warmup_epochs": warmup_epochs,
                "weight_decay": weight_decay,
                "checkpoint_path": checkpoint_path,
                "multi_gpu_strategy": multi_gpu.strategy,
                "world_size": multi_gpu.world_size,
                "rank": multi_gpu.rank,
                **cfg  # Include all config parameters
            },
            dir=checkpoint_path  # Save wandb files in checkpoint directory
        )
        logger.debug("WandB initialized")

        # Watch the model to log gradients and parameters
        # For DDP, watch the underlying module
        model_to_watch = model.module if hasattr(model, 'module') else model
        wandb.watch(model_to_watch, log="all", log_freq=100)
        logger.debug("WandB watching model")
    else:
        logger.debug(
            f"Worker process - skipping wandb init - rank {multi_gpu.rank}")

    best_tIoU = 0
    best_epoch = 0
    logger.debug(
        f"About to start training loop - epochs {start_epoch} to {num_epochs} - rank {multi_gpu.rank}")

    for epoch in range(start_epoch, num_epochs):
        logger.debug(
            f"Starting epoch {epoch+1}/{num_epochs} - rank {multi_gpu.rank}")
        # Set epoch for distributed sampler to ensure proper shuffling
        if hasattr(train_data_loader.sampler, 'set_epoch'):
            logger.debug(
                f"Setting epoch {epoch} for distributed sampler - rank {multi_gpu.rank}")
            train_data_loader.sampler.set_epoch(epoch)

        logger.debug(f"Setting model to train mode - rank {multi_gpu.rank}")
        model.train()
        total_loss = 0
        total_cls_loss = 0
        start_time = time.time()
        logger.debug(
            f"Starting training loop for epoch {epoch+1} - rank {multi_gpu.rank}")
        logger.debug(
            f"Training data loader has {len(train_data_loader)} batches - rank {multi_gpu.rank}")

        for i, batch in enumerate(train_data_loader):
            if i == 0:
                logger.debug(
                    f"Processing first batch of epoch {epoch+1} - rank {multi_gpu.rank}")
            elif i % 10 == 0:
                logger.debug(
                    f"Processing batch {i+1}/{len(train_data_loader)} of epoch {epoch+1} - rank {multi_gpu.rank}")
            # Move tensors to appropriate device
            batch['visual_feats'] = batch['visual_feats'].to(
                multi_gpu.device, non_blocking=True)
            batch['audio_feats'] = batch['audio_feats'].to(
                multi_gpu.device, non_blocking=True)
            batch['text_feats'] = batch['text_feats'].to(
                multi_gpu.device, non_blocking=True)
            batch['masks'] = batch['masks'].to(
                multi_gpu.device, non_blocking=True)
            batch['labels'] = batch['labels'].to(
                multi_gpu.device, non_blocking=True)
            batch['segments'] = batch['segments'].to(
                multi_gpu.device, non_blocking=True)

            if i == 0:
                logger.debug(
                    f"About to call model forward pass - batch {i+1}, epoch {epoch+1} - rank {multi_gpu.rank}")
            output = model(batch)
            if i == 0:
                logger.debug(
                    f"Model forward pass completed - batch {i+1}, epoch {epoch+1} - rank {multi_gpu.rank}")
            # Access losses method through module for DDP-wrapped models
            if hasattr(model, 'module'):
                losses = model.module.losses(*output)
            else:
                losses = model.losses(*output)
            if i == 0:
                logger.debug(
                    f"Loss computation completed - batch {i+1}, epoch {epoch+1} - rank {multi_gpu.rank}")

            final_loss = losses['cls_loss'] / batch_size
            if i == 0:
                logger.debug(
                    f"About to start backward pass - batch {i+1}, epoch {epoch+1} - rank {multi_gpu.rank}")
            optimizer.zero_grad()
            if i == 0:
                logger.debug(
                    f"Zero grad completed - batch {i+1}, epoch {epoch+1} - rank {multi_gpu.rank}")
            final_loss.backward()
            if i == 0:
                logger.debug(
                    f"Backward pass completed - batch {i+1}, epoch {epoch+1} - rank {multi_gpu.rank}")
            optimizer.step()
            if i == 0:
                logger.debug(
                    f"Optimizer step completed - batch {i+1}, epoch {epoch+1} - rank {multi_gpu.rank}")

            # Reduce losses across GPUs for accurate logging
            if i == 0:
                logger.debug(
                    f"About to reduce losses across GPUs - batch {i+1}, epoch {epoch+1} - rank {multi_gpu.rank}")
            cls_loss_tensor = multi_gpu.reduce_tensor(
                losses["cls_loss"] / batch_size)
            final_loss_tensor = multi_gpu.reduce_tensor(final_loss)
            if i == 0:
                logger.debug(
                    f"Loss reduction completed - batch {i+1}, epoch {epoch+1} - rank {multi_gpu.rank}")

            cls_loss = cls_loss_tensor.item()
            batch_loss = final_loss_tensor.item()
            total_cls_loss += cls_loss
            total_loss += batch_loss

            # Log batch-level metrics (only on main process)
            if is_main_process():
                global_step_iter = epoch * len(train_data_loader) + i
                wandb.log({
                    'batch/cls_loss': cls_loss,
                    'batch/total_loss': batch_loss,
                    'batch/learning_rate': optimizer.param_groups[0]['lr']
                }, step=global_step_iter)
                if i == 0:
                    logger.debug(
                        f"Batch-level metrics logged to wandb - batch {i+1}, epoch {epoch+1}")

            if i == 0:
                logger.debug(
                    f"About to update learning rate scheduler - batch {i+1}, epoch {epoch+1} - rank {multi_gpu.rank}")
            if global_step < warmup_steps:
                warmup_scheduler.step()
            else:
                cosine_scheduler.step()
            global_step += 1
            if i == 0:
                logger.debug(
                    f"Learning rate scheduler updated, global_step: {global_step} - batch {i+1}, epoch {epoch+1} - rank {multi_gpu.rank}")

            # Intra-epoch validation (using validation dataset, not test dataset)
            intra_epoch_eval_freq = cfg['train'].get(
                'intra_epoch_eval_freq', 50)
            if (i + 1) % intra_epoch_eval_freq == 0:
                logger.debug(
                    f"Starting intra-epoch validation - batch {i+1}, epoch {epoch+1} - rank {multi_gpu.rank}")
                model.eval()
                with torch.no_grad():
                    # Sample a few batches from validation set for quick eval
                    val_cls_losses = []
                    val_total_losses = []

                    # Use iterator to get a few validation batches
                    val_iter = iter(val_data_loader)
                    # Evaluate on 10 batches max
                    num_val_batches = min(10, len(val_data_loader))

                    for _ in range(num_val_batches):
                        try:
                            val_batch = next(val_iter)
                        except StopIteration:
                            break

                        # Move validation batch to device
                        val_batch['visual_feats'] = val_batch['visual_feats'].to(
                            multi_gpu.device, non_blocking=True)
                        val_batch['audio_feats'] = val_batch['audio_feats'].to(
                            multi_gpu.device, non_blocking=True)
                        val_batch['text_feats'] = val_batch['text_feats'].to(
                            multi_gpu.device, non_blocking=True)
                        val_batch['masks'] = val_batch['masks'].to(
                            multi_gpu.device, non_blocking=True)
                        if 'labels' in val_batch:
                            val_batch['labels'] = val_batch['labels'].to(
                                multi_gpu.device, non_blocking=True)
                        if 'segments' in val_batch:
                            val_batch['segments'] = val_batch['segments'].to(
                                multi_gpu.device, non_blocking=True)

                        # Calculate validation losses
                        val_output = model(val_batch)
                        # Access losses method through module for DDP-wrapped models
                        if hasattr(model, 'module'):
                            val_losses = model.module.losses(*val_output)
                        else:
                            val_losses = model.losses(*val_output)
                        val_batch_size = val_batch['text_feats'].shape[0]

                        val_cls_loss = val_losses['cls_loss'].item(
                        ) / val_batch_size
                        val_final_loss = val_cls_loss

                        val_cls_losses.append(val_cls_loss)
                        val_total_losses.append(val_final_loss)

                    # Calculate average validation losses
                    if val_cls_losses:
                        avg_val_cls = sum(val_cls_losses) / len(val_cls_losses)
                        avg_val_total = sum(
                            val_total_losses) / len(val_total_losses)

                        # Log intra-epoch validation metrics
                        if is_main_process():
                            wandb.log({
                                'intra_eval/cls_loss': avg_val_cls,
                                'intra_eval/total_loss': avg_val_total,
                                'intra_eval/iteration': i + 1,
                                'intra_eval/epoch': epoch
                            }, step=global_step_iter)

                            print(
                                f"\nIteration {i+1}: Train Loss: {batch_loss:.4f}, Val Loss: {avg_val_total:.4f}")

                # Switch back to training mode
                logger.debug(
                    f"Intra-epoch validation completed, switching back to train mode - batch {i+1}, epoch {epoch+1} - rank {multi_gpu.rank}")
                model.train()

            # Print progress (only on main process)
            if is_main_process():
                print(f"Epoch {epoch+1}/{num_epochs}, Iter {i+1}/{len(train_data_loader)}, Total Loss: {batch_loss:.3f}, cls Loss: {cls_loss:.3f}, Time: {time.time() - start_time:.3f}s", end='\r')

            # Add periodic debug checkpoint
            if i > 0 and i % 100 == 0:
                logger.debug(
                    f"Completed {i+1} batches in epoch {epoch+1} - rank {multi_gpu.rank}")
        end_time = time.time()
        logger.debug(
            f"Training loop completed for epoch {epoch+1} - rank {multi_gpu.rank}")

        # Calculate average losses
        logger.debug(
            f"Calculating average losses for epoch {epoch+1} - rank {multi_gpu.rank}")
        num_batches = len(train_data_loader)
        avg_cls_loss = total_cls_loss / num_batches
        avg_total_loss = total_loss / num_batches
        logger.debug(f"Average losses calculated - rank {multi_gpu.rank}")

        # save checkpoint to disk (only on main process)
        if epoch % cfg['train']['save_epochs'] == 0 and is_main_process():
            logger.debug(
                f"Saving checkpoint for epoch {epoch+1} - main process")
            # Handle DDP model state dict
            model_state = model.module.state_dict() if hasattr(
                model, 'module') else model.state_dict()
            checkpoint = {
                'model': model_state,
                'optimizer': optimizer.state_dict(),
                'cosine_scheduler': cosine_scheduler.state_dict(),
                'warmup_scheduler': warmup_scheduler.state_dict(),
                'epoch': epoch,
                'loss': avg_total_loss
            }
            checkpoint_file = os.path.join(
                checkpoint_path, f'epoch_{epoch}.pth')
            torch.save(checkpoint, checkpoint_file)
            logger.debug(
                f"Checkpoint saved to {checkpoint_file} - main process")

        # Reduce across processes for distributed training
        logger.debug(
            f"About to reduce losses across processes for distributed training - epoch {epoch+1} - rank {multi_gpu.rank}")
        if multi_gpu.strategy == 'ddp' and multi_gpu.world_size > 1:
            avg_cls_loss_tensor = torch.tensor(
                avg_cls_loss, device=multi_gpu.device)
            avg_total_loss_tensor = torch.tensor(
                avg_total_loss, device=multi_gpu.device)

            avg_cls_loss = multi_gpu.reduce_tensor(avg_cls_loss_tensor).item()
            avg_total_loss = multi_gpu.reduce_tensor(
                avg_total_loss_tensor).item()
            logger.debug(
                f"Cross-process loss reduction completed - epoch {epoch+1} - rank {multi_gpu.rank}")

        # Log epoch-level metrics (only on main process)
        logger.debug(
            f"About to log epoch-level metrics - epoch {epoch+1} - rank {multi_gpu.rank}")
        if is_main_process():
            wandb.log({
                'epoch/avg_cls_loss': avg_cls_loss,
                'epoch/avg_total_loss': avg_total_loss,
                'epoch/epoch': epoch + 1,
                'epoch/time': end_time - start_time
            }, step=epoch)
            logger.debug(
                f"Epoch-level metrics logged to wandb - epoch {epoch+1}")

        epoch_message = f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_total_loss:.3f}, Avg cls Loss: {avg_cls_loss:.3f}, Time: {end_time - start_time:.3f}s"

        if is_main_process():
            print(epoch_message)

            with open(os.path.join(checkpoint_path, 'a-log.txt'), 'a') as f:
                f.write(epoch_message + '\n')
            logger.debug(
                f"Epoch message written to log file - epoch {epoch+1}")

        if epoch % cfg['train']['eval_freq'] == 0:
            logger.debug(
                f"Starting final evaluation for epoch {epoch+1} - rank {multi_gpu.rank}")
            logger.debug(
                f"Test data loader has {len(test_data_loader)} batches - rank {multi_gpu.rank}")
            model.eval()
            logger.debug(f"Model set to eval mode - rank {multi_gpu.rank}")
            # Initialize debug visualizer for this validation run
            debug_viz = ValidationDebugger(output_dir=os.path.join(
                checkpoint_path, "debug_outputs")) if is_main_process() else None
            logger.debug(
                f"Debug visualizer initialized - rank {multi_gpu.rank}")

            with torch.no_grad():
                total_AP = []
                totol_recall = []
                count = 0
                total_tIoU = []
                # Track validation losses
                total_val_cls_loss = 0
                total_val_loss = 0
                # Use tqdm only on main process for cleaner output
                data_iter = tqdm(
                    test_data_loader) if is_main_process() else test_data_loader

                for batch_idx, batch in enumerate(data_iter):
                    count += 1
                    if batch_idx == 0:
                        logger.debug(
                            f"Processing first validation batch - epoch {epoch+1} - rank {multi_gpu.rank}")
                    elif batch_idx % 10 == 0:
                        logger.debug(
                            f"Processing validation batch {batch_idx+1} - epoch {epoch+1} - rank {multi_gpu.rank}")
                    # Move tensors to appropriate device
                    batch['visual_feats'] = batch['visual_feats'].to(
                        multi_gpu.device, non_blocking=True)
                    batch['audio_feats'] = batch['audio_feats'].to(
                        multi_gpu.device, non_blocking=True)
                    batch['text_feats'] = batch['text_feats'].to(
                        multi_gpu.device, non_blocking=True)
                    batch['masks'] = batch['masks'].to(
                        multi_gpu.device, non_blocking=True)

                    # Also move labels and segments for loss calculation
                    if 'labels' in batch:
                        batch['labels'] = batch['labels'].to(
                            multi_gpu.device, non_blocking=True)
                    if 'segments' in batch:
                        batch['segments'] = batch['segments'].to(
                            multi_gpu.device, non_blocking=True)

                    # Calculate validation losses
                    if batch_idx == 0:
                        logger.debug(
                            f"About to call model forward pass for validation - batch {batch_idx+1} - rank {multi_gpu.rank}")
                    output = model(batch)
                    if batch_idx == 0:
                        logger.debug(
                            f"Validation forward pass completed - batch {batch_idx+1} - rank {multi_gpu.rank}")
                    # Access losses method through module for DDP-wrapped models
            if hasattr(model, 'module'):
                losses = model.module.losses(*output)
            else:
                losses = model.losses(*output)
                    if batch_idx == 0:
                        logger.debug(
                            f"Validation loss computation completed - batch {batch_idx+1} - rank {multi_gpu.rank}")
                    batch_size = batch['text_feats'].shape[0]

                    # Accumulate losses
                    val_cls_loss = losses['cls_loss'].item() / batch_size
                    val_total_loss = val_cls_loss

                    total_val_cls_loss += val_cls_loss
                    total_val_loss += val_total_loss

                    # Log samples for debugging (only for first 10 batches and on main process)
                    if debug_viz and count <= 10:
                        # Unpack the output
                        masks, out_cls_logits, out_offsets, gt_cls_labels, gt_offsets, feats = output

                        # Log each sample in the batch
                        # Log max 2 samples per batch
                        for sample_idx in range(min(batch_size, 2)):
                            debug_viz.log_validation_sample(
                                batch_idx=count,
                                video_id=batch['video_id'][sample_idx].item() if torch.is_tensor(
                                    batch['video_id'][sample_idx]) else batch['video_id'][sample_idx],
                                pred_offsets=out_offsets[sample_idx],
                                gt_offsets=gt_offsets[sample_idx],
                                cls_logits=out_cls_logits[sample_idx],
                                gt_labels=gt_cls_labels[sample_idx],
                                cls_loss=val_cls_loss,
                                reg_loss=0.0,  # Regression loss removed, using placeholder
                                masks=masks[sample_idx]
                            )

                    # Get model predictions (handle DDP wrapper)
                    if batch_idx == 0:
                        logger.debug(
                            f"About to get model predictions for validation - batch {batch_idx+1} - rank {multi_gpu.rank}")
                        # Access inference method through module for DDP-wrapped models
                        if hasattr(model, 'module'):
                            preds = model.module.inference_(batch, cfg['test_cfg'])
                        else:
                            preds = model.inference_(batch, cfg['test_cfg'])
                    if batch_idx == 0:
                        logger.debug(
                            f"Model predictions completed - batch {batch_idx+1} - rank {multi_gpu.rank}")

                    if batch_idx == 0:
                        logger.debug(
                            f"About to calculate tIoU metrics - batch {batch_idx+1} - rank {multi_gpu.rank}")
                    for i in range(len(preds)):
                        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
                        precision_per_threshold = calculate_tiou(
                            batch['gt_segments'][i], preds[i]['segments'].tolist(), thresholds)
                        total_tIoU.append(precision_per_threshold)
                    if batch_idx == 0:
                        logger.debug(
                            f"tIoU metrics calculated - batch {batch_idx+1} - rank {multi_gpu.rank}")

                # Calculate metrics
                logger.debug(
                    f"Calculating final evaluation metrics - epoch {epoch+1} - rank {multi_gpu.rank}")
                tIoU = {}
                for threshold in thresholds:
                    tIoU[threshold] = sum([item[threshold]
                                          for item in total_tIoU]) / len(total_tIoU)
                AtIoU = sum(item for item in tIoU.values()) / len(tIoU)
                logger.debug(
                    f"Final evaluation metrics calculated - epoch {epoch+1} - rank {multi_gpu.rank}")

                # Calculate average validation losses
                avg_val_cls_loss = total_val_cls_loss / count
                avg_val_total_loss = total_val_loss / count

                # Synchronize evaluation results across processes
                logger.debug(
                    f"About to synchronize evaluation results across processes - rank {multi_gpu.rank}")
                multi_gpu.barrier()
                logger.debug(
                    f"Evaluation synchronization completed - rank {multi_gpu.rank}")

                if AtIoU > best_tIoU:
                    best_tIoU = AtIoU
                    best_epoch = epoch

                    # Save best checkpoint (only on main process)
                    if is_main_process():
                        # Handle DDP model state dict
                        model_state = model.module.state_dict() if hasattr(
                            model, 'module') else model.state_dict()
                        checkpoint = {
                            'model': model_state,
                            'optimizer': optimizer.state_dict(),
                            'cosine_scheduler': cosine_scheduler.state_dict(),
                            'warmup_scheduler': warmup_scheduler.state_dict(),
                            'epoch': epoch,
                            'loss': avg_total_loss
                        }
                        best_checkpoint_file = os.path.join(
                            checkpoint_path, f'best.pth')
                        torch.save(checkpoint, best_checkpoint_file)
                        logger.debug(
                            f"Best checkpoint saved to {best_checkpoint_file} - main process")

                        # Log best metrics to wandb
                        wandb.run.summary["best_tIoU"] = best_tIoU
                        wandb.run.summary["best_epoch"] = best_epoch
                # Log evaluation metrics (only on main process)
                if is_main_process():
                    eval_metrics = {
                        f"eval/tIoU@{threshold}": tIoU[threshold]
                        for threshold in thresholds
                    }
                    eval_metrics["eval/AtIoU"] = AtIoU
                    eval_metrics[f"eval/AP_threshold={cfg['test_cfg']['pre_nms_thresh']}@tIoU"] = AtIoU

                    # Add validation losses
                    eval_metrics["eval/cls_loss"] = avg_val_cls_loss
                    eval_metrics["eval/total_loss"] = avg_val_total_loss

                    # Use global step to align with batch metrics
                    global_step = (epoch + 1) * len(train_data_loader)
                    wandb.log(eval_metrics, step=global_step)
                    eval_message = f"Epoch {epoch+1}/{num_epochs}, tIoU: {tIoU}, best tIoU: {best_tIoU:.4f}, best epoch: {best_epoch}"
                    print(eval_message)
                    with open(os.path.join(checkpoint_path, 'log.txt'), 'a') as f:
                        f.write(eval_message + '\n')

                    # Create debug visualizations and save logs
                    if debug_viz:
                        viz_paths = debug_viz.visualize_predictions(
                            epoch=epoch, num_samples=5, prefix="test")
                        log_paths = debug_viz.save_debug_logs(epoch=epoch)
                        debug_summary = debug_viz.get_debug_summary()
                        print(
                            f"Debug outputs saved to: {debug_summary['debug_dir']}")

                        # Upload debug files to wandb
                        # Log all visualizations together, grouped by epoch
                        if viz_paths:
                            viz_log = {}
                            for viz_path, video_id, prefix in viz_paths:
                                # Create grouped key with prefix
                                key = f"debug/{prefix}/{video_id}"
                                viz_log[key] = wandb.Image(viz_path,
                                                           caption=f"Epoch {epoch}, {prefix.capitalize()} Video {video_id}")

                            wandb.log(viz_log, step=global_step)

                        # Save log files to wandb
                        for log_path in log_paths:
                            wandb.save(log_path, base_path=checkpoint_path)

                    # Visualize training samples
                    if is_main_process():
                        print("\nProcessing training samples for visualization...")
                        train_debug_viz = ValidationDebugger(output_dir=os.path.join(
                            checkpoint_path, "debug_outputs"))

                        # Process each debug training sample
                        logger.debug(
                            f"Processing debug training samples - epoch {epoch+1}")
                        # Limit to 5 samples
                        for sample_idx, sample_info in enumerate(debug_train_samples[:5]):
                            logger.debug(
                                f"Processing debug training sample {sample_idx+1}/5 - video_id: {sample_info['video_id']} - epoch {epoch+1}")
                            # Create a single-sample batch
                            sample_data = sample_info['data']
                            train_batch = collate_fn([sample_data])

                            # Move to device
                            train_batch['visual_feats'] = train_batch['visual_feats'].to(
                                multi_gpu.device, non_blocking=True)
                            train_batch['audio_feats'] = train_batch['audio_feats'].to(
                                multi_gpu.device, non_blocking=True)
                            train_batch['text_feats'] = train_batch['text_feats'].to(
                                multi_gpu.device, non_blocking=True)
                            train_batch['masks'] = train_batch['masks'].to(
                                multi_gpu.device, non_blocking=True)
                            train_batch['labels'] = train_batch['labels'].to(
                                multi_gpu.device, non_blocking=True)
                            train_batch['segments'] = train_batch['segments'].to(
                                multi_gpu.device, non_blocking=True)

                            # Get model predictions
                            logger.debug(
                                f"About to get training sample predictions - sample {sample_idx+1} - epoch {epoch+1}")
                            with torch.no_grad():
                                output = model(train_batch)
                                logger.debug(
                                    f"Training sample forward pass completed - sample {sample_idx+1} - epoch {epoch+1}")
                                # Access losses method through module for DDP-wrapped models
                                if hasattr(model, 'module'):
                                    losses = model.module.losses(*output)
                                else:
                                    losses = model.losses(*output)
                                batch_size = train_batch['text_feats'].shape[0]
                                train_cls_loss = losses['cls_loss'].item(
                                ) / batch_size

                                # Unpack the output
                                masks, out_cls_logits, out_offsets, gt_cls_labels, gt_offsets, feats = output

                                # Log the sample
                                train_debug_viz.log_validation_sample(
                                    batch_idx=0,
                                    video_id=sample_info['video_id'],
                                    pred_offsets=out_offsets[0],
                                    gt_offsets=gt_offsets[0],
                                    cls_logits=out_cls_logits[0],
                                    gt_labels=gt_cls_labels[0],
                                    cls_loss=train_cls_loss,
                                    reg_loss=0.0,
                                    masks=masks[0]
                                )

                        # Create visualizations for training samples
                        train_viz_paths = train_debug_viz.visualize_predictions(
                            epoch=epoch, num_samples=5, prefix="train")

                        logger.debug(
                            f"Debug training sample processing completed - epoch {epoch+1}")
                        # Upload training visualizations to wandb
                        if train_viz_paths:
                            logger.debug(
                                f"Uploading training visualizations to wandb - epoch {epoch+1}")
                            train_viz_log = {}
                            for viz_path, video_id, prefix in train_viz_paths:
                                # Create grouped key with prefix
                                key = f"debug/{prefix}/{video_id}"
                                train_viz_log[key] = wandb.Image(viz_path,
                                                                 caption=f"Epoch {epoch}, {prefix.capitalize()} Video {video_id}")

                            wandb.log(train_viz_log, step=global_step)
                            logger.debug(
                                f"Training visualizations uploaded - epoch {epoch+1}")
    logger.debug(
        f"Training completed, starting cleanup - rank {multi_gpu.rank}")
    # Cleanup and finish (only on main process)
    if is_main_process():
        logger.debug(f"Finishing wandb - main process")
        wandb.finish()
        logger.debug(f"WandB finished - main process")

    # Clean up distributed training
    logger.debug(
        f"About to cleanup distributed training - rank {multi_gpu.rank}")
    multi_gpu.cleanup()
    logger.debug(
        f"Distributed training cleanup completed - rank {multi_gpu.rank}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    main(args)
