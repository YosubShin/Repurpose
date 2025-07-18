from dataset.RepurposeClip import RepurposeClip, RepurposeClipTest
from dataset.RepurposeClip import collate_fn, collate_fn_test
from models.MMCTransformer import MMCTransformer
from utils.metrics import *
from utils.distributed import MultiGPUStrategy, is_main_process, get_rank, get_world_size
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


def main(args):
    # load config
    cfg = load_config(args.config_path)

    # Initialize multi-GPU strategy
    gpu_strategy = cfg.get('distributed', {}).get('strategy', 'auto')
    multi_gpu = MultiGPUStrategy(strategy=gpu_strategy)

    # Setup distributed training if needed
    if not multi_gpu.setup():
        raise RuntimeError("Failed to setup multi-GPU training")

    # Print setup info (only from main process)
    if is_main_process():
        multi_gpu.print_setup_info()

    seed_everything(cfg['train']['seed'])

    checkpoint_path = time.strftime('saved_model/ckpt_%Y%m%d_%H%M%S')
    if is_main_process():
        os.makedirs(checkpoint_path, exist_ok=True)
        with open(os.path.join(checkpoint_path, 'config.yaml'), 'w') as f:
            yaml.dump(cfg, f)
        print('The checkpoint path is %s' % checkpoint_path)

    # Synchronize all processes
    multi_gpu.barrier()

    train_dataset = RepurposeClip(**cfg['train_dataset'])
    test_dataset = RepurposeClipTest(**cfg['test_dataset'])

    model = MMCTransformer(**cfg['model'])
    model = multi_gpu.wrap_model(model)

    learning_rate = cfg['train']['lr']
    weight_decay = cfg['train']['weight_decay']
    batch_size = cfg['train']['batch_size']
    num_epochs = cfg['train']['epochs']
    warmup_epochs = cfg['train']['warmup_epochs']

    # Create data loaders with multi-GPU support
    train_data_loader = multi_gpu.create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=min(24, 4)  # Reduce workers for multi-GPU
    )

    test_data_loader = multi_gpu.create_dataloader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn_test,
        num_workers=min(24, 4)
    )

    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    num_iters = len(train_data_loader)

    warmup_steps = warmup_epochs * num_iters

    total_steps = num_epochs * num_iters

    def warmup_lambda(global_step):
        return (global_step + 1) / warmup_steps if (global_step + 1) <= warmup_steps else 1

    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)

    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    start_epoch = 0

    global_step = 0

    if args.resume:
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

    # Initialize wandb (only on main process)
    if is_main_process():
        effective_batch_size = multi_gpu.get_effective_batch_size(batch_size)
        run_name = f"mmc_{multi_gpu.strategy}_{multi_gpu.world_size}gpu_{time.strftime('%Y%m%d_%H%M%S')}"
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

        # Watch the model to log gradients and parameters
        # For DDP, watch the underlying module
        model_to_watch = model.module if hasattr(model, 'module') else model
        wandb.watch(model_to_watch, log="all", log_freq=100)

    best_tIoU = 0
    best_epoch = 0

    for epoch in range(start_epoch, num_epochs):
        # Set epoch for distributed sampler to ensure proper shuffling
        if hasattr(train_data_loader.sampler, 'set_epoch'):
            train_data_loader.sampler.set_epoch(epoch)

        model.train()
        total_loss = 0
        total_cls_loss = 0
        total_reg_loss = 0
        start_time = time.time()

        for i, batch in enumerate(train_data_loader):
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

            output = model(batch)
            losses = model.losses(*output)

            lambda_ = 0.2
            final_loss = (
                lambda_ * losses['cls_loss'] + (1-lambda_) * losses['reg_loss']) / batch_size
            optimizer.zero_grad()
            final_loss.backward()
            optimizer.step()

            # Reduce losses across GPUs for accurate logging
            cls_loss_tensor = multi_gpu.reduce_tensor(
                losses["cls_loss"] / batch_size)
            reg_loss_tensor = multi_gpu.reduce_tensor(
                losses["reg_loss"] / batch_size)
            final_loss_tensor = multi_gpu.reduce_tensor(final_loss)

            cls_loss = cls_loss_tensor.item()
            reg_loss = reg_loss_tensor.item()
            batch_loss = final_loss_tensor.item()
            total_cls_loss += cls_loss
            total_reg_loss += reg_loss
            total_loss += batch_loss

            # Log batch-level metrics (only on main process)
            if is_main_process():
                global_step_iter = epoch * len(train_data_loader) + i
                wandb.log({
                    'batch/cls_loss': cls_loss,
                    'batch/reg_loss': reg_loss,
                    'batch/total_loss': batch_loss,
                    'batch/learning_rate': optimizer.param_groups[0]['lr']
                }, step=global_step_iter)

            if global_step < warmup_steps:
                warmup_scheduler.step()
            else:
                cosine_scheduler.step()

            # Print progress (only on main process)
            if is_main_process():
                print(f"Epoch {epoch+1}/{num_epochs}, Iter {i+1}/{len(train_data_loader)}, Total Loss: {batch_loss:.3f},  cls Loss: {cls_loss:.3f}, reg Loss: {reg_loss:.3f}, Time: {time.time() - start_time:.3f}s", end='\r')
        end_time = time.time()

        # Calculate average losses
        num_batches = len(train_data_loader)
        avg_cls_loss = total_cls_loss / num_batches
        avg_reg_loss = total_reg_loss / num_batches
        avg_total_loss = total_loss / num_batches

        # save checkpoint to disk (only on main process)
        if epoch % cfg['train']['save_epochs'] == 0 and is_main_process():
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

            # Log checkpoint to wandb
            wandb.save(checkpoint_file)

        # Reduce across processes for distributed training
        if multi_gpu.strategy == 'ddp' and multi_gpu.world_size > 1:
            avg_cls_loss_tensor = torch.tensor(
                avg_cls_loss, device=multi_gpu.device)
            avg_reg_loss_tensor = torch.tensor(
                avg_reg_loss, device=multi_gpu.device)
            avg_total_loss_tensor = torch.tensor(
                avg_total_loss, device=multi_gpu.device)

            avg_cls_loss = multi_gpu.reduce_tensor(avg_cls_loss_tensor).item()
            avg_reg_loss = multi_gpu.reduce_tensor(avg_reg_loss_tensor).item()
            avg_total_loss = multi_gpu.reduce_tensor(
                avg_total_loss_tensor).item()

        # Log epoch-level metrics (only on main process)
        if is_main_process():
            wandb.log({
                'epoch/avg_cls_loss': avg_cls_loss,
                'epoch/avg_reg_loss': avg_reg_loss,
                'epoch/avg_total_loss': avg_total_loss,
                'epoch/epoch': epoch + 1,
                'epoch/time': end_time - start_time
            }, step=epoch)

        epoch_message = f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_total_loss:.3f}, Avg cls Loss: {avg_cls_loss:.3f}, Avg reg Loss: {avg_reg_loss:.3f}, Time: {end_time - start_time:.3f}s"

        if is_main_process():
            print(epoch_message)

            with open(os.path.join(checkpoint_path, 'a-log.txt'), 'a') as f:
                f.write(epoch_message + '\n')

        if epoch % cfg['train']['eval_freq'] == 0:
            model.eval()
            with torch.no_grad():
                total_AP = []
                totol_recall = []
                count = 0
                total_tIoU = []
                # Use tqdm only on main process for cleaner output
                data_iter = tqdm(
                    test_data_loader) if is_main_process() else test_data_loader

                for batch in data_iter:
                    count += 1
                    # Move tensors to appropriate device
                    batch['visual_feats'] = batch['visual_feats'].to(
                        multi_gpu.device, non_blocking=True)
                    batch['audio_feats'] = batch['audio_feats'].to(
                        multi_gpu.device, non_blocking=True)
                    batch['text_feats'] = batch['text_feats'].to(
                        multi_gpu.device, non_blocking=True)
                    batch['masks'] = batch['masks'].to(
                        multi_gpu.device, non_blocking=True)

                    # Get model predictions (handle DDP wrapper)
                    if hasattr(model, 'module'):
                        preds = model.module.inference_(batch, cfg['test_cfg'])
                    else:
                        preds = model.inference_(batch, cfg['test_cfg'])

                    for i in range(len(preds)):
                        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
                        precision_per_threshold = calculate_tiou(
                            batch['gt_segments'][i], preds[i]['segments'].tolist(), thresholds)
                        total_tIoU.append(precision_per_threshold)

                # Calculate metrics
                tIoU = {}
                for threshold in thresholds:
                    tIoU[threshold] = sum([item[threshold]
                                          for item in total_tIoU]) / len(total_tIoU)
                AtIoU = sum(item for item in tIoU.values()) / len(tIoU)

                # Synchronize evaluation results across processes
                multi_gpu.barrier()

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

                        # Log best model to wandb
                        wandb.save(best_checkpoint_file)
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
                    wandb.log(eval_metrics, step=epoch)
                    eval_message = f"Epoch {epoch+1}/{num_epochs}, tIoU: {tIoU}, best tIoU: {best_tIoU:.4f}, best epoch: {best_epoch}"
                    print(eval_message)
                    with open(os.path.join(checkpoint_path, 'log.txt'), 'a') as f:
                        f.write(eval_message + '\n')
    # Cleanup and finish (only on main process)
    if is_main_process():
        wandb.finish()

    # Clean up distributed training
    multi_gpu.cleanup()


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
