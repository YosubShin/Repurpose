# Distributed Training Guide

This guide explains how to run distributed training on various environments including HPC clusters with SLURM.

## Quick Start

### Single GPU
```bash
python main.py --config_path configs/Repurpose.yaml
```

### Multi-GPU (Single Node) - Recommended
```bash
# Using torchrun (PyTorch 1.10+)
torchrun --nproc_per_node=2 main.py --config_path configs/Repurpose.yaml

# Using the provided launcher script
./run_ddp.sh -n 2
```

### HPC/SLURM Environment
```bash
# Submit job to SLURM
sbatch slurm_job.sh

# Or use the launcher script which auto-detects SLURM
./run_ddp.sh -n 2
```

## Configuration

The distributed training behavior is controlled by the `distributed` section in your config file:

```yaml
distributed:
  strategy: auto          # Options: auto, single, dp, ddp
  backend: nccl          # Communication backend (nccl for GPU, gloo for CPU)
  timeout: 1800          # Timeout for initialization (seconds)
  find_unused_parameters: false  # Set to true if you have unused parameters
```

### Strategy Options

- **auto**: Automatically detect the best strategy based on environment
- **single**: Single GPU/CPU training
- **dp**: DataParallel (single-node multi-GPU, legacy)
- **ddp**: DistributedDataParallel (recommended for multi-GPU)

## Launch Methods

### 1. Using torchrun (Recommended)

```bash
torchrun \
    --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29500 \
    main.py --config_path configs/Repurpose.yaml
```

### 2. Using torch.distributed.launch (Deprecated)

```bash
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --use_env \
    main.py --config_path configs/Repurpose.yaml
```

### 3. Using the provided launcher script

```bash
./run_ddp.sh -n 2 -c configs/Repurpose.yaml -l INFO
```

## HPC/SLURM Setup

### Environment Variables

The code automatically detects SLURM environment variables:

- `SLURM_PROCID`: Process rank
- `SLURM_NTASKS`: Total number of processes
- `SLURM_LOCALID`: Local rank on node
- `SLURM_STEP_NODELIST`: List of nodes

### SLURM Job Script

Customize `slurm_job.sh` for your cluster:

```bash
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=2           # Number of GPUs per node
#SBATCH --cpus-per-task=4             # CPU cores per GPU
#SBATCH --gres=gpu:2                  # GPU allocation
#SBATCH --mem=32G                     # Memory per node
#SBATCH --time=24:00:00               # Time limit
#SBATCH --partition=gpu               # Partition name
```

Submit the job:
```bash
sbatch slurm_job.sh
```

## Troubleshooting

### Common Issues

1. **TCP Connection Failures**
   - Set `NCCL_IB_DISABLE=1` to disable InfiniBand
   - Set `NCCL_P2P_DISABLE=1` to disable peer-to-peer communication
   - Increase timeout in config: `timeout: 3600`

2. **Hanging During Initialization**
   - Check if all processes can communicate
   - Verify MASTER_ADDR and MASTER_PORT are accessible
   - Enable debug logging: `export TORCH_DISTRIBUTED_DEBUG=INFO`

3. **Out of Memory Errors**
   - Reduce batch size in config
   - Enable gradient checkpointing
   - Use `find_unused_parameters: true` if needed

### Debug Environment Variables

Enable verbose logging for debugging:

```bash
export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1
```

### Check GPU Visibility

```bash
python -c "import torch; print(f'GPUs visible: {torch.cuda.device_count()}')"
nvidia-smi
```

## Performance Tips

1. **Use NCCL backend** for GPU communication (default)
2. **Set appropriate number of workers**: Usually 2-4 per GPU
3. **Pin memory** for faster data loading
4. **Use appropriate batch size**: Scale with number of GPUs
5. **Consider gradient accumulation** for very large effective batch sizes

## Multi-Node Training

For multi-node training, adjust the SLURM script:

```bash
#SBATCH --nodes=2                     # Number of nodes
#SBATCH --ntasks-per-node=2           # GPUs per node
```

And use:
```bash
torchrun \
    --nnodes=2 \
    --nproc_per_node=2 \
    --node_rank=$SLURM_NODEID \
    --master_addr=$(hostname -i) \
    --master_port=29500 \
    main.py --config_path configs/Repurpose.yaml
```

## Monitoring

### Check Job Status
```bash
squeue -u $USER                    # Check job queue
scontrol show job JOBID            # Detailed job info
scancel JOBID                      # Cancel job
```

### View Logs
```bash
tail -f logs/slurm_JOBID.out      # Follow output log
tail -f logs/slurm_JOBID.err      # Follow error log
```

### Resource Usage
```bash
sacct -j JOBID --format=JobID,JobName,State,Time,CPUTime,MaxRSS
```

## Example Configurations

### Small Scale (2 GPUs, Development)
```yaml
train:
  batch_size: 4
  epochs: 10

distributed:
  strategy: auto
  backend: nccl
  timeout: 1800
```

### Large Scale (Multiple Nodes)
```yaml
train:
  batch_size: 2  # Per GPU
  epochs: 50

distributed:
  strategy: ddp
  backend: nccl
  timeout: 3600
  find_unused_parameters: false
```

## Getting Help

If you encounter issues:

1. Check the logs in `logs/slurm_*.out` and `logs/slurm_*.err`
2. Enable debug logging with `--log-level DEBUG`
3. Verify your cluster's specific requirements
4. Contact your HPC administrator for cluster-specific issues