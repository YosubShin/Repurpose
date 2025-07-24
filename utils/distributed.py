"""
Distributed Training Utilities for Multi-GPU Support

This module provides utilities for setting up and managing distributed training
with PyTorch DistributedDataParallel (DDP) and DataParallel (DP).
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel as DP
from torch.utils.data.distributed import DistributedSampler
import logging
from typing import Optional, Tuple, Dict, Any
import socket
import subprocess
import time
import datetime


def find_free_port() -> int:
    """Find a free port for distributed training."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def detect_slurm_env() -> Dict[str, Any]:
    """
    Detect SLURM environment variables and return distributed training parameters.

    Returns:
        Dict containing rank, world_size, local_rank, and master_addr/port
    """
    slurm_info = {}

    # Check if running under SLURM
    if 'SLURM_PROCID' in os.environ:
        slurm_info['rank'] = int(os.environ['SLURM_PROCID'])
        slurm_info['world_size'] = int(os.environ['SLURM_NTASKS'])
        slurm_info['local_rank'] = int(os.environ.get('SLURM_LOCALID', 0))

        # Get master node information
        slurm_info['master_addr'] = os.environ.get(
            'SLURM_LAUNCH_NODE_IPADDR', 'localhost')
        if 'SLURM_STEP_NODELIST' in os.environ:
            # Parse first node from nodelist
            nodelist = os.environ['SLURM_STEP_NODELIST']
            if '[' in nodelist:
                # Handle compressed nodelist like "node[01-04]"
                master_node = nodelist.split('[')[
                    0] + nodelist.split('[')[1].split('-')[0].split(',')[0]
            else:
                master_node = nodelist.split(',')[0]
            slurm_info['master_addr'] = master_node

        # Set a fixed port for SLURM jobs (can be overridden by environment)
        slurm_info['master_port'] = int(os.environ.get('MASTER_PORT', 29500))

        slurm_info['is_slurm'] = True
        logging.info(f"Detected SLURM environment: rank={slurm_info['rank']}, "
                     f"world_size={slurm_info['world_size']}, master_addr={slurm_info['master_addr']}")
    else:
        slurm_info['is_slurm'] = False

    return slurm_info


def get_network_interface() -> str:
    """
    Get the appropriate network interface for distributed training.

    Returns:
        Network interface name or IP address
    """
    # Common HPC network interfaces
    hpc_interfaces = ['ib0', 'ib1', 'eth0', 'enp0s8']

    try:
        # Try to get IP from hostname
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
        if not ip.startswith('127.'):
            return ip
    except:
        pass

    # Fallback to localhost
    return 'localhost'


def setup_distributed(rank: int, world_size: int, backend: str = 'nccl',
                      master_addr: str = 'localhost', master_port: str = None,
                      timeout: int = 1800) -> bool:
    """
    Initialize distributed training process group with HPC support.

    Args:
        rank: Process rank (0 to world_size-1)
        world_size: Total number of processes
        backend: Communication backend ('nccl' for GPU, 'gloo' for CPU)
        master_addr: Address of rank 0 process
        master_port: Port for communication
        timeout: Timeout for initialization in seconds

    Returns:
        bool: True if setup successful, False otherwise
    """
    logger = logging.getLogger(__name__)

    try:
        # Detect SLURM environment
        slurm_info = detect_slurm_env()

        if slurm_info['is_slurm']:
            # Use SLURM-provided values
            rank = slurm_info['rank']
            world_size = slurm_info['world_size']
            master_addr = slurm_info['master_addr']
            master_port = str(slurm_info['master_port'])
            local_rank = slurm_info['local_rank']
        else:
            # Use provided values or environment variables
            rank = int(os.environ.get('RANK', rank))
            world_size = int(os.environ.get('WORLD_SIZE', world_size))
            master_addr = os.environ.get('MASTER_ADDR', master_addr)
            local_rank = int(os.environ.get('LOCAL_RANK', rank %
                             torch.cuda.device_count()))

            if master_port is None:
                master_port = os.environ.get(
                    'MASTER_PORT', str(find_free_port()))

        # Set environment variables
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = str(master_port)
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['LOCAL_RANK'] = str(local_rank)

        logger.info(f"Initializing distributed training:")
        logger.info(f"  Backend: {backend}")
        logger.info(f"  Rank: {rank}/{world_size}")
        logger.info(f"  Local Rank: {local_rank}")
        logger.info(f"  Master: {master_addr}:{master_port}")
        logger.info(f"  Timeout: {timeout}s")

        # Set device for this process
        if torch.cuda.is_available() and backend == 'nccl':
            device_id = local_rank % torch.cuda.device_count()
            torch.cuda.set_device(device_id)
            logger.info(f"  CUDA Device: {device_id}")
            # Log additional device info for debugging
            logger.info(f"  Available GPUs: {torch.cuda.device_count()}")
            logger.info(f"  Current device: {torch.cuda.current_device()}")

        # Initialize process group with timeout
        start_time = time.time()
        timeout_timedelta = datetime.timedelta(seconds=timeout)

        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
            timeout=timeout_timedelta
        )

        init_time = time.time() - start_time
        logger.info(
            f"Distributed initialization successful in {init_time:.2f}s")

        # Test communication
        if dist.is_initialized():
            # Simple all-reduce test
            test_tensor = torch.ones(
                1).cuda() if torch.cuda.is_available() else torch.ones(1)
            dist.all_reduce(test_tensor)
            expected_sum = world_size
            if abs(test_tensor.item() - expected_sum) < 1e-6:
                logger.info("Distributed communication test passed")
                return True
            else:
                logger.error(
                    f"Distributed communication test failed: expected {expected_sum}, got {test_tensor.item()}")
                return False

        return True

    except Exception as e:
        logger.error(f"Failed to setup distributed training: {e}")
        logger.error(f"Environment variables:")
        for key in ['RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT']:
            logger.error(f"  {key}: {os.environ.get(key, 'Not set')}")
        return False


def cleanup_distributed():
    """Clean up distributed training process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_rank() -> int:
    """Get current process rank."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get total number of processes."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    """Check if current process is the main process (rank 0)."""
    return get_rank() == 0


def get_device() -> torch.device:
    """Get the appropriate device for current process."""
    if torch.cuda.is_available():
        if dist.is_available() and dist.is_initialized():
            # Use LOCAL_RANK instead of RANK for proper GPU assignment
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            return torch.device(f'cuda:{local_rank}')
        else:
            return torch.device('cuda')
    return torch.device('cpu')


class MultiGPUStrategy:
    """
    Multi-GPU training strategy manager.

    Handles both DataParallel and DistributedDataParallel strategies with HPC support.
    """

    def __init__(self, strategy: str = 'auto', backend: str = 'nccl',
                 timeout: int = 1800, find_unused_parameters: bool = False):
        """
        Initialize multi-GPU strategy.

        Args:
            strategy: 'auto', 'dp', 'ddp', or 'single'
            backend: Communication backend for DDP
            timeout: Timeout for distributed initialization
            find_unused_parameters: Whether to find unused parameters in DDP
        """
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.strategy = strategy
        self.backend = backend
        self.timeout = timeout
        self.find_unused_parameters = find_unused_parameters
        self.world_size = 1
        self.rank = 0
        self.local_rank = 0
        self.device = torch.device('cpu')
        self.is_distributed = False
        self.slurm_info = detect_slurm_env()

        # Auto-detect strategy if needed
        if strategy == 'auto':
            self.strategy = self._auto_detect_strategy()

        self._setup_device_info()

    def _auto_detect_strategy(self) -> str:
        """Auto-detect the best multi-GPU strategy with HPC support."""
        if not torch.cuda.is_available():
            return 'single'

        gpu_count = torch.cuda.device_count()

        # Check for SLURM environment first
        if self.slurm_info['is_slurm']:
            world_size = self.slurm_info['world_size']
            if world_size > 1:
                self.logger.info(
                    f"SLURM detected with {world_size} processes - using DDP")
                return 'ddp'

        # Check if we're in a distributed environment (torchrun, torch.distributed.launch)
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            world_size = int(os.environ['WORLD_SIZE'])
            if world_size > 1:
                self.logger.info(
                    f"Distributed environment detected with {world_size} processes - using DDP")
                return 'ddp'

        # Single process cases
        if gpu_count < 2:
            return 'single'
        elif gpu_count == 1:
            return 'single'

        # Multi-GPU single-node: prefer DDP for better performance and memory usage
        self.logger.info(
            f"Single-node multi-GPU detected ({gpu_count} GPUs) - using DDP")
        return 'ddp'

    def _setup_device_info(self):
        """Setup device information based on strategy."""
        if self.strategy == 'single':
            self.device = get_device()
            self.world_size = 1
            self.rank = 0
            self.local_rank = 0

        elif self.strategy == 'dp':
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
            self.world_size = 1
            self.rank = 0
            self.local_rank = 0

        elif self.strategy == 'ddp':
            self.is_distributed = True

            # Use SLURM info if available
            if self.slurm_info['is_slurm']:
                self.rank = self.slurm_info['rank']
                self.world_size = self.slurm_info['world_size']
                self.local_rank = self.slurm_info['local_rank']
            elif 'RANK' in os.environ:
                self.rank = int(os.environ['RANK'])
                self.world_size = int(os.environ['WORLD_SIZE'])
                self.local_rank = int(os.environ.get(
                    'LOCAL_RANK', self.rank % torch.cuda.device_count()))
            else:
                # Single-node multi-GPU case
                gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
                self.rank = 0
                self.world_size = gpu_count
                self.local_rank = 0

            # Set device based on local_rank for proper GPU assignment
            if torch.cuda.is_available():
                self.device = torch.device(f'cuda:{self.local_rank}')
                self.logger.info(f"DDP process rank {self.rank} assigned to GPU {self.local_rank}")
            else:
                self.device = torch.device('cpu')

    def setup(self) -> bool:
        """Setup the multi-GPU environment."""
        if self.strategy == 'ddp' and self.world_size > 1:
            # Get master address and port
            master_addr = 'localhost'
            master_port = None

            if self.slurm_info['is_slurm']:
                master_addr = self.slurm_info['master_addr']
                master_port = str(self.slurm_info['master_port'])
            else:
                master_addr = os.environ.get(
                    'MASTER_ADDR', get_network_interface())
                master_port = os.environ.get('MASTER_PORT')

            success = setup_distributed(
                rank=self.rank,
                world_size=self.world_size,
                backend=self.backend,
                master_addr=master_addr,
                master_port=master_port,
                timeout=self.timeout
            )

            if not success:
                self.logger.warning(
                    "DDP setup failed, falling back to DataParallel")
                self.strategy = 'dp'
                self.is_distributed = False
                self.world_size = 1
                self.rank = 0
                return True

            return success
        return True

    def cleanup(self):
        """Cleanup the multi-GPU environment."""
        if self.strategy == 'ddp':
            cleanup_distributed()

    def wrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Wrap model for multi-GPU training.

        Args:
            model: PyTorch model to wrap

        Returns:
            Wrapped model ready for multi-GPU training
        """
        # Move model to appropriate device first
        model = model.to(self.device)

        if self.strategy == 'dp' and torch.cuda.device_count() > 1:
            # DataParallel
            model = DP(model)
            logging.info(
                f"Model wrapped with DataParallel on {torch.cuda.device_count()} GPUs")

        elif self.strategy == 'ddp' and self.world_size > 1:
            # DistributedDataParallel
            device_ids = [
                self.local_rank] if torch.cuda.is_available() else None
            output_device = self.local_rank if torch.cuda.is_available() else None

            model = DDP(
                model,
                device_ids=device_ids,
                output_device=output_device,
                find_unused_parameters=self.find_unused_parameters
            )
            logging.info(f"Model wrapped with DistributedDataParallel, rank {self.rank}/{self.world_size}, "
                         f"local_rank {self.local_rank}, find_unused_parameters={self.find_unused_parameters}")

        else:
            logging.info(f"Model using single GPU/CPU: {self.device}")

        return model

    def create_dataloader(self, dataset, batch_size: int, shuffle: bool = True,
                          num_workers: int = 0, **kwargs) -> torch.utils.data.DataLoader:
        """
        Create a DataLoader appropriate for the multi-GPU strategy.

        Args:
            dataset: PyTorch dataset
            batch_size: Batch size per GPU
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            **kwargs: Additional DataLoader arguments

        Returns:
            Configured DataLoader
        """
        from torch.utils.data import DataLoader

        sampler = None
        if self.strategy == 'ddp' and self.world_size > 1:
            # Use DistributedSampler for DDP
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=shuffle
            )
            shuffle = False  # DistributedSampler handles shuffling

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            **kwargs
        )

        return dataloader

    def reduce_tensor(self, tensor: torch.Tensor, average: bool = True) -> torch.Tensor:
        """
        Reduce tensor across all processes in distributed training.

        Args:
            tensor: Tensor to reduce
            average: Whether to average the values

        Returns:
            Reduced tensor
        """
        if self.strategy != 'ddp' or self.world_size <= 1:
            return tensor

        # Clone tensor to avoid modifying original
        reduced_tensor = tensor.clone()

        # All-reduce across processes
        dist.all_reduce(reduced_tensor, op=dist.ReduceOp.SUM)

        if average:
            reduced_tensor /= self.world_size

        return reduced_tensor

    def barrier(self):
        """Synchronize all processes."""
        if self.strategy == 'ddp' and self.world_size > 1:
            dist.barrier()

    def print_setup_info(self):
        """Print detailed setup information."""
        self.logger.info("=" * 60)
        self.logger.info("DISTRIBUTED TRAINING SETUP")
        self.logger.info("=" * 60)
        self.logger.info(f"Strategy: {self.strategy}")
        self.logger.info(f"Backend: {self.backend}")
        self.logger.info(f"World Size: {self.world_size}")
        self.logger.info(f"Rank: {self.rank}")
        self.logger.info(f"Local Rank: {self.local_rank}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            self.logger.info(f"CUDA Device Count: {torch.cuda.device_count()}")
            self.logger.info(
                f"Current CUDA Device: {torch.cuda.current_device()}")

        # Environment variables
        self.logger.info("Environment Variables:")
        env_vars = ['RANK', 'WORLD_SIZE',
                    'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT']
        for var in env_vars:
            value = os.environ.get(var, 'Not set')
            self.logger.info(f"  {var}: {value}")

        # SLURM info
        if self.slurm_info['is_slurm']:
            self.logger.info("SLURM Environment:")
            slurm_vars = ['SLURM_PROCID', 'SLURM_NTASKS',
                          'SLURM_LOCALID', 'SLURM_STEP_NODELIST']
            for var in slurm_vars:
                value = os.environ.get(var, 'Not set')
                self.logger.info(f"  {var}: {value}")

        self.logger.info("=" * 60)

    def save_checkpoint(self, state_dict: Dict[str, Any], filepath: str):
        """
        Save checkpoint, handling DDP module unwrapping.

        Args:
            state_dict: State dictionary to save
            filepath: Path to save checkpoint
        """
        # Only save from main process in distributed training
        if not is_main_process():
            return

        # Unwrap DDP module if needed
        if 'model' in state_dict:
            model_state = state_dict['model']
            if hasattr(model_state, 'module'):
                state_dict['model'] = model_state.module.state_dict()

        torch.save(state_dict, filepath)

    def load_checkpoint(self, model: torch.nn.Module, filepath: str,
                        optimizer: Optional[torch.optim.Optimizer] = None) -> Dict[str, Any]:
        """
        Load checkpoint, handling DDP module wrapping.

        Args:
            model: Model to load weights into
            filepath: Path to checkpoint file
            optimizer: Optional optimizer to load state into

        Returns:
            Loaded checkpoint dictionary
        """
        # Load checkpoint
        checkpoint = torch.load(filepath, map_location=self.device)

        # Handle model state loading
        if 'model' in checkpoint:
            model_state = checkpoint['model']

            # Handle DDP wrapped models
            if hasattr(model, 'module'):
                model.module.load_state_dict(model_state)
            else:
                model.load_state_dict(model_state)

        # Handle optimizer state loading
        if optimizer is not None and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])

        return checkpoint

    def get_effective_batch_size(self, batch_size: int) -> int:
        """
        Get effective batch size across all GPUs.

        Args:
            batch_size: Batch size per GPU

        Returns:
            Total effective batch size
        """
        if self.strategy == 'dp':
            return batch_size  # DataParallel uses full batch size on each GPU
        elif self.strategy == 'ddp':
            return batch_size * self.world_size
        else:
            return batch_size

    def print_setup_info(self):
        """Print information about the multi-GPU setup."""
        if is_main_process():
            logging.info("=" * 50)
            logging.info("MULTI-GPU SETUP INFO")
            logging.info("=" * 50)
            logging.info(f"Strategy: {self.strategy.upper()}")
            logging.info(f"World Size: {self.world_size}")
            logging.info(f"Rank: {self.rank}")
            logging.info(f"Device: {self.device}")
            logging.info(f"CUDA Available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                logging.info(f"CUDA Device Count: {torch.cuda.device_count()}")
                logging.info(
                    f"Current CUDA Device: {torch.cuda.current_device()}")
            logging.info("=" * 50)


def auto_select_strategy() -> str:
    """
    Automatically select the best multi-GPU strategy.

    Returns:
        Recommended strategy: 'single', 'dp', or 'ddp'
    """
    if not torch.cuda.is_available():
        return 'single'

    gpu_count = torch.cuda.device_count()
    if gpu_count < 2:
        return 'single'

    # Check available GPU memory
    min_memory_gb = min(
        torch.cuda.get_device_properties(i).total_memory / (1024**3)
        for i in range(gpu_count)
    )

    # If we have limited memory per GPU, recommend DDP for better memory efficiency
    if min_memory_gb < 8:
        return 'ddp'

    # For 2-4 GPUs with good memory, DataParallel is simpler
    if gpu_count <= 4:
        return 'dp'

    # For more GPUs, use DDP
    return 'ddp'


# Context manager for distributed training
class DistributedManager:
    """Context manager for distributed training setup and cleanup."""

    def __init__(self, strategy: str = 'auto', backend: str = 'nccl'):
        self.strategy_manager = MultiGPUStrategy(strategy, backend)

    def __enter__(self):
        success = self.strategy_manager.setup()
        if not success:
            raise RuntimeError("Failed to setup distributed training")
        return self.strategy_manager

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.strategy_manager.cleanup()
