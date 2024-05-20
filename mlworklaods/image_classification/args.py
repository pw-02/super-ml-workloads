from dataclasses import dataclass, field
from typing import Optional, Union
from pathlib import Path
from numpy import Infinity


@dataclass
class TrainArgs:
    job_id: int
    model_name: str
    dataloader_kind: str
    batch_size: int  # Global batch size for all devices
    dataload_only: bool = False
    max_steps: Optional[int] = None
    max_epochs: Optional[int] = None
    limit_train_batches: Optional[int] = None
    limit_val_batches: Optional[int] = None
    grad_accum_steps: Optional[int] = 1
    # validation_frequency: Optional[int] = 1
    # Optimization parameters
    learning_rate: float = 1e-3
    weight_decay: float = 0.02
    run_training: bool = True
    run_evaluation: bool = False
    devices: Optional[int] = 1
    accelerator: Optional[str] = 'gpu'
    seed: int = 41
    log_dir: Optional[Path] = None
    log_freq: int = 1

    def get_epoch_max_iters(self, devices: int) -> int:
        """Calculate max iterations per epoch per device."""
        if self.epoch_max_iters:
            epoch_iters = self.epoch_max_iters #// devices
            if epoch_iters <= 0:
                raise ValueError("Epoch max iterations must be greater than zero.")
            return epoch_iters
        return Infinity

    def get_batch_size(self, devices: int) -> int:
        """Calculate batch size per device."""
        batch_size_per_device = self.batch_size #// devices
        if batch_size_per_device <= 0:
            raise ValueError("Batch size per device must be greater than zero.")
        return batch_size_per_device


@dataclass
class DataArgs:
    train_data_dir: Path = Path("data/alpaca")
    val_data_dir: Optional[Path] = None
    num_classes : Optional[Path] = None


@dataclass
class SUPERArgs:
    num_pytorch_workers: int = 0
    super_address: Optional[str] = None
    cache_address: Optional[str] = None
    simulate_data_delay: Optional[str] = None
    


@dataclass
class SHADEArgs:
    num_pytorch_workers: int = 0
    cache_address: Optional[str] = None
    working_set_size: Optional[str] = None
    replication_factor: Optional[str] = None


@dataclass
class LRUTorchArgs:
    num_pytorch_workers: int = 0
    cache_address: Optional[str] = None
    cache_granularity: Optional[str] = None
    shuffle: bool = False
