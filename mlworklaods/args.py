
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from numpy import Infinity

@dataclass
class TrainArgs:
    job_id: int
    model_name: str
    shuffle: bool = False
    dataload_only: bool = False
    num_pytorch_workers:int = 0
    epochs: int = 1 #Number of epochs to run
    global_batch_size: int = 64
    global_epoch_max_iters: Optional[int] = None #Size of the epoch
  
    # Optimization args
    learning_rate: float = 1e-3
    weight_decay: float = 0.02
    run_training: bool = True
    run_evaluation : bool = False
    simulate_data_delay: Optional[float] = None #Size of the epoch
    dataload_only : bool = False

    def epoch_max_iters(self, devices: int) -> int:
        if self.global_epoch_max_iters:
            epoch_max_iters = self.global_epoch_max_iters // devices
            assert epoch_max_iters > 0
            return epoch_max_iters
        else:
            return Infinity
        
    def batch_size(self, devices: int) -> int:
        """Number of samples between optimizer steps per data-parallel rank"""
        batch_size = self.global_batch_size // devices
        assert batch_size > 0
        return batch_size

@dataclass
class IOArgs:
    dataloader_kind: str
    train_data_dir: Optional[Path] = Path("data/alpaca")
    val_data_dir: Optional[Path] = None
    log_dir: Optional[Path] = None
    log_interval: Optional[int] = 1
    super_address: Optional[str] = None
    cache_address: Optional[str] = None
