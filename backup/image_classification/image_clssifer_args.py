from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from numpy import Infinity

@dataclass
class TrainArgs:
    model_name: str
    dataloader_kind: str
    shuffle: bool = False
    num_pytorch_workers:int = 0

    epochs: int = 1 #Number of epochs to run
    global_batch_size: int = 64
    global_epoch_max_iters: Optional[int] = None #Size of the epoch

    # Optimization args
    learning_rate: float = 1e-3
    weight_decay: float = 0.02

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
    """Inputs and outputs related arguments"""
    # Optional because pretrain/tinyllama hardcodes the path
    train_data_dir: Optional[Path] = Path("data/alpaca")
    """Where to read training data from"""
    val_data_dir: Optional[Path] = None
    """Where to read validation data from"""
    # checkpoint_dir: Optional[Path] = None
    """Where to read weights and tokenizer data from"""
    # out_dir: Path = Path("out/adapter/alpaca")
    """Where to save artifacts"""
    log_dir: Optional[Path] = None
    log_interval: Optional[int] = 1

