# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
import math
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
from omegaconf import DictConfig
import os

@dataclass
class BaseTrainArgs:
    job_id: int
    model_name: str
    dataloader_name: str
    log_dir: str
    num_workers: int =0
    log_interval: int = 1
    devices: int =1
    seed: int = 42
    accelerator: Optional[str] = 'gpu' 
    global_batch_size: int = 64
    micro_batch_size: int = 4
    checkpoint_interval: Optional[int] = 1000
    run_training: bool = True
    run_evaluation: bool = False
    max_steps: Optional[int] = None

    def gradient_accumulation_iters(self, devices: int) -> int:
        """Number of iterations between gradient synchronizations"""
        gradient_accumulation_iters = self.batch_size(devices) // self.micro_batch_size
        assert gradient_accumulation_iters > 0
        return gradient_accumulation_iters

    def batch_size(self, devices: int) -> int:
        """Number of samples between optimizer steps per data-parallel rank"""
        batch_size = self.global_batch_size // devices
        assert batch_size > 0
        return batch_size

    def warmup_iters(self, devices: int, max_iters: int, train_dataloader) -> int:
        """Number of iterations to warm up the learning rate."""
        if self.lr_warmup_fraction:
            return min(max_iters, math.ceil(self.lr_warmup_fraction * len(train_dataloader)))
        if self.lr_warmup_steps:
            return min(max_iters, self.lr_warmup_steps * self.gradient_accumulation_iters(devices))
        return 0
@dataclass
class LLMTrainArgs(BaseTrainArgs):
    lr_warmup_steps: Optional[int] = 0
    lr_warmup_fraction: Optional[float] = None
    max_tokens: Optional[int] = None
    max_seq_length: Optional[int] = None
    tie_embeddings: Optional[bool] = None
    learning_rate: float = 1e-3
    weight_decay: float = 0.2
    beta1: float = 0.9
    beta2: float = 0.95
    max_norm: Optional[float] = None
    min_lr: float = 6e-5

    def __post_init__(self) -> None:
        if self.lr_warmup_fraction and self.lr_warmup_steps:
            raise ValueError("Can't provide both `lr_warmup_fraction` and `lr_warmup_steps`. Choose one.")
        if self.lr_warmup_fraction and not (0 <= self.lr_warmup_fraction <= 1):
            raise ValueError("`lr_warmup_fraction` must be between 0 and 1.")


# Specific dataclass for Image Classifier training arguments
@dataclass
class ImgClassifierTrainArgs(BaseTrainArgs):
    # run_training: bool = True
    # run_evaluation: bool = False
    learning_rate: float = 1e-3
    weight_decay: float = 0.
    max_epochs: Optional[int] = None
    limit_train_batches: Optional[int] = None  
    limit_val_batches: Optional[int] = None

@dataclass
class EvalArgs:
    """Evaluation-related arguments"""
    interval: int = 600
    """Number of optimizer steps between evaluation calls"""
    max_new_tokens: Optional[int] = None
    """Number of tokens to generate"""
    max_iters: int = 100
    """Number of iterations"""
    initial_validation: bool = False
    """Whether to evaluate on the validation set at the beginning of the training"""

@dataclass
class DataArgs:
    dataset_name: str
    train_data_dir: Optional[Path] = None
    val_data_dir: Optional[Path] = None


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


def prepare_args(config: DictConfig,expid):
    
    log_dir = f"{config.log_dir}/{config.dataset.name}/{config.training.model_name}/{expid}"
    if config['training'].get('max_tokens') is not None:
       
        train_args = LLMTrainArgs(
            job_id=os.getpid(),
            model_name=config.training.model_name,
            dataloader_name=config.dataloader.name,
            seed=config.seed,
            log_dir=log_dir,
            log_interval = config.log_interval,
            devices=config.num_devices_per_job,
            num_workers = config.num_workers_per_job,
            checkpoint_interval=config.checkpoint_interval,
            accelerator=config.accelerator,
            global_batch_size=config.training.global_batch_size,
            micro_batch_size=config.training.micro_batch_size,
            lr_warmup_steps=config.training.lr_warmup_steps,
            lr_warmup_fraction=config.training.lr_warmup_fraction,
            max_tokens=config.training.max_tokens,
            tie_embeddings=config.training.tie_embeddings,
            learning_rate=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            beta1=config.training.beta1,
            beta2=config.training.beta2,
            max_norm=config.training.max_norm,
            min_lr=config.training.min_lr,
            run_training = config.run_training,
            run_evaluation = config.run_evaluation,
            max_steps = config.training.max_steps,
            max_seq_length = config.training.max_seq_length,

        )
    else:
        train_args = ImgClassifierTrainArgs(
        job_id=os.getpid(),
        model_name=config.training.model_name,
        dataloader_name=config.dataloader.name,
        seed=config.seed,
        log_dir=log_dir,
        log_interval = config.log_interval,
        devices=config.num_devices_per_job,
        num_workers = config.num_workers_per_job,
        checkpoint_interval=config.checkpoint_interval,
        accelerator=config.accelerator,
        global_batch_size=config.training.global_batch_size,
        micro_batch_size=config.training.micro_batch_size,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        max_epochs=config.training.max_epochs,
        limit_train_batches = config.training.limit_train_batches,
        limit_val_batches=config.training.limit_val_batches,
        run_training = config.run_training,
        max_steps = config.training.max_steps,
        run_evaluation = config.run_evaluation)
    
    data_args = DataArgs(
        dataset_name=config.dataset.name,
        train_data_dir=config.dataset.train_dir,
        val_data_dir=config.dataset.val_dir,)

    if 'super' in train_args.dataloader_name:
        super_args = SUPERArgs(
            num_pytorch_workers=config.num_workers_per_job,
            super_address=config.dataloader.super_address,
            cache_address=config.dataloader.cache_address,
            simulate_data_delay=config.dataloader.simulate_data_delay)
        if super_args.simulate_data_delay is not None:
            train_args.dataload_only = True
        return train_args, data_args, super_args

    elif 'shade' in train_args.dataloader_name:
        shade_args = SHADEArgs(
            num_pytorch_workers=config.num_workers_per_job,
            cache_address=config.dataloader.cache_address,
            working_set_size=config.dataloader.working_set_size,
            replication_factor=config.dataloader.replication_factor)
        return train_args, data_args, shade_args

    elif 'torch_lru' in train_args.dataloader_name:
        torchlru_args = LRUTorchArgs(
            num_pytorch_workers=config.num_workers_per_job,
            cache_address=config.dataloader.cache_address,
            cache_granularity=config.dataloader.cache_granularity,
            shuffle=config.dataloader.shuffle)

        return train_args, data_args, torchlru_args


    
