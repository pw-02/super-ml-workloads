
from omegaconf import DictConfig
import hydra
from mlworklaods.args import *
from mlworklaods.log_utils import  get_next_exp_version
import torch.multiprocessing as mp 
from torch.multiprocessing import Pool, Process, set_start_method 
from typing import List
from typing import Dict, Any
import os
import time
from mlworklaods.image_classification.image_classifer_trainer import MyCustomTrainer
import torch
from image_classification.imager_classifer_model import ImageClassifierModel
# Helper function to prepare arguments for a job
from torch_lru.batch_sampler_with_id import BatchSamplerWithID
from torch_lru.torch_lru_dataset import TorchLRUDataset
from mlworklaods.common import transform
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler  
from super_dl.dataset.super_dataset import SUPERDataset
from mlworklaods.image_classification.callbacks import LoggingCallback
from mlworklaods.log_utils import ExperimentLogger
from lightning.fabric.loggers import CSVLogger


def prepare_args(config: DictConfig):
    log_dir = f"{config.log_dir}/{config.dataset.name}"
    # exp_version = get_next_exp_version(log_dir_base, config.dataloader.kind)
    # full_log_dir = os.path.join(log_dir_base, config.dataloader.kind, str(exp_version))

    train_args = TrainArgs(
        job_id=os.getpid(),
        model_name=config.training.model_name,
        max_steps=config.training.max_steps,
        max_epochs=config.training.max_epochs,
        limit_train_batches=config.training.limit_train_batches,
        limit_val_batches=config.training.limit_val_batches,
        batch_size=config.training.batch_size,
        grad_accum_steps=config.training.grad_accum_steps,
        run_training=config.run_training,
        run_evaluation=config.run_evaluation,
        # validation_frequency = config.training.validation_frequency,
        devices=config.num_devices_per_job,
        accelerator=config.accelerator,
        seed=config.seed,
        log_dir=log_dir,
        log_freq=config.log_freq,
        dataloader_kind=config.dataloader.kind,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay)
    



    data_args = DataArgs(
        train_data_dir=config.dataset.train_dir,
        val_data_dir=config.dataset.val_dir,
        num_classes=config.dataset.num_classes
    )

    if 'super' in train_args.dataloader_kind:
        super_args = SUPERArgs(
            num_pytorch_workers=config.training.num_pytorch_workers,
            super_address=config.dataloader.super_address,
            cache_address=config.dataloader.cache_address,
            simulate_data_delay=config.dataloader.simulate_data_delay)
        if super_args.simulate_data_delay is not None:
            train_args.dataload_only = True
        return train_args, data_args, super_args

    elif 'shade' in train_args.dataloader_kind:
        shade_args = SHADEArgs(
            num_pytorch_workers=config.training.num_pytorch_workers,
            cache_address=config.dataloader.cache_address,
            working_set_size=config.dataloader.working_set_size,
            replication_factor=config.dataloader.replication_factor)
        return train_args, data_args, shade_args

    elif 'torch_lru' in train_args.dataloader_kind:
        torchlru_args = LRUTorchArgs(
            num_pytorch_workers=config.training.num_pytorch_workers,
            cache_address=config.dataloader.cache_address,
            cache_granularity=config.dataloader.cache_granularity,
            shuffle=config.dataloader.shuffle)
        
        return train_args, data_args, torchlru_args


def make_lru_torch_datalaoders(train_args: TrainArgs, data_args: DataArgs, lru_torch_args:LRUTorchArgs, world_size:int):
    train_dataloader = None
    val_dataloader = None
    if train_args.run_training:
        train_dataset =  TorchLRUDataset(
            data_dir = data_args.train_data_dir,
            transform=transform(),
            cache_address=lru_torch_args.cache_address,
            cache_granularity=lru_torch_args.cache_granularity)
          
        train_base_sampler = RandomSampler(data_source=train_dataset) if lru_torch_args.shuffle else SequentialSampler(data_source=train_dataset)
        train_batch_sampler = BatchSamplerWithID(sampler=train_base_sampler, batch_size=train_args.get_batch_size(world_size), drop_last=False)  
        train_dataloader = DataLoader(dataset=train_dataset, sampler=train_batch_sampler, batch_size=None, num_workers=lru_torch_args.num_pytorch_workers)

    if train_args.run_evaluation:
        val_dataset =  TorchLRUDataset(
            data_dir = data_args.val_data_dir,
            transform=transform(),
            cache_address=lru_torch_args.cache_address,
            cache_granularity=lru_torch_args.cache_granularity)
        
        val_base_sampler = RandomSampler(data_source=val_dataset) if lru_torch_args.shuffle else SequentialSampler(data_source=val_dataset)
        val_batch_sampler = BatchSamplerWithID(sampler=val_base_sampler, batch_size=train_args.get_batch_size(world_size), drop_last=False)
        
        val_dataloader = DataLoader(dataset=val_dataset, sampler=val_batch_sampler, batch_size=None, num_workers=lru_torch_args.num_pytorch_workers)
    return train_dataloader, val_dataloader


# Dataloader creation function
def make_super_dataloaders(train_args: TrainArgs, data_args: DataArgs, super_args:SUPERArgs, world_size:int):
    train_dataloader = None
    val_dataloader = None
    if train_args.run_training:  
        dataset = SUPERDataset(
            job_id=train_args.job_id,
            data_dir=data_args.train_data_dir,
            batch_size=train_args.get_batch_size(world_size),
            transform=transform(),
            world_size=world_size,
            super_address=super_args.super_address,
            cache_address=super_args.cache_address,
            simulate_delay=super_args.simulate_data_delay)
        
        train_dataloader = DataLoader(dataset=dataset, batch_size=None, num_workers=super_args.num_pytorch_workers)

    if train_args.run_evaluation:
         dataset = SUPERDataset(
            job_id=train_args.job_id,
            data_dir=data_args.val_data_dir,
            batch_size=train_args.get_batch_size(world_size),
            transform=transform(),
            world_size=world_size,
            super_address=super_args.super_address,
            cache_address=super_args.cache_address,
            simulate_delay=super_args.simulate_data_delay)
    
    return train_dataloader, val_dataloader


def get_default_supported_precision(training: bool) -> str:
    from lightning.fabric.accelerators import MPSAccelerator
    if MPSAccelerator.is_available() or (torch.cuda.is_available() and not torch.cuda.is_bf16_supported()):
        return "16-mixed" if training else "16-true"
    return "bf16-mixed" if training else "bf16-true"


@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(config: DictConfig):

    train_args, data_args, dataloader_args = prepare_args(config)

    model = ImageClassifierModel(train_args.model_name, train_args.learning_rate, num_classes = data_args.num_classes)
    
    logger = CSVLogger(root_dir=train_args.log_dir, name=train_args.model_name,flush_logs_every_n_steps=train_args.log_freq)

    trainer = MyCustomTrainer(
        accelerator=train_args.accelerator,
        precision=get_default_supported_precision(True),
        devices=train_args.devices, 
        limit_train_batches=train_args.limit_train_batches, 
        limit_val_batches=train_args.limit_val_batches, 
        max_epochs=train_args.max_epochs,
        max_steps=train_args.max_steps,
        loggers=[logger],
        grad_accum_steps=train_args.grad_accum_steps
    )

    if 'super' in train_args.dataloader_kind:
         train_loader, val_loader = make_super_dataloaders(train_args, data_args,dataloader_args, trainer.fabric.world_size)
    elif 'torch_lru' in train_args.dataloader_kind:
         train_loader, val_loader = make_lru_torch_datalaoders(train_args, data_args,dataloader_args, trainer.fabric.world_size)
    else:
        raise Exception(f"Unknown dataloader_kind {train_args.dataloader_kind}")
    
    trainer.fit(model, train_loader, val_loader,train_args.seed)

    






   
if __name__ == "__main__":
    main()