#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

from enum import Enum
from pathlib import Path
from typing import Optional, List, Any
import hydra
from hydra.core.hydra_config import HydraConfig
from numpy import Infinity
from omegaconf import DictConfig
from lightning.fabric import Fabric
import time
from torch import nn, optim
from mlworklaods.utils import get_default_supported_precision
from torch.utils.data import DataLoader
from mlworklaods.models import ModelInterface, EmptyModel, TorchVisionModel
import torchvision
from mlworklaods.super_dl.datasets.s3mapdataset import S3MapDataset
from torch.utils.data import Sampler, SequentialSampler, RandomSampler,Dataset         
from logutil import ExperimentLogger, create_exp_summary_report
import os

# from s3torchconnector import S3IterableDataset, S3Reader, S3MapDataset

@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_experiment(config: DictConfig):
   
    start_time = time.perf_counter()

    precision = get_default_supported_precision(training=True)
    fabric = Fabric(accelerator=config.accelerator, devices=config.devices, strategy="auto", precision=precision)
    exp_version = get_next_exp_version(config.log_dir,config.dataset.name)
    config.log_dir = os.path.join(config.log_dir, config.dataset.name, str(exp_version))

    if not config.training.max_minibatches_per_epoch:
         config.training.max_minibatches_per_epoch = Infinity
   

    result = fabric.launch(main, config=config)
    
    fabric.print(f"Creating overall report for experiment")

    create_exp_summary_report(config.log_dir)

    fabric.print(f"Exeperiment completed. Total Duration: {time.perf_counter() - start_time}")


def main(fabric: Fabric, config: DictConfig):
    # Load model
    t0 = time.perf_counter()
    model = make_model(fabric, config)
    fabric.print(f"Time to instantiate {model.name} model: {time.perf_counter() - t0:.02f} seconds")
    fabric.print(f"Total parameters in {model.name} model: {model.num_parameters:,}")
    train_dataloader = None
    val_dataloader = None
    best_acc1 = 0
    best_acc5 = 0

    if config.run_training:
        train_dataset = make_dataset(dataloader_kind=config.dataloader.kind,data_dir=config.dataset.train_dir, transform=model.transform)
        train_sampler = make_sampler(sampler_kind =config.dataloader.sampler_kind, shuffle=config.dataloader.shuffle, data_source=train_dataset)
        train_dataloader = make_dataloader(dataset=train_dataset,sampler=train_sampler, num_workers=config.dataloader.num_workers,batch_size=config.dataloader.batch_size)
        train_dataloader = fabric.setup_dataloaders(train_dataloader, move_to_device=True, use_distributed_sampler=False)
    
    if config.run_evaluation:
        val_dataset = make_dataset(dataloader_kind=config.dataloader.kind,data_dir=config.dataset.val_dir, transform=model.transform)
        val_sampler = make_sampler(sampler_kind =config.dataloader.sampler_kind, shuffle=False, data_source=val_dataset)
        val_dataloader = make_dataloader(dataset=val_dataset,sampler=val_sampler, num_workers=config.dataloader.num_workers,batch_size=config.dataloader.batch_size)
        val_dataloader = fabric.setup_dataloaders(val_dataloader, move_to_device=True, use_distributed_sampler=False)

    logger = ExperimentLogger(fabric, config.log_dir, config.log_freq, config.print_freq)

    if fabric.is_global_zero:
        logger.log_hyperparams(config)
    

    start_time = time.perf_counter()
    for epoch in range(0, config.training.max_epochs):
        if config.run_training and train_dataloader:
            fabric.print(f"Stating training loop for epoch {epoch}")
            acc1,  acc5 = model.train(
                epoch=epoch,
                train_dataloader=train_dataloader,
                logger=logger,
                max_minibatches=config.training.max_minibatches_per_epoch,
                grad_acc_steps=config.training.grad_acc_steps
                )
                      # remember best acc@1 and acc@5
            best_acc1 = max(acc1, best_acc1)
            best_acc5 = max(acc5, best_acc5)
        
        if config.run_evaluation and val_dataloader:
            fabric.print(f"Stating validation loop for epoch {epoch}")
            acc1,  acc5 = model.validate(
                epoch=epoch,
                val_dataloader=val_dataloader,
                logger=logger,
                max_minibatches=config.training.max_minibatches_per_epoch,
                )
            
            # remember best acc@1 and acc@5
            best_acc1 = max(acc1, best_acc1)
            best_acc5 = max(acc5, best_acc5)

                  
    training_time = time.perf_counter() - start_time

    fabric.print(f"Training Finished on device {fabric.global_rank}. Total Duration: {training_time}")
    fabric.print(f"Creating job report for device {fabric.global_rank}..")

    logger.create_job_report()

    return training_time

def make_dataset(dataloader_kind: str, data_dir: str, transform)->Dataset:
    if dataloader_kind == 's3mapdataset':
         dataset =  S3MapDataset(data_dir = data_dir, transform=transform)
    elif dataloader_kind == 'supermapdataset':
         pass
    else:
        raise Exception(f"unknown dataset kind {dataloader_kind}")
    return dataset

def make_sampler(sampler_kind: str, shuffle: bool, data_source)->Sampler:
    if sampler_kind == 'classic_pytorch':
        sampler = RandomSampler(data_source=data_source) if shuffle else SequentialSampler(data_source=data_source)
    else:
        raise Exception(f"unknown dataset kind {sampler_kind}")
    return sampler

def make_dataloader(dataset: Dataset, sampler:Sampler, num_workers: int, batch_size: int):
    dataloader = DataLoader(dataset=dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers)
    return dataloader

def make_model(fabric: Fabric, config: DictConfig)-> ModelInterface:

    if config.training.model == "emptymodel":
        return EmptyModel()
    elif config.training.model in torchvision.models.list_models():
        return TorchVisionModel(fabric=fabric, model_name= config.training.model)
    else:
        raise Exception(f"unknown model {config.training.model}")

def get_next_exp_version(root_dir, name):
    from lightning.fabric.utilities.cloud_io import _is_dir, get_filesystem
    versions_root = os.path.join(root_dir, name)



    fs = get_filesystem(root_dir)
    if not _is_dir(fs, versions_root, strict=True):
            #log.warning("Missing logger folder: %s", versions_root)
            fs.makedirs(versions_root, exist_ok=True)

            f"version_{0}"
    
    existing_versions = []
    for d in fs.listdir(versions_root):
        full_path = d["name"]
        name = os.path.basename(full_path)
        if _is_dir(fs, full_path) and name.startswith("version_"):
            dir_ver = name.split("_")[1]
            if dir_ver.isdigit():
                existing_versions.append(int(dir_ver))
    if len(existing_versions) == 0:
        return f"version_{0}"
    return f"version_{max(existing_versions) + 1 }" 


if __name__ == "__main__":
    run_experiment()