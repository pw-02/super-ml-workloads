import time
from typing import Iterator
import redis
from torch import nn, optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
from jsonargparse._namespace import Namespace
import sys

from lightning.fabric import Fabric
from super_client import SuperClient

from image_classification.utils import *
from image_classification.datasets import *
from image_classification.samplers import *
from image_classification.training import *


def main(fabric: Fabric,hparams:Namespace) -> None:
    exp_start_time = time.time()
   
    # Prepare for training
    model, optimizer, scheduler, train_dataloader, val_dataloader, logger, super_client = prepare_for_training(
        fabric=fabric,hparams=hparams)
        
    logger.log_hyperparams(hparams)

    # Run training
    run_training(
        fabric=fabric,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        hparams=hparams,
        logger=logger,
        super_client=super_client
    )

    exp_duration = time.time() - exp_start_time

    create_job_report(hparams.workload.exp_name,logger.log_dir)

    fabric.print(f"Experiment ended. Duration: {exp_duration}")


def prepare_for_training(fabric: Fabric,hparams:Namespace):
    # Set seed
    if hparams.workload.seed is not None:
        fabric.seed_everything(hparams.workload.seed, workers=True)
    
    #Load model
    t0 = time.perf_counter()
    model = initialize_model(fabric, hparams.model.arch)
    fabric.print(f"Time to instantiate {hparams.model.arch} model: {time.perf_counter() - t0:.02f} seconds")
    fabric.print(f"Total parameters in {hparams.model.arch} model: {num_model_parameters(model):,}")

    #Initialize loss, optimizer and scheduler
    optimizer =  initialize_optimizer(optimizer_type = hparams.model.optimizer,  model_parameters=model.parameters(),learning_rate=hparams.model.lr, momentum=hparams.model.momentum, weight_decay=hparams.model.weight_decay)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.1 ** (epoch // 30)) #TODO: Add support for other scheduler
     # call `setup` to prepare for model / optimizer for distributed training. The model is moved automatically to the right device.
    model, optimizer = fabric.setup(model,optimizer, move_to_device=True) 
    #Initialize data transformations
    transformations = initialize_transformations()

    # Initialize cache and super
    cache_client = redis.StrictRedis(host=hparams.super_dl.cache_host, port=hparams.super_dl.cache_port) if hparams.super_dl.use_cache else None    
    super_client = SuperClient(hparams.super_dl.server_address) if hparams.data.dataloader_backend == 'super' else None
    
    # Initialize dataloaders
    eval_dataloader = None
    train_dataloader = None

    if hparams.workload.run_training:
        train_dataloader = initialize_dataloader(
            hparams=hparams,
            transformations=transformations,
            is_training=True,
            cache_client=cache_client,
            super_client=super_client
        )
        train_dataloader = fabric.setup_dataloaders(train_dataloader)
    
    if hparams.workload.run_evaluate:
        eval_dataloader = initialize_dataloader(
            hparams=hparams,
            transformations=transformations,
            is_training=False,
            cache_client=cache_client,
            super_client=super_client
        )
        eval_dataloader = fabric.setup_dataloaders(eval_dataloader)

    #register job with super
    if hparams.data.dataloader_backend == 'super' and super_client is not None:
        register_job_with_super(
            super_client=super_client,
            job_id=   hparams.job_id,
            train_dataset= None if train_dataloader is None else train_dataloader.dataset,
            evaluation_dataset= None if eval_dataloader is None else eval_dataloader.dataset)


    #Initialize logger
    logger = SUPERLogger( fabric=fabric, root_dir=hparams.workload.log_dir,
                          flush_logs_every_n_steps=hparams.workload.flush_logs_every_n_steps,
                          print_freq= hparams.workload.print_freq,
                          exp_name=hparams.workload.exp_name)
   
    
    return model, optimizer, scheduler, train_dataloader, eval_dataloader, logger, super_client

def initialize_model(fabric: Fabric, arch: str) -> nn.Module: 
    with fabric.init_module(empty_init=True): #model is instantiated with randomly initialized weights by default.
        model: nn.Module = models.get_model(arch)
    return model

def initialize_optimizer(optimizer_type:str, model_parameters:Iterator[nn.Parameter], learning_rate, momentum, weight_decay):
    if optimizer_type == "sgd":
        optimizer = optim.SGD(params=model_parameters, 
                              lr=learning_rate, 
                              momentum=momentum, 
                              weight_decay=weight_decay)
    elif optimizer_type == "rmsprop":
        optimizer = optim.RMSprop(params=model_parameters, 
                              lr=learning_rate,
                              momentum=momentum, 
                              weight_decay=weight_decay)
    return optimizer


def initialize_dataloader(hparams:Namespace, transformations, is_training = False, cache_client = None, super_client = None):
    
    dataset = initialize_dataset(
        dataloader_backend=hparams.data.dataloader_backend,
        transformations=transformations,
        data_dir=hparams.data.train_data_dir if is_training else hparams.data.eval_data_dir,
        source_system=hparams.super_dl.source_system,
        s3_bucket_name=hparams.data.s3_bucket_name,
        cache_client=cache_client,
        super_client=super_client
        )
    
    sampler = initialize_sampler(
        dataset, 
        hparams.job_id, 
        super_client,
        shuffle=hparams.data.shuffle,
        batch_size=hparams.data.batch_size,
        drop_last=hparams.data.drop_last,
        prefetch_lookahead=hparams.super_dl.prefetch_lookahead)
        
    return DataLoader(dataset, sampler=sampler, batch_size=None, num_workers=hparams.pytorch.workers)


def register_job_with_super(super_client: SuperClient, job_id, train_dataset:SUPERDataset, evaluation_dataset:SUPERDataset):
    #dataset_id = hashlib.sha256(f"{data_source_system}_{data_dir}".encode()).hexdigest()
    job_dataset_ids = []
    if train_dataset is not None:
        super_client.register_dataset(train_dataset.dataset_id, train_dataset.data_dir, train_dataset.source_system, None)
        job_dataset_ids.append(train_dataset.dataset_id)
    if evaluation_dataset is not None:
        super_client.register_dataset(evaluation_dataset.dataset_id, evaluation_dataset.data_dir, evaluation_dataset.source_system, None)
        job_dataset_ids.append(evaluation_dataset.dataset_id)

    super_client.register_new_job(job_id=job_id,job_dataset_ids=job_dataset_ids)
    
    return super_client
    

def initialize_sampler(dataset, job_id,super_client,shuffle, batch_size, drop_last,prefetch_lookahead):
    
    return SUPERSampler(
        dataset=dataset,
        job_id=job_id,
        super_client=super_client,
        shuffle=shuffle,
        seed=1,
        batch_size=batch_size,
        drop_last=drop_last,
        prefetch_lookahead=prefetch_lookahead
    )


def initialize_dataset( 
                        dataloader_backend:str, 
                        transformations: transforms.Compose,
                        data_dir:str,
                        source_system:str,
                        s3_bucket_name:str,
                        cache_client=None,
                        super_client=None,

                        ):
    

    if dataloader_backend == "super":
        return SUPERDataset(
                data_dir=data_dir,
                transform=transformations,
                cache_client=cache_client,
                source_system=source_system,
                s3_bucket_name=s3_bucket_name,
                super_client=super_client
                )
    

def initialize_transformations() -> transforms.Compose:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transformations = transforms.Compose([transforms.ToTensor(), normalize])
    return transformations

if __name__ == "__main__":
    pass