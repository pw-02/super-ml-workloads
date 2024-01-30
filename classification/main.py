import time
from typing import Iterator
import redis
from torch import nn, optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
from jsonargparse._namespace import Namespace
import sys
from torch.utils.data.dataloader import default_collate
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

    create_job_report(hparams.reporting.exp_name,logger.log_dir)

    fabric.print(f"Experiment ended. Duration: {exp_duration}")


def custom_collate(batch):
    imgs, labels, indices, fetch_times, transform_times = zip(*batch)

    # Convert images and labels to tensors using default_collate
    img_tensor = default_collate(imgs)
    label_tensor = default_collate(labels)
    
    total_fetch_time = sum(list(fetch_times))
    total_transform_time = sum(list(transform_times))

    # Convert other information to tensors if needed
    batch_id = abs(hash(tuple(indices)))

    return img_tensor, label_tensor, batch_id, False,  total_fetch_time, total_transform_time


def prepare_for_training(fabric: Fabric,hparams:Namespace):
    # Set seed
    if hparams.workload.training_seed is not None:
        fabric.seed_everything(hparams.workload.training_seed, workers=True)
    
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

    # Initialize cache and super
    cache_client = redis.StrictRedis(host=hparams.data.cache_host, port=hparams.data.cache_port) if hparams.data.use_cache else None    
    super_client = SuperClient(hparams.data.super_address) if hparams.data.dataloader_backend == 'super' else None
    
    # Initialize dataloaders
    eval_dataloader = None
    train_dataloader = None

    if hparams.workload.run_training:
        train_dataloader = initialize_dataloader(
            fabric=fabric,
            hparams=hparams,
            is_training=True,
            cache_client=cache_client,
            super_client=super_client
        )
        train_dataloader = fabric.setup_dataloaders(train_dataloader)
    
    if hparams.workload.run_evaluate:
        eval_dataloader = initialize_dataloader(
            fabric=fabric,
            hparams=hparams,
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
    logger = SUPERLogger( fabric=fabric, root_dir=hparams.reporting.report_dir,
                          flush_logs_every_n_steps=hparams.reporting.flush_logs_every_n_steps,
                          print_freq= hparams.reporting.print_freq,
                          exp_name=hparams.reporting.exp_name)
   
    
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


def initialize_dataloader(fabric: Fabric, hparams:Namespace, is_training = False, cache_client = None, super_client = None):

    transformations = initialize_transformations(hparams.data.train_data_dir if is_training else hparams.data.eval_data_dir)

    dataset = initialize_dataset(
        dataloader_backend=hparams.data.dataloader_backend,
        transformations=transformations,
        data_dir=hparams.data.train_data_dir if is_training else hparams.data.eval_data_dir,
        cache_client=cache_client,
        super_client=super_client
        )
    
    fabric.print(f"Dataset initialized: {hparams.data.train_data_dir if is_training else hparams.data.eval_data_dir}, size: {len(dataset)} files")

    sampler = initialize_sampler(
        dataset, 
        hparams.data.dataloader_backend,
        hparams.job_id, 
        super_client,
        shuffle=hparams.data.shuffle,
        batch_size=hparams.data.batch_size,
        drop_last=hparams.data.drop_last,
        prefetch_lookahead=hparams.data.super_prefetch_lookahead,
        sampler_seed=hparams.data.sampler_seed)
        
    if  hparams.data.dataloader_backend == "pytorch-vanillia":
        return DataLoader(dataset=dataset, sampler=sampler, batch_size=hparams.data.batch_size, 
                          num_workers=hparams.workload.workers, collate_fn=custom_collate)
    else:
        return DataLoader(dataset=dataset, sampler=sampler, batch_size=None, num_workers=hparams.workload.workers)


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
    

def initialize_sampler(dataset, dataloader_backend, job_id, super_client,shuffle,sampler_seed, batch_size, drop_last,prefetch_lookahead):
    
    if dataloader_backend == "super":
      
      return SUPERSampler(
        data_source=dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        job_id=job_id,
        shuffle=shuffle,
        seed=sampler_seed,
        super_client=super_client,
        prefetch_lookahead=prefetch_lookahead)
    
    elif dataloader_backend == "pytorch-batch":
        return PytorchBatchSampler(
        data_source=dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        shuffle=shuffle,
        seed=sampler_seed
        )
    
    elif dataloader_backend == "pytorch-vanillia":
        return PytorchVanilliaSampler(
        data_source=dataset,
        shuffle=shuffle,
        seed=sampler_seed
        )


def initialize_dataset( dataloader_backend:str, transformations: transforms.Compose,data_dir:str,cache_client=None,super_client=None, ):
    
    if dataloader_backend == "super":
        return SUPERDataset(data_dir=data_dir,transform=transformations,cache_client=cache_client,super_client=super_client)
    
    elif dataloader_backend == "pytorch-batch":
        return PytorchBatchDataset(data_dir=data_dir,transform=transformations)
    
    elif dataloader_backend == "pytorch-vanillia":
        return PytorchVanilliaDataset(data_dir=data_dir,transform=transformations)
    

def initialize_transformations(data_dir) -> transforms.Compose:
    
    if 'resnet' in data_dir.lower():
        #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transformations =  transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transformations = transforms.Compose([transforms.ToTensor(), normalize])
    
    return transformations

if __name__ == "__main__":
    pass