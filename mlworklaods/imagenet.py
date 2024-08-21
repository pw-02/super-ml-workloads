import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig
from lightning.fabric import Fabric, seed_everything
from lightning.fabric.loggers import CSVLogger
from torchvision.models import get_model
from custom_torch_datasets.s3_mapped_dataset import S3MappedDataset
from custom_torch_samplers.batch_sampler import BatchSamplerWithID
from torch.utils.data import RandomSampler, SequentialSampler
import time
import os
from collections import OrderedDict
import numpy as np
from resource_monitor import ResourceMonitor
  
def run_imagenet(config: DictConfig):
    # Initialize TorchFabric
    fabric = Fabric(
        accelerator=config.accelerator, 
        devices=config.devices, 
        precision=config.workload.precision,
        loggers=[CSVLogger(root_dir=config.log_dir, name="", flush_logs_every_n_steps=config.log_interval)]    
    )

    seed_everything(config.seed) # instead of torch.manual_seed(...)
    
    model = get_model(name=config.workload.model_architecture, weights=None, num_classes=config.workload.num_classes)
    optimizer = optim.Adam(model.parameters(), lr=config.workload.learning_rate)
    model, optimizer = fabric.setup(model, optimizer)

    # Set up data transforms for ImageNet
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataloader = None
    val_dataloader = None

    if config.dataloader.name == 'super':
        pass
    
    elif config.dataloader.name == 'pytorch':
        # PyTorch DataLoader
        if config.workload.run_training:
            train_dataset = S3MappedDataset(s3_bucket=config.workload.s3_bucket, s3_prefix=config.workload.s3_train_prefix, transform=train_transform)

            train_sampler = BatchSamplerWithID(
                sampler= RandomSampler(data_source=train_dataset,generator=torch.Generator().manual_seed(config.seed)),
                batch_size=config.workload.batch_size, 
                drop_last=False
                )
            train_dataloader = DataLoader(train_dataset, batch_size=None, sampler=train_sampler, num_workers=config.workload.num_pytorch_workers)
            train_dataloader = fabric.setup_dataloaders(train_dataloader)
        if config.workload.run_validation:
            val_dataset = S3MappedDataset(s3_bucket=config.workload.s3_bucket, s3_prefix=config.workload.s3_val_prefix, transform=val_transform)
            val_sampler = BatchSamplerWithID(
                sampler= SequentialSampler(data_source=train_dataset),
                batch_size=config.workload.batch_size, 
                drop_last=False
                )

            val_dataloader =  DataLoader(val_dataset, batch_size=None, sampler=val_sampler, num_workers=config.workload.num_pytorch_workers)
            val_dataloader = fabric.setup_dataloaders(val_dataloader)

    # Start training
    metric_collector = ResourceMonitor(interval=1, flush_interval=10, file_path= f'{ fabric.loggers[0].log_dir}/resource_usage_metrics.json')
    metric_collector.start()
    global_step_count = 0
    current_epoch=0
    should_stop = False
    train_start_time = time.perf_counter()
    
    while not should_stop:

        current_epoch += 1

        global_step_count = train_loop(fabric, 
                                       model, 
                                       optimizer,
                                       train_dataloader, 
                                       train_start_time, 
                                       current_epoch, 
                                       global_step_count, 
                                       max_steps = config.workload.max_steps,
                                       limit_train_batches = config.workload.limit_train_batches)
        
        if val_dataloader is not None and current_epoch % config.workload.validation_frequency == 0:
            validate_loop(fabric, model, val_dataloader)

        if current_epoch % config.workload.checkpoint_frequency == 0:
            checkpoint = {'epoch': current_epoch,
                          'model_state_dict': model.state_dict(),
                          'optimizer_state_dict': optimizer.state_dict()}
            fabric.save(os.path.join(config.checkpoint_dir, f"epoch-{current_epoch:04d}.ckpt"), checkpoint)
        
        if config.workload.max_steps is not None and global_step_count >= config.workload.max_steps:
            should_stop = True
        if config.workload.max_epochs is not None and current_epoch >= config.workload.max_epochs:
            should_stop = True

    elapsed_time = time.perf_counter() - train_start_time
    fabric.print(f"Training completed in {elapsed_time:.2f} seconds")
    metric_collector.stop()


def train_loop(fabric, model, optimizer, train_dataloader, train_start_time, current_epoch, global_step_count, max_steps = None, limit_train_batches = np.inf, criterion=nn.CrossEntropyLoss()):
    model.train()
    end = time.perf_counter()

    for batch_idx, (batch, data_fetch_time, transformation_time, cache_hit) in enumerate(train_dataloader):
        data_load_time = time.perf_counter() - end
                        # end epoch if stopping training completely or max batches for this epoch reached
        if batch_idx >= limit_train_batches:   
            break
        inputs, labels = batch

        # Synchronize to ensure previous GPU operations are finished
        torch.cuda.synchronize()

        # Forward pass
        gpu_processing_started = time.perf_counter()
        outputs  = model(inputs)
        loss = criterion(outputs, labels)
        train_acc = (outputs.argmax(dim=1) == labels).float().mean().item()
        train_loss = loss.item()

        # Backpropagation and optimization
        fabric.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()
        gpu_processing_time = time.perf_counter() - gpu_processing_started
        global_step_count +=1
        metrics= OrderedDict({
                        "Elapsed Time (s)": time.perf_counter() - train_start_time,
                        "Device": fabric.global_rank,
                        "Epoch Index": current_epoch,
                        "Batch Index": batch_idx+1,
                        "Batch Size": batch[0].size(0),
                        "Total Iteration Time (s)": time.perf_counter() - end,
                        "Data Loading Time (s)": data_load_time,
                        "GPU Processing Time (s)": gpu_processing_time,
                        "Data Fetch Time (s)": data_fetch_time,
                        "Transformation Time (s)": transformation_time,
                        "Cache Hit/Miss": 1 if cache_hit else 0,
                        # "cpu (%)": get_cpu_usage(),
                        # "gpu (%)": get_gpu_usage(),
                        "Train Loss": train_loss,
                        "Train Accuracy": train_acc,
                        "Validation  Loss": 0,
                        "Validation  Accuracy": 0,
                        })
        fabric.log_dict(metrics,step=global_step_count)
        
        fabric.print(
                f"Epoch {metrics['Epoch Index']} ({metrics['Batch Index']}/{min(len(train_dataloader),limit_train_batches)}) |"
                # f" loss train: {metrics['Train Loss']:.3f} |"
                # f" val: {val_loss} |"
                f" iter time: {metrics['Total Iteration Time (s)']:.2f} |"
                f" dataload time: {metrics['Data Loading Time (s)']:.2f} |"
                f" gpu time: {metrics['GPU Processing Time (s)']:.2f} |"
                # f" fetch time: {metrics['Data Fetch Time (s)']:.2f} |"
                # f" transform time: {metrics['Transformation Time (s)']:.2f} |"
                # f" CPU Usage (%): {metrics['CPU Usage (%)']:.2f} |"
                # f" GPU Usage (%): {metrics['GPU Usage (%)']:.2f} |"
                f" elapsed time: {metrics['Elapsed Time (s)']:.2f} |"

                )

        # stopping criterion on step level
        if max_steps is not None and global_step_count >= max_steps:
            break

        end = time.perf_counter()

    return  global_step_count


def validate_loop(fabric, model, dataloader):
    # Validation
    model.eval()
    for i, batch in enumerate(dataloader):
        with torch.no_grad():
            output = model(batch)
            # Do something with output

@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(config: DictConfig):
    run_imagenet(config)


if __name__ == "__main__":
    main()
