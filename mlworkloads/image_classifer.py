import torch
import sys
print(sys.path)
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig
from lightning.fabric import Fabric, seed_everything
from lightning.fabric.loggers import CSVLogger
from torchvision.models import get_model
from datalaoding.s3.s3_mapped_dataset import S3MappedDataset
from datalaoding.super.super_sampler import SUPERSampler
from datalaoding.super.super_mapped_dataset import SUPERMappedDataset
from datalaoding.s3.batch_sampler import BatchSamplerWithID
# from mlworkloads.datalaoding.s3.s3_mapped_dataset import S3MappedDataset
# from mlworkloads.datalaoding.super.super_sampler import SUPERSampler
# from mlworkloads.datalaoding.super.super_mapped_dataset import SUPERMappedDataset
# from mlworkloads.datalaoding.s3.batch_sampler import BatchSamplerWithID
from torch.utils.data import RandomSampler, SequentialSampler
import time
import os
from collections import OrderedDict
import numpy as np
from resource_monitor import ResourceMonitor
import datetime
from lightning.pytorch.core.saving import save_hparams_to_yaml

def train_image_classifer(config: DictConfig):
    # Initialize TorchFabric
    # timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # log_dir = f"{config.log_dir}/{config.workload.name.replace('/','_')}/{timestamp}".lower()
    log_dir = f"{config.log_dir}/{config.workload.name.replace('/','_')}/{config.exp_id}/{config.job_id}".lower()
    # os.makedirs(log_dir, exist_ok=True)
    train_logger = CSVLogger(root_dir=log_dir, name="train", prefix='', flush_logs_every_n_steps=config.log_interval)
    val_logger = CSVLogger(root_dir=log_dir, name="val", prefix='', flush_logs_every_n_steps=config.log_interval)

    fabric = Fabric(
        accelerator=config.accelerator, 
        devices=config.devices, 
        precision=config.workload.precision   
    )

    seed_everything(config.seed) # instead of torch.manual_seed(...)
    
    model = get_model(name=config.workload.model_architecture, weights=None, num_classes=config.workload.num_classes)
    optimizer = optim.Adam(model.parameters(), lr=config.workload.learning_rate)
    model, optimizer = fabric.setup(model, optimizer)
    
    train_transform, val_transform = get_transforms(config.workload.name)

    train_dataloader = None
    val_dataloader = None

    if config.dataloader.name == 'super':
        if config.workload.run_training:
            train_dataset = SUPERMappedDataset(s3_data_dir=config.workload.s3_train_prefix, transform=train_transform)

            train_sampler = SUPERSampler(
                dataset=train_dataset,
                grpc_server_address=config.dataloader.grpc_server_address,
                batch_size=config.workload.batch_size
                )
            train_dataloader = DataLoader(train_dataset, batch_size=None, sampler=train_sampler, num_workers=config.workload.num_pytorch_workers)
            train_dataloader = fabric.setup_dataloaders(train_dataloader, move_to_device=True)
        
        if config.workload.run_validation:
            val_dataset = SUPERMappedDataset(s3_data_dir=config.workload.s3_val_prefix, transform=val_transform)

            val_sampler = SUPERSampler(
                dataset=val_dataset,
                grpc_server_address=config.dataloader.grpc_server_address,
                batch_size=config.workload.batch_size
                )
            val_dataloader =  DataLoader(val_dataset, batch_size=None, sampler=val_sampler, num_workers=config.workload.num_pytorch_workers)
            val_dataloader = fabric.setup_dataloaders(val_dataloader,move_to_device=True)


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
            train_dataloader = fabric.setup_dataloaders(train_dataloader, move_to_device=True)
        
        if config.workload.run_validation:
            val_dataset = S3MappedDataset(s3_bucket=config.workload.s3_bucket, s3_prefix=config.workload.s3_val_prefix, transform=val_transform)
            val_sampler = BatchSamplerWithID(
                sampler= SequentialSampler(data_source=val_dataset),
                batch_size=config.workload.batch_size, 
                drop_last=False
                )

            val_dataloader =  DataLoader(val_dataset, batch_size=None, sampler=val_sampler, num_workers=config.workload.num_pytorch_workers)
            val_dataloader = fabric.setup_dataloaders(val_dataloader,move_to_device=True)

    # Start training
    metric_collector = ResourceMonitor(interval=1, flush_interval=10, file_path= f'{log_dir}/resource_usage_metrics.json')
    metric_collector.start()
    global_train_step_count = 0
    global_val_step_count = 0
    current_epoch=0
    should_stop = False
    train_start_time = time.perf_counter()
    
    while not should_stop:

        current_epoch += 1

        global_train_step_count = train_loop(fabric, 
                                      config.job_id,
                                       train_logger,
                                       model, 
                                       optimizer,
                                       train_dataloader, 
                                       train_start_time, 
                                       current_epoch, 
                                       global_train_step_count, 
                                       max_steps = config.workload.max_steps,
                                       limit_train_batches = config.workload.limit_train_batches)
        
        if val_dataloader is not None and current_epoch % config.workload.validation_frequency == 0:
            global_val_step_count=  validate_loop(fabric, 
                                        config.job_id,
                                        val_logger,
                                       model, 
                                       val_dataloader,
                                       train_start_time, 
                                       current_epoch, 
                                       global_val_step_count, 
                                       config.workload.limit_val_batches)
  
        if current_epoch % config.workload.checkpoint_frequency == 0:
            checkpoint = {'epoch': current_epoch,
                          'model_state_dict': model.state_dict(),
                          'optimizer_state_dict': optimizer.state_dict()}
            fabric.save(os.path.join(config.checkpoint_dir, f"epoch-{current_epoch:04d}.ckpt"), checkpoint)
        
        if config.workload.max_steps is not None and global_train_step_count >= config.workload.max_steps:
            should_stop = True
        if config.workload.max_epochs is not None and current_epoch >= config.workload.max_epochs:
            should_stop = True

    elapsed_time = time.perf_counter() - train_start_time
    fabric.print(f"Training completed in {elapsed_time:.2f} seconds")
    metric_collector.stop()
    save_hparams_to_yaml(os.path.join(log_dir, "hparms.yaml"), config)



def get_transforms(workload_name):
    if workload_name == 'ResNet50/ImageNet':
        # Set up data transforms for ImageNet
        train_transform = transforms.Compose([
            transforms.Resize(256), 
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
    elif workload_name == 'ResNet18/Cifar-10':
          # Set up data transforms for ImageNet
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),    # Random crop with padding
            transforms.RandomHorizontalFlip(),        # Random horizontal flip
            transforms.ToTensor(),                    # Convert to tensor
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])  # Normalize
        ])
        
        val_transform = transforms.Compose([
            transforms.ToTensor(),                    # Convert to tensor
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])  # Normalize
       ])
    else:
        raise ValueError(f"Invalid workload: {workload_name}")
    return train_transform, val_transform

def train_loop(fabric:Fabric, job_id, train_logger:CSVLogger, model, optimizer, train_dataloader, train_start_time, current_epoch, global_step_count, max_steps = None, limit_train_batches = np.inf, criterion=nn.CrossEntropyLoss()):
    model.train()

    total_samples = 0
    total_train_loss = 0.0
    correct_preds = 0

    end = time.perf_counter()
    for batch_idx, (batch, data_fetch_time, transformation_time, cache_hit) in enumerate(train_dataloader):

        data_load_time = time.perf_counter() - end
                        # end epoch if stopping training completely or max batches for this epoch reached
        if batch_idx >= limit_train_batches:   
            break
        inputs, labels = batch

        # Synchronize to ensure previous GPU operations are finished
        # Forward pass
        gpu_processing_started = time.perf_counter()
        outputs  = model(inputs)
        loss = criterion(outputs, labels)
        train_loss = loss.item()
        # Backpropagation and optimization
        optimizer.zero_grad()  # Clear previous gradients
        fabric.backward(loss)  # Perform backpropagation
        optimizer.step()  # Update weights
         # Synchronize after the backward pass and optimization
        if fabric.device.type == 'gpu':
            torch.cuda.synchronize()
        gpu_processing_time = time.perf_counter() - gpu_processing_started

        train_acc = (outputs.argmax(dim=1) == labels).float().mean().item()
    

        # Metrics calculation
        total_train_loss += train_loss * inputs.size(0)  # accumulate total loss
        correct_preds += (outputs.argmax(dim=1) == labels).sum().item() # accumulate correct predictions
        total_samples += inputs.size(0)

        # Calculate average loss and accuracy across all batches
        avg_train_loss = total_train_loss / total_samples
        avg_train_acc = correct_preds / total_samples
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
                        "Avg Train Loss": avg_train_loss, #calculates the average training loss across all batches.
                        "Avg Train Accuracy": avg_train_acc, #calculates the average training accuracy across all batches.
                        })
        train_logger.log_metrics(metrics,step=global_step_count)
        
        fabric.print(
                f" Job {job_id} | Epoch: {metrics['Epoch Index']}({metrics['Batch Index']}/{min(len(train_dataloader),limit_train_batches)}) |"
                # f" loss train: {metrics['Train Loss']:.3f} |"
                # f" val: {val_loss} |"
                f" iter time: {metrics['Total Iteration Time (s)']:.2f} |"
                f" dataload time: {metrics['Data Loading Time (s)']:.2f} |"
                f" gpu time: {metrics['GPU Processing Time (s)']:.2f} |"
                f" fetch time: {metrics['Data Fetch Time (s)']:.2f} |"
                f" transform time: {metrics['Transformation Time (s)']:.2f} |"
                f" elapsed time: {metrics['Elapsed Time (s)']:.2f} |"

                )

        # stopping criterion on step level
        if max_steps is not None and global_step_count >= max_steps:
            break

        end = time.perf_counter()

    return  global_step_count


def validate_loop(fabric,job_id, val_logger:CSVLogger, model, dataloader, val_start_time, current_epoch, global_step_count, limit_val_batches=np.inf, criterion=nn.CrossEntropyLoss()):
    model.eval()
    end = time.perf_counter()

    total_val_loss = 0.0
    correct_preds = 0
    total_samples = 0

    for batch_idx, (batch, data_fetch_time, transformation_time, cache_hit) in enumerate(dataloader):
        if batch_idx >= limit_val_batches:
            break
        
        inputs, labels = batch

        # Forward pass
        gpu_processing_started = time.perf_counter()
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        gpu_processing_time = time.perf_counter() - gpu_processing_started

        # Metrics calculation
        val_loss = loss.item()
        val_acc = (outputs.argmax(dim=1) == labels).float().mean().item()
        
        total_val_loss += val_loss * inputs.size(0)  # accumulate total loss
        correct_preds += (outputs.argmax(dim=1) == labels).sum().item()
        total_samples += inputs.size(0)

        # Calculate average loss and accuracy
        avg_val_loss = total_val_loss / total_samples #Calculates the average validation loss across all batches.
        avg_val_acc = correct_preds / total_samples #Calculates the average validation accuracy across all batches.
        global_step_count +=1

        metrics = OrderedDict({
            "Elapsed Time (s)": time.perf_counter() - val_start_time,
            "Device": fabric.global_rank,
            "Epoch Index": current_epoch,
            "Batch Index": batch_idx + 1,
            "Batch Size": batch[0].size(0),
            "Total Iteration Time (s)": time.perf_counter() - end,
            "Data Fetch Time (s)": data_fetch_time,
            "Transformation Time (s)": transformation_time,
            "Cache Hit/Miss": 1 if cache_hit else 0,
            "Avg Validation Loss": avg_val_loss,
            "Avg Validation Accuracy": avg_val_acc
        })
        
        val_logger.log_metrics(metrics, step=global_step_count)

        fabric.print(
            f" Job {job_id} | Epoch {metrics['Epoch Index']}({metrics['Batch Index']}/{min(len(dataloader), limit_val_batches)}) |"
            f" iter time: {metrics['Total Iteration Time (s)']:.2f} |"
            f" dataload time: {metrics['Data Fetch Time (s)']:.2f} |"
            f" gpu time: {metrics['Data Fetch Time (s)']:.2f} |"
            f" elapsed time: {metrics['Elapsed Time (s)']:.2f} |"
            f" val loss: {metrics['Avg Validation Loss']:.3f} |"
            f" val acc: {metrics['Avg Validation Accuracy']:.3f}"
        )

        end = time.perf_counter()

    return global_step_count

@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(config: DictConfig):
    train_image_classifer(config)


if __name__ == "__main__":
    main()
