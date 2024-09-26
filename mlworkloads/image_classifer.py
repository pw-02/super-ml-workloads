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
from dataloading.coordl.coordl_mapped_vision_dataset import CoorDLMappedVisionDataset
from dataloading.super.super_sampler import SUPERSampler
from dataloading.super.super_mapped_dataset import SUPERMappedDataset
from torch.utils.data import RandomSampler, SequentialSampler
import time
import os
from collections import OrderedDict
import numpy as np
import redis
import heapdict
from dataloading.shade.shadedataset import ShadeDataset
from dataloading.shade.shadesampler import ShadeSampler
import math

#Initialization of local cache, PQ and ghost cache (for shade)
PQ = heapdict.heapdict()
ghost_cache = heapdict.heapdict()

def train_image_classifer(config: DictConfig,  train_logger: CSVLogger, val_logger: CSVLogger):
  
    fabric = Fabric(
        accelerator=config.accelerator, 
        devices=config.devices, 
        precision=config.workload.precision   
    )

    if config.seed is not None:
        seed_everything(config.seed) # instead of torch.manual_seed(...)
    else:
        seed_everything(config.job_id) # instead of torch.manual_seed(...)

    model = get_model(name=config.workload.model_architecture, weights=None, num_classes=config.workload.num_classes)
    optimizer = optim.Adam(model.parameters(), lr=config.workload.learning_rate)
    model, optimizer = fabric.setup(model, optimizer)
    
    train_transform, val_transform = get_transforms(config.workload.name)

    train_dataloader = None
    val_dataloader = None

    if config.dataloader.name == 'super':
        if config.workload.run_training:
            train_dataset = SUPERMappedDataset(s3_data_dir=config.workload.s3_train_prefix, 
                                               transform=train_transform,
                                               cache_address=config.dataloader.cache_address)

            train_sampler = SUPERSampler(
                dataset=train_dataset,
                grpc_server_address=config.dataloader.grpc_server_address,
                batch_size=config.workload.batch_size
                )
            train_dataloader = DataLoader(train_dataset, batch_size=None, sampler=train_sampler, num_workers=config.workload.num_pytorch_workers)
            train_dataloader = fabric.setup_dataloaders(train_dataloader, move_to_device=True)
        
        if config.workload.run_validation:
            val_dataset = SUPERMappedDataset(s3_data_dir=config.workload.s3_val_prefix, 
                                             transform=val_transform,
                                             cache_address=config.dataloader.cache_address)
            val_sampler = SUPERSampler(
                dataset=val_dataset,
                grpc_server_address=config.dataloader.grpc_server_address,
                batch_size=config.workload.batch_size
                )
            val_dataloader =  DataLoader(val_dataset, batch_size=None, sampler=val_sampler, num_workers=config.workload.num_pytorch_workers)
            val_dataloader = fabric.setup_dataloaders(val_dataloader,move_to_device=True)
    
    elif config.dataloader.name == 'shade':
        global PQ
        # global key_id_map
        global ghost_cache

        if config.workload.run_training:
            # key_id_map = redis.StrictRedis(host=config.dataloader.cache_address.split(":")[0], port=config.dataloader.cache_address.split(":")[1])
            train_dataset = ShadeDataset(s3_data_dir=config.workload.s3_train_prefix, 
                                        transform=train_transform,
                                        cache_address=config.dataloader.cache_address,
                                        PQ=PQ,
                                        ghost_cache=ghost_cache,
                                        wss=config.dataloader.wss)
            train_sampler = ShadeSampler(
                dataset=train_dataset,
                num_replicas=1,
                rank=0,
                batch_size=config.workload.batch_size,
                seed=config.job_id,
                host_ip=config.dataloader.cache_address.split(":")[0],
                port_num=config.dataloader.cache_address.split(":")[1],
                rep_factor=config.dataloader.rep_factor,
                )
            
            train_dataloader = DataLoader(train_dataset, 
                                        #   shuffle=config.dataloader.shuffle,
                                          batch_size=config.workload.batch_size,
                                          sampler=train_sampler, 
                                          num_workers=config.workload.num_pytorch_workers)
            train_dataloader = fabric.setup_dataloaders(train_dataloader,move_to_device=True)


    elif config.dataloader.name == 'coordl':
        # PyTorch DataLoader
        if config.workload.run_training:
            train_dataset = CoorDLMappedVisionDataset(s3_data_dir=config.workload.s3_train_prefix, transform=train_transform, cache_address=config.dataloader.cache_address, wss=config.dataloader.wss)
            if config.dataloader.shuffle:
                train_sampler = RandomSampler(data_source=train_dataset)
            else:
                train_sampler = SequentialSampler(data_source=train_dataset)

            train_dataloader = DataLoader(train_dataset, batch_size=config.workload.batch_size, sampler=train_sampler, num_workers=config.workload.num_pytorch_workers)
            train_dataloader = fabric.setup_dataloaders(train_dataloader, move_to_device=True)
        
        if config.workload.run_validation:
            val_dataset = CoorDLMappedVisionDataset(s3_prefix=config.workload.s3_val_prefix, transform=val_transform, cache_address=config.dataloader.cache_address, wss=config.dataloader.wss)
            if config.dataloader.shuffle:
                val_sampler = RandomSampler(data_source=val_dataset)
            else:
                val_sampler = SequentialSampler(data_source=val_dataset)
            val_dataloader =  DataLoader(val_dataset, batch_size=config.workload.batch_size, sampler=val_sampler, num_workers=config.workload.num_pytorch_workers)
            val_dataloader = fabric.setup_dataloaders(val_dataloader,move_to_device=True)

    # # # Start training
    # metric_collector = ResourceMonitor(interval=1, flush_interval=10, file_path= f'{log_dir}/resource_usage_metrics.json')
    # # metric_collector.start()
    global_train_step_count = 0
    global_val_step_count = 0
    current_epoch=0
    should_stop = False
    train_start_time = time.perf_counter()

    if config.dataloader.name == 'shade':
        batch_wts = []
        for j in range(config.workload.batch_size):
            batch_wts.append(math.log(j+10))
    else:
        batch_wts = None
    
    if config.workload.limit_train_batches is None:
        config.workload.limit_train_batches = len(train_dataloader)
        
    while not should_stop:

        if isinstance(train_dataloader.sampler, ShadeSampler):
            train_dataloader.sampler.set_epoch(current_epoch)

        current_epoch += 1

        global_train_step_count = train_loop(
            fabric=fabric,
            job_id=config.job_id,
            train_logger=train_logger,
            model=model,
            optimizer=optimizer,
            train_dataloader=train_dataloader,
            train_start_time=train_start_time,
            current_epoch=current_epoch,
            global_step_count=global_train_step_count,
            max_steps=config.workload.max_steps,
            limit_train_batches=config.workload.limit_train_batches,
            criterion=nn.CrossEntropyLoss(reduction = 'none'), # if isinstance(train_dataloader.sampler, ShadeSampler) else nn.CrossEntropyLoss(),
            batch_wts=batch_wts)
    
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

    if isinstance(train_dataloader.sampler, SUPERSampler):
        train_dataloader.sampler.send_job_ended_notfication()


    elapsed_time = time.perf_counter() - train_start_time

    fabric.print(f"Training completed in {elapsed_time:.2f} seconds")
    # metric_collector.stop()



def get_transforms(workload_name):
    if workload_name == 'imagenet_resnet50':
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
    elif workload_name == 'cifar10_resnet18':
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

def train_loop(fabric:Fabric, job_id, train_logger:CSVLogger, model, optimizer, train_dataloader:DataLoader, train_start_time, current_epoch, global_step_count, max_steps = None, limit_train_batches = np.inf, criterion=nn.CrossEntropyLoss(), batch_wts = None):
    model.train()
    total_samples = 0
    total_train_loss = 0.0
    correct_preds = 0

    end = time.perf_counter()
    for batch_idx, (batch, data_load_time, transformation_time, is_cache_hit, cached_on_miss) in enumerate(train_dataloader):
            
            wait_for_data_time = time.perf_counter() - end
            # end epoch if stopping training completely or max batches for this epoch reached
            if limit_train_batches is not None and batch_idx >= limit_train_batches:
                break
            
             # Unpack batch
            if isinstance(train_dataloader.sampler, ShadeSampler) or isinstance(train_dataloader.dataset, CoorDLMappedVisionDataset):
                inputs, labels = batch
            elif isinstance(train_dataloader.sampler, SUPERSampler):
                inputs, labels, batch_id = batch

            # Forward pass: Compute model output and loss
            gpu_processing_started = time.perf_counter()
            outputs  = model(inputs)
            item_loss = criterion(outputs, labels)
            loss = item_loss.mean()

            # Backpropagation and optimization
            optimizer.zero_grad()  # Clear previous gradients
            fabric.backward(loss)  # Backpropagation
            optimizer.step()  # Update weights

            # Accumulate metrics directly on GPU to avoid synchronization
            correct_preds += (outputs.argmax(dim=1) == labels).sum().item()  # No .item(), stays on GPU
            total_train_loss += loss.item() * inputs.size(0)  # Convert loss to CPU for accumulation
            
            if fabric.device.type == 'cuda':
                torch.cuda.synchronize()


             # Track time taken for GPU processing
            gpu_processing_time = time.perf_counter() - gpu_processing_started
            
            # Metrics calculation
            total_samples += inputs.size(0)
            avg_train_loss = total_train_loss / total_samples
            avg_train_acc = correct_preds / total_samples
            global_step_count +=1

            # Calculate average loss and accuracy across all batches
            avg_train_loss = total_train_loss / total_samples
            
            if isinstance(train_dataloader.sampler, SUPERSampler):
                cache_hit_samples = batch[0].size(0) if is_cache_hit == True else 0
                cache_hit_bacth = 1 if is_cache_hit == True else 0

            if isinstance(train_dataloader.sampler, ShadeSampler) or isinstance(train_dataloader.dataset, CoorDLMappedVisionDataset):
                data_load_time = float(data_load_time.sum())
                transformation_time = float(transformation_time.sum())
                cache_hit_samples = int(is_cache_hit.sum())
                cache_hit_bacth = 1 if cache_hit_samples == len(is_cache_hit) else 0

                if isinstance(train_dataloader.sampler, ShadeSampler):
                    train_dataloader.sampler.pass_batch_important_scores(item_loss.cpu())
                    sorted_img_indices = train_dataloader.sampler.get_sorted_index_list()

                    key_id_map = redis.StrictRedis(host=train_dataloader.dataset.cache_host, port=train_dataloader.dataset.cache_port)
                    global ghost_cache
                    global PQ
                    track_batch_indx = 0
                    if current_epoch > 1:
                        PQ = train_dataloader.dataset.get_PQ()
                        ghost_cache = train_dataloader.dataset.get_ghost_cache()
                    for indx in sorted_img_indices:
                        if key_id_map.exists(indx.item()):
                            if indx.item() in PQ:
                                PQ[indx.item()] = (batch_wts[track_batch_indx],PQ[indx.item()][1]+1)
                                ghost_cache[indx.item()] = (batch_wts[track_batch_indx],ghost_cache[indx.item()][1]+1)
                                track_batch_indx+=1
                            else:
                                PQ[indx.item()] = (batch_wts[track_batch_indx],1)
                                ghost_cache[indx.item()] = (batch_wts[track_batch_indx],1)
                                track_batch_indx+=1
                        else:
                            if indx.item() in ghost_cache:
                                ghost_cache[indx.item()] = (batch_wts[track_batch_indx],ghost_cache[indx.item()][1]+1)
                                track_batch_indx+=1
                            else:
                                ghost_cache[indx.item()] = (batch_wts[track_batch_indx],1)
                                track_batch_indx+=1
                    train_dataloader.dataset.set_PQ(PQ)
                    train_dataloader.dataset.set_ghost_cache(ghost_cache)
                    train_dataloader.dataset.set_num_local_samples()

            if isinstance(train_dataloader.sampler, SUPERSampler):
                train_dataloader.sampler.send_job_update_to_super(
                    batch_id,
                    wait_for_data_time,
                    is_cache_hit,
                    gpu_processing_time,
                    cached_on_miss
                )

            metrics= OrderedDict({
                            "Elapsed Time (s)": time.perf_counter() - train_start_time,
                            "Num Torch Workers": train_dataloader.num_workers,
                            "Device": fabric.global_rank,
                            "Epoch Index": current_epoch,
                            "Batch Index": batch_idx+1,
                            "Batch Size": batch[0].size(0),
                            "Iteration Time (s)": time.perf_counter() - end,
                            "Wait for Data Time (s)": wait_for_data_time,
                            "GPU Processing Time (s)": gpu_processing_time,
                            "Data Load Time (s)": data_load_time,
                            "Transformation Time (s)": transformation_time,
                            "Cache_Hit (Batch)": cache_hit_bacth,
                            "Cache_Hits (Samples)": cache_hit_samples,
                            "Train Loss (Avg)": avg_train_loss, #calculates the average training loss across all batches.
                            "Train Accuracy (Avg)": avg_train_acc, #calculates the average training accuracy across all batches.
                            })
            train_logger.log_metrics(metrics,step=global_step_count)
            
            fabric.print(
                    f" Job {job_id} | Epoch: {metrics['Epoch Index']}({metrics['Batch Index']}/{min(len(train_dataloader),limit_train_batches)}) |"
                    # f" loss train: {metrics['Train Loss']:.3f} |"
                    # f" val: {val_loss} |"
                    f" iter:{metrics['Iteration Time (s)']:.2f}s |"
                    f" data_delay:{metrics['Wait for Data Time (s)']:.2f}s |"
                    f" gpu:{metrics['GPU Processing Time (s)']:.2f}s |"
                    f" data_fetch:{metrics['Data Load Time (s)']:.2f}s |"
                    f" transform:{metrics['Transformation Time (s)']:.2f}s |"
                    f" elapsed:{metrics['Elapsed Time (s)']:.2f}s |"
                    f" loss: {metrics['Train Loss (Avg)']:.3f} |"
                    f" acc: {metrics['Train Accuracy (Avg)']:.3f} |"
                    F" cache hit: {metrics['Cache_Hit (Batch)']} |"
                    )

            # stopping criterion on step level
            if max_steps is not None and global_step_count >= max_steps:
                break

            end = time.perf_counter()
    
    if isinstance(train_dataloader.sampler, ShadeSampler):
        train_dataloader.sampler.on_epoch_end(total_train_loss/batch_idx)
        
    return  global_step_count


def validate_loop(fabric,job_id, val_logger:CSVLogger, model, dataloader, val_start_time, current_epoch, global_step_count, limit_val_batches=np.inf, criterion=nn.CrossEntropyLoss()):
    model.eval()
    end = time.perf_counter()

    total_val_loss = 0.0
    correct_preds = 0
    total_samples = 0

    for batch_idx, (batch, data_load_time, transformation_time, is_cache_hit) in enumerate(dataloader):
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
            "Data Fetch Time (s)": data_load_time,
            "Transformation Time (s)": transformation_time,
            "Cache Hit/Miss": 1 if is_cache_hit else 0,
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


@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(config: DictConfig):

    log_dir = f"{config.log_dir}/{config.workload.name}/{config.dataloader.name}/{config.exp_id}/{config.job_id}".lower()
    log_dir = os.path.normpath(log_dir)  # Normalize path for Windows
    
    train_logger = CSVLogger(root_dir=log_dir, name="train", prefix='', flush_logs_every_n_steps=config.log_interval)
    val_logger = CSVLogger(root_dir=log_dir, name="val", prefix='', flush_logs_every_n_steps=config.log_interval)
    train_image_classifer(config, train_logger,val_logger)

  

if __name__ == "__main__":
    main()
