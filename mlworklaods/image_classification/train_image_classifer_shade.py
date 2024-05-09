import time
from torch import nn, optim, Tensor, no_grad
import torchvision
from typing import List, Dict
from omegaconf import DictConfig
from lightning.fabric import Fabric
from common import make_model, transform, accuracy

from torch.utils.data import DataLoader

# Additional imports
from mlworklaods.args import TrainArgs, DataArgs, SHADEArgs
from mlworklaods.utils import ResourceMonitor, get_default_supported_precision, num_model_parameters
from mlworklaods.log_utils import ExperimentLogger, AverageMeter, ProgressMeter, create_exp_summary_report
from shade.shadedataset import ShadeDataset
from shade.shadesampler import ShadeSampler
from torch_lru.batch_sampler_with_id import BatchSamplerWithID

import redis
import heapdict
import math

# red_local = redis.StrictRedis(host='172.17.0.2', port='6379')
PQ = heapdict.heapdict()
ghost_cache = heapdict.heapdict()
key_counter  = 0

def run_shade_job(pid:int, config: DictConfig, train_args: TrainArgs, data_args: DataArgs, shade_args:SHADEArgs):
    start_time = time.perf_counter()
    precision = get_default_supported_precision(training=True)
    fabric = Fabric(accelerator=train_args.accelerator, devices=train_args.devices, strategy="auto", precision=precision)

    fabric.launch(train_model, train_args.seed, config, train_args, data_args, shade_args)

    # Create report at the end
    fabric.print("Creating overall report for experiment")
    output_file_path = create_exp_summary_report(train_args.log_dir)
    fabric.print(f"Job Ended. Total Duration: {(time.perf_counter() - start_time):.2f}s. Report: {output_file_path}")


def train_model(fabric: Fabric, seed: int, config: DictConfig, train_args: TrainArgs, data_args: DataArgs, shade_args:SHADEArgs) -> None:
    fabric.seed_everything(seed)
    
    t0 = time.perf_counter()
    model:nn.Module = make_model(fabric, train_args.model_name)
    fabric.print(f"Time to instantiate {train_args.model_name} model: {time.perf_counter() - t0:.02f} seconds")
    fabric.print(f"Total parameters in {train_args.model_name} model: {num_model_parameters(model):,}")

    optimizer = optim.Adam(model.parameters(), lr=train_args.learning_rate)
    #optimizer = optim.SGD(model.parameters(), lr=train_args.learning_rate)
    
    # Set up the model and optimizer
    model, optimizer = fabric.setup(model, optimizer, move_to_device=True)

    # Set up train and validation dataloaders
    train_dataloader, val_dataloader = make_dataloaders(fabric, train_args, data_args,shade_args)

    # Create and set up logger
    logger = ExperimentLogger(fabric, train_args.log_dir, train_args.log_interval)
    if fabric.is_global_zero:
        logger.log_hyperparams(config)
        
    # Train/Valdate the model
    best_acc1_train = 0
    best_acc5_train = 0
    best_acc1_eval = 0
    best_acc5_eval = 0

    batch_wts = []
    for j in range(train_args.get_batch_size(fabric.world_size)):
        batch_wts.append(math.log(j+10))

    for epoch in range(1, train_args.epochs + 1):
        if train_dataloader:
            max_iters = min(len(train_dataloader), train_args.get_epoch_max_iters(fabric.world_size))
            fabric.print(f"Starting training loop for epoch {epoch}")
            model.train(True)
            loss_train, acc1_train, acc5_train = train_loop(fabric, epoch, model, optimizer, train_dataloader, max_iters, logger, train_args.dataload_only, shade_args, batch_wts)
            best_acc1_train = max(acc1_train, best_acc1_train)
            best_acc5_train = max(acc5_train, best_acc5_train)

        if val_dataloader:
            max_iters = min(len(val_dataloader), train_args.get_epoch_max_iters(fabric.world_size))
            fabric.print(f"Starting validation loop for epoch {epoch}")
            model.eval()
            loss_eval, acc1_eval, acc5_eval = train_loop(fabric, epoch, model, optimizer, val_dataloader, max_iters, logger, train_args.dataload_only, shade_args, batch_wts)
            best_acc1_eval = max(acc1_eval, best_acc1_eval)
            best_acc5_eval = max(acc5_eval, best_acc5_eval)
    
    fabric.print(f"Training finished on device {fabric.global_rank}.")

    # Generate report at the end
    logger.create_job_report()
  

# Train loop
def train_loop(fabric: Fabric, epoch: int, model: nn.Module, optimizer: optim.Optimizer, train_dataloader: DataLoader, max_iters: int, logger: ExperimentLogger, dataload_only: bool, shade_args:SHADEArgs,batch_wts:List):
    global PQ
    global key_id_map
    global key_counter
    # global red_local
    global ghost_cache
    cache_host,cache_port = shade_args.cache_address.split(":")
    # key_id_map = redis.StrictRedis(host='172.17.0.2', port='6379')
    key_id_map = redis.StrictRedis(host=cache_host, port=cache_port)

    toal_cahce_hits = 0
    total_samples = 0
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    compute_time = AverageMeter("Compute", ":6.3f")
    losses = AverageMeter("Loss", ":6.2f")
    top1 = AverageMeter("Acc1", ":6.2f")
    top5 = AverageMeter("Acc5", ":6.2f")
    cache_hit_ratio = AverageMeter("hit%", ":6.2f")

    progress = ProgressMeter(max_iters, [batch_time, data_time, compute_time, losses, cache_hit_ratio], prefix=f"Epoch {epoch}")

    with ResourceMonitor() as monitor:
        end = time.perf_counter()

        for batch_idx, (images, target, cache_hits) in enumerate(train_dataloader):
            data_time.update(time.perf_counter() - end)
            batch_size = images.size(0)
            toal_cahce_hits += cache_hits
            cache_hit_ratio.update(cache_hits / batch_size, 1)
            
            if not dataload_only:
                is_accumulating = False
                with fabric.no_backward_sync(model, enabled=is_accumulating):
                    output = model(images)
                    item_loss = nn.functional.cross_entropy(output, target, reduce=False)
                    loss = item_loss.mean()
                    fabric.backward(loss)
                    
                    train_dataloader.sampler.sampler.pass_batch_important_scores(item_loss)

                    if not is_accumulating:
                        optimizer.step()
                        optimizer.zero_grad()

                    acc1, acc5 = accuracy(output.data, target, topk=(1, 5))
                    losses.update(loss.item(), batch_size)
                    top1.update(acc1.item(), batch_size)
                    top5.update(acc5.item(), batch_size)

                compute_time.update(time.perf_counter() - end - data_time.val)
            
            key_counter = key_id_map.dbsize()
            sorted_img_indices = train_dataloader.sampler.sampler.get_sorted_index_list()
            track_batch_indx = 0
            # Updating PQ and ghost cache during training.
            if epoch > 1:
                PQ = train_dataloader.dataset.get_PQ()
                ghost_cache = train_dataloader.dataset.get_ghost_cache()
            
            for indx in sorted_img_indices:
                    if key_id_map.exists(indx.item()):
                        if indx.item() in PQ:
                            #print("Train_index: %d Importance_Score: %f Frequency: %d Time: %s N%dG%d" %(indx.item(),batch_loss,PQ[indx.item()][1]+1,insertion_time,args.nr+1,gpu+1))
                            PQ[indx.item()] = (batch_wts[track_batch_indx],PQ[indx.item()][1]+1)
                            ghost_cache[indx.item()] = (batch_wts[track_batch_indx],ghost_cache[indx.item()][1]+1)
                            track_batch_indx+=1
                        else:
                            #print("Train_index: %d Importance_Score: %f Time: %s N%dG%d" %(indx.item(),batch_loss,insertion_time,args.nr+1,gpu+1))
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
            train_dataloader.dataset.set_num_local_samples(key_counter)

            batch_time.update(time.perf_counter() - end)
            total_samples += batch_size

            if batch_idx % logger.log_freq == 0:
                progress.display(batch_idx + 1)

                logger.save_train_batch_metrics(
                    epoch=epoch,
                    step=batch_idx + 1,
                    global_step=(epoch * max_iters) + batch_idx + 1,
                    num_samples=batch_size,
                    total_time=batch_time.val,
                    data_time=data_time.val,
                    compute_time=compute_time.val,
                    cache_hits=cache_hits,
                    loss=losses.val,
                    acc1=top1.val,
                    acc5=top5.val,
                    avg_cpu=monitor.resource_data["cpu_util"].summarize()["mean"],
                    max_cpu=monitor.resource_data["cpu_util"].summarize()["max"],
                    avg_gpu=monitor.resource_data["gpu_util"].summarize()["mean"],
                    max_gpu=monitor.resource_data["gpu_util"].summarize()["max"],
                )

            if batch_idx >= max_iters:
                break

            end = time.perf_counter()

        logger.save_train_epoch_metrics(
            epoch=epoch,
            total_samples=total_samples,
            total_time=batch_time.sum,
            data_time=data_time.sum,
            compute_time=compute_time.sum,
            loss=losses.avg,
            acc1=top1.avg,
            acc5=top5.avg,
            cache_hits=toal_cahce_hits,
            avg_cpu=monitor.resource_data["cpu_util"].summarize()["mean"],
            max_cpu=monitor.resource_data["cpu_util"].summarize()["max"],
            avg_gpu=monitor.resource_data["gpu_util"].summarize()["mean"],
            max_gpu=monitor.resource_data["gpu_util"].summarize()["max"],
        )
        train_dataloader.sampler.sampler.on_epoch_end(losses.avg)

    return losses.avg, top1.avg, top5.avg



# Dataloader creation function
def make_dataloaders(fabric: Fabric, train_args: TrainArgs, data_args: DataArgs, shade_args:SHADEArgs):
    train_dataloader = None
    val_dataloader = None
    if train_args.run_training:
        tarin_dataset = ShadeDataset(
            data_dir = data_args.train_data_dir,
            transform=transform(),
            target_transform=None,
            cache_data=True if shade_args.cache_address is not None else False,
            PQ=PQ,
            ghost_cache=ghost_cache,
            key_counter=key_counter,
            wss=shade_args.working_set_size,
            cache_address=shade_args.cache_address
            )
        
        train_sampler = ShadeSampler(
            dataset = tarin_dataset,
            num_replicas=fabric.world_size,
            rank=fabric.node_rank, 
            batch_size=train_args.get_batch_size(fabric.world_size), 
            seed = train_args.seed,
            drop_last= False,
            replacement=True,
            cache_address=shade_args.cache_address,
            rep_factor=shade_args.replication_factor,
            init_fac=1,
            ls_init_fac = 0.01
            )
        
        train_batch_sampler = BatchSamplerWithID(sampler=train_sampler, batch_size=train_args.get_batch_size(fabric.world_size), drop_last=False)
        train_dataloader = DataLoader(dataset=tarin_dataset, sampler=train_batch_sampler, batch_size=None, num_workers=train_args.num_pytorch_workers)
        train_dataloader = fabric.setup_dataloaders(train_dataloader, move_to_device=True, use_distributed_sampler=True)

    if train_args.run_evaluation:
        val_dataset = ShadeDataset(
            data_dir = data_args.val_data_dir,
            transform=transform(),
            target_transform=None,
            cache_data=True if shade_args.cache_address is not None else False,
            PQ=PQ,
            ghost_cache=ghost_cache,
            key_counter=key_counter,
            wss=shade_args.working_set_size,
            cache_address=shade_args.cache_address
            )
        
        val_sampler = ShadeSampler(
            dataset = val_dataset,
            num_replicas=fabric.world_size,
            rank=fabric.node_rank, 
            batch_size=train_args.get_batch_size(fabric.world_size), 
            seed = train_args.seed,
            drop_last= False,
            replacement=True,
            cache_address=shade_args.cache_address,
            rep_factor=shade_args.replication_factor,
            init_fac=1,
            ls_init_fac = 0.01
            )
        val_batch_sampler = BatchSamplerWithID(sampler=val_sampler, batch_size=train_args.get_batch_size(fabric.world_size),  drop_last=False)
        val_dataloader = DataLoader(dataset=val_dataset, sampler=val_batch_sampler, batch_size=None, num_workers=train_args.num_pytorch_workers)
        val_dataloader = fabric.setup_dataloaders(train_dataloader, move_to_device=True, use_distributed_sampler=True)

    return train_dataloader, val_dataloader
