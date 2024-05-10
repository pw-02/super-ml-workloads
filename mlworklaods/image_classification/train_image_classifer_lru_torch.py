import time
from torch import nn, optim, no_grad
from typing import List, Dict
from omegaconf import DictConfig
from lightning.fabric import Fabric
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler  
from common import make_model, transform, accuracy
# Additional imports
from mlworklaods.args import TrainArgs, DataArgs, LRUTorchArgs
from mlworklaods.utils import ResourceMonitor, get_default_supported_precision, num_model_parameters
from mlworklaods.log_utils import ExperimentLogger, AverageMeter, ProgressMeter, create_exp_summary_report
from torch_lru.batch_sampler_with_id import BatchSamplerWithID
from torch_lru.torch_lru_dataset import TorchLRUDataset


def run_lru_torch_job(pid:int, config: DictConfig, train_args: TrainArgs, data_args: DataArgs, lru_torch_args:LRUTorchArgs):
    start_time = time.perf_counter()
    precision = get_default_supported_precision(training=True)
    fabric = Fabric(accelerator=train_args.accelerator, devices=train_args.devices, strategy="auto", precision=precision)

    fabric.launch(train_model, train_args.seed, config, train_args, data_args, lru_torch_args)

    # Create report at the end
    fabric.print("Creating overall report for experiment")
    output_file_path = create_exp_summary_report(train_args.log_dir)
    fabric.print(f"Job Ended. Total Duration: {(time.perf_counter() - start_time):.2f}s. Report: {output_file_path}")


def train_model(fabric: Fabric, seed: int, config: DictConfig, train_args: TrainArgs, data_args: DataArgs, lru_torch_args:LRUTorchArgs) -> None:
    
    if seed:
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
    train_dataloader, val_dataloader = make_dataloaders(fabric, train_args, data_args,lru_torch_args)

    # Create and set up logger
    logger = ExperimentLogger(fabric, train_args.log_dir, train_args.log_interval)
    if fabric.is_global_zero:
        logger.log_hyperparams(config)
        
    # Train/Valdate the model
    best_acc1_train = 0
    best_acc5_train = 0
    best_acc1_eval = 0
    best_acc5_eval = 0

    for epoch in range(1, train_args.epochs + 1):
        if train_dataloader:
            max_iters = min(len(train_dataloader), train_args.get_epoch_max_iters(fabric.world_size))
            fabric.print(f"Starting training loop for epoch {epoch}")
            model.train(True)
            loss_train, acc1_train, acc5_train = train_loop(fabric, epoch, model, optimizer, train_dataloader, max_iters, logger, train_args.dataload_only)
            best_acc1_train = max(acc1_train, best_acc1_train)
            best_acc5_train = max(acc5_train, best_acc5_train)

        if val_dataloader:
            max_iters = min(len(val_dataloader), train_args.get_epoch_max_iters(fabric.world_size))
            fabric.print(f"Starting validation loop for epoch {epoch}")
            model.eval()
            loss_eval, acc1_eval, acc5_eval = val_loop(fabric, epoch, model, val_dataloader, max_iters, logger)
            best_acc1_eval = max(acc1_eval, best_acc1_eval)
            best_acc5_eval = max(acc5_eval, best_acc5_eval)
    
    fabric.print(f"Training finished on device {fabric.global_rank}.")

    # Generate report at the end
    logger.create_job_report()
  

# Train loop
def train_loop(fabric: Fabric, epoch: int, model: nn.Module, optimizer: optim.Optimizer, train_dataloader: DataLoader, max_iters: int, logger: ExperimentLogger, dataload_only: bool):
    
    toal_cahce_hits = 0
    total_samples = 0
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    fetch_time = AverageMeter("Fetch", ":6.3f")
    transform_time = AverageMeter("Transform", ":6.3f")
    compute_time = AverageMeter("Compute", ":6.3f")
    losses = AverageMeter("Loss", ":6.2f")
    top1 = AverageMeter("Acc1", ":6.2f")
    top5 = AverageMeter("Acc5", ":6.2f")
    cache_hit_ratio = AverageMeter("hit%", ":6.2f")

    progress = ProgressMeter(max_iters, [batch_time, data_time, fetch_time, transform_time, compute_time, losses, cache_hit_ratio], prefix=f"Epoch {epoch}")

    with ResourceMonitor() as monitor:
        end = time.perf_counter()

        for batch_idx, (images, target, cache_hits, fetch_duration, transform_duration) in enumerate(train_dataloader):
            data_time.update(time.perf_counter() - end)
            fetch_time.update(fetch_duration)
            transform_time.update(transform_duration)

            batch_size = images.size(0)
            toal_cahce_hits += cache_hits
            cache_hit_ratio.update(cache_hits / batch_size, 1)
            
            if not dataload_only:
                is_accumulating = False
                with fabric.no_backward_sync(model, enabled=is_accumulating):
                    output = model(images)
                    loss = nn.functional.cross_entropy(output, target)
                    fabric.backward(loss)

                    if not is_accumulating:
                        optimizer.step()
                        optimizer.zero_grad()

                    acc1, acc5 = accuracy(output.data, target, topk=(1, 5))
                    losses.update(loss.item(), batch_size)
                    top1.update(acc1.item(), batch_size)
                    top5.update(acc5.item(), batch_size)

                compute_time.update(time.perf_counter() - end - data_time.val)

            batch_time.update(time.perf_counter() - end)
            total_samples += batch_size

            if batch_idx % logger.log_freq == 0:
                fabric.print(f'{progress.display(batch_idx + 1)}')
                # progress.display(batch_idx + 1)

                logger.save_train_batch_metrics(
                    epoch=epoch,
                    step=batch_idx + 1,
                    global_step=(epoch * max_iters) + batch_idx + 1,
                    num_samples=batch_size,
                    total_time=batch_time.val,
                    data_time=data_time.val,
                    fetch_time = fetch_time.val,
                    transform_time = transform_time.val,
                    compute_time=compute_time.val,
                    cache_hits=cache_hits,
                    loss=losses.val,
                    acc1=top1.val,
                    acc5=top5.val,
                    avg_cpu=monitor.resource_data["cpu_util"].summarize()["mean"],
                    max_cpu=monitor.resource_data["cpu_util"].summarize()["max"],
                    avg_gpu=monitor.resource_data["gpu_util"].summarize()["mean"],
                    max_gpu=monitor.resource_data["gpu_util"].summarize()["max"]
                )

            if batch_idx >= max_iters:
                break

            end = time.perf_counter()

        logger.save_train_epoch_metrics(
            epoch=epoch,
            num_samples=total_samples,
            num_batches = batch_idx + 1,
            global_step=(epoch * max_iters) + batch_idx + 1,
            total_time=batch_time.sum,
            data_time=data_time.sum,
            fetch_time=fetch_time.sum,
            transform_time = transform_time.sum,
            compute_time=compute_time.sum,
            loss=losses.avg,
            acc1=top1.avg,
            acc5=top5.avg,
            cache_hits=toal_cahce_hits,
            avg_cpu=monitor.resource_data["cpu_util"].summarize()["mean"],
            max_cpu=monitor.resource_data["cpu_util"].summarize()["max"],
            avg_gpu=monitor.resource_data["gpu_util"].summarize()["mean"],
            max_gpu=monitor.resource_data["gpu_util"].summarize()["max"]
        )

    return losses.avg, top1.avg, top5.avg

# Validation loop
def val_loop(fabric: Fabric, epoch: int, model: nn.Module, val_dataloader: DataLoader, max_iters: int, logger: ExperimentLogger):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":6.2f")
    top1 = AverageMeter("Acc1", ":6.2f")
    top5 = AverageMeter("Acc5", ":6.2f")

    progress = ProgressMeter(max_iters, [batch_time, losses, top1, top5], prefix=f"Test (Epoch {epoch}):")

    with no_grad():
        total_samples = 0
        end = time.perf_counter()

        for batch_idx, (images, target,cache_hits) in enumerate(val_dataloader):
            batch_size = images.size(0)
            output = model(images)
            loss = nn.functional.cross_entropy(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            losses.update(loss.item(), batch_size)
            top1.update(acc1[0], batch_size)
            top5.update(acc5[0], batch_size)
            batch_time.update(time.perf_counter() - end)

            total_samples += batch_size

            if batch_idx % logger.log_freq == 0:
                progress.display(batch_idx + 1)

                logger.save_eval_batch_metrics(
                    epoch=epoch,
                    step=batch_idx + 1,
                    global_step=(epoch * max_iters) + batch_idx + 1,
                    num_samples=batch_size,
                    batch_time=batch_time.val,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                )

            if batch_idx >= max_iters:
                break

            end = time.perf_counter()

    logger.save_eval_epoch_metrics(
        epoch=epoch,
        num_samples=total_samples,
        total_time=batch_time.sum,
        loss=losses.avg,
        top1=top1.avg,
        top5=top5.avg,
    )

    return top1.avg, top5.avg

# Dataloader creation function
def make_dataloaders(fabric: Fabric, train_args: TrainArgs, data_args: DataArgs, lru_torch_args:LRUTorchArgs):
    train_dataloader = None
    val_dataloader = None
    if train_args.run_training:
        train_dataset =  TorchLRUDataset(
            data_dir = data_args.train_data_dir,
            transform=transform(),
            cache_address=lru_torch_args.cache_address,
            cache_granularity=lru_torch_args.cache_granularity)
        
        train_base_sampler = RandomSampler(data_source=train_dataset) if lru_torch_args.shuffle else SequentialSampler(data_source=train_dataset)
        train_batch_sampler = BatchSamplerWithID(sampler=train_base_sampler, batch_size=train_args.get_batch_size(fabric.world_size), drop_last=False)
        
        train_dataloader = DataLoader(dataset=train_dataset, sampler=train_batch_sampler, batch_size=None, num_workers=train_args.num_pytorch_workers)
        train_dataloader = fabric.setup_dataloaders(train_dataloader, move_to_device=True, use_distributed_sampler=True)

    if train_args.run_evaluation:
        val_dataset =  TorchLRUDataset(
            data_dir = data_args.val_data_dir,
            transform=transform(),
            cache_address=lru_torch_args.cache_address,
            cache_granularity=lru_torch_args.cache_granularity)
        
        val_base_sampler = RandomSampler(data_source=val_dataset) if lru_torch_args.shuffle else SequentialSampler(data_source=val_dataset)
        val_batch_sampler = BatchSamplerWithID(sampler=val_base_sampler, batch_size=train_args.get_batch_size(fabric.world_size), drop_last=False)
        
        val_dataloader = DataLoader(dataset=val_dataset, sampler=val_batch_sampler, batch_size=None, num_workers=train_args.num_pytorch_workers)
        val_dataloader = fabric.setup_dataloaders(train_dataloader, move_to_device=True, use_distributed_sampler=True)

    
    return train_dataloader, val_dataloader
