from typing import List
import hydra
from omegaconf import DictConfig
from lightning.fabric import Fabric
import time
from torch import nn, optim, Tensor, no_grad
from torch.utils.data import DataLoader
import torchvision
from super_dl.dataset.s3_imgae_mapped import S3MappedImageDataset
from super_dl.dataset.super_dataset import SUPERDataset
from super_dl.dataloader.super_dataloader import SUPERDataLoader
from super_dl.super_client import SuperClient
from mlworklaods.args import *
from torch.utils.data import SequentialSampler, RandomSampler  
from mlworklaods.utils import ResourceMonitor, get_default_supported_precision, num_model_parameters
from mlworklaods.log_utils import AverageMeter, ProgressMeter, Summary, ExperimentLogger, get_next_exp_version, create_exp_summary_report
import os
from torch_datasets.vanilla_image_dataset import VanillaTorchImageDataset

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def setup(config: DictConfig):

    start_time = time.perf_counter()
    precision = get_default_supported_precision(training=True)
    fabric = Fabric(accelerator=config.accelerator, devices=config.devices, strategy="auto", precision=precision)
    exp_version = get_next_exp_version(config.log_dir,config.dataset.name)

    train_args: TrainArgs = TrainArgs(
        job_id=os.getpid(),
        model_name = config.training.model_name,
        epochs = config.training.epochs,
        global_batch_size=config.training.batch_size,
        global_epoch_max_iters = config.training.iters_per_epoch,
        learning_rate = config.training.learning_rate,
        weight_decay = config.training.weight_decay,
        num_pytorch_workers = config.training.num_pytorch_workers,
        shuffle= config.dataloader.shuffle,
        run_training = config.run_training,
        run_evaluation = config.run_evaluation,
        simulate_data_delay=config.dataloader.simulate_data_delay,
        dataload_only = config.dataload_only
        )
     
    io_args: IOArgs = IOArgs(
        dataloader_kind= config.dataloader.kind,
        train_data_dir=config.dataset.train_dir,
        val_data_dir=config.dataset.val_dir,
        log_dir = os.path.join(config.log_dir, config.dataset.name, str(exp_version)),
        log_interval = config.log_interval
        )
    
    if 'super' in io_args.dataloader_kind:
        super_client:SuperClient = SuperClient(super_addresss=io_args.super_address)     
        super_client.register_job(job_id=train_args.job_id, data_dir=io_args.train_data_dir)
        del(super_client)
        io_args.super_address=config.dataloader.super_address,
        io_args.cache_address=config.dataloader.cache_address,
    
    if 'shade' in io_args.dataloader_kind:
        io_args.cache_address=config.dataloader.cache_address,

    fabric.launch(main, config.seed, config, train_args,io_args)

    fabric.print(f"Creating overall report for experiment")
    create_exp_summary_report(io_args.log_dir)

    fabric.print(f"Experiement Ended. Total Duration {(time.perf_counter()-start_time):.2f}s")


def main(fabric: Fabric, seed: int, config: DictConfig, train_args: TrainArgs, io_args: IOArgs,) -> None:        
        fabric.seed_everything(seed)  # same seed for every process to init model (FSDP)
        #srtup model 
        t0 = time.perf_counter()
        model:nn.Module = make_model(fabric, train_args.model_name) 
        fabric.print(f"Time to instantiate {train_args.model_name} model: {time.perf_counter() - t0:.02f} seconds")
        fabric.print(f"Total parameters in {train_args.model_name} model: {num_model_parameters(model):,}")
        optimizer = optim.Adam(model.parameters(), lr=train_args.learning_rate)
        #optimizer = optim.SGD(model.parameters(), lr=train_args.learning_rate)

        model, optimizer = fabric.setup(model,optimizer, move_to_device=True)

        train_dataloader, val_dataloader = None, None
        if train_args.run_training:
            train_dataloader = make_dataloader(train_args.job_id, 
                                               io_args.dataloader_kind, 
                                               io_args.train_data_dir, 
                                               train_args.shuffle, 
                                               train_args.global_batch_size, 
                                               train_args.num_pytorch_workers,
                                               fabric.world_size, 
                                               io_args.super_address,
                                               io_args.cache_address,
                                               train_args.simulate_data_delay)
            
            train_dataloader = fabric.setup_dataloaders(train_dataloader, move_to_device=True, use_distributed_sampler=True)
        if train_args.run_evaluation:
            val_dataloader = make_dataloader(
                train_args.job_id,
                io_args.dataloader_kind,
                io_args.val_data_dir, 
                False, 
                train_args.global_batch_size, 
                train_args.num_pytorch_workers,
                fabric.world_size(), 
                io_args.super_address,
                io_args.cache_address,
                train_args.simulate_data_delay)
            val_dataloader = fabric.setup_dataloaders(val_dataloader, move_to_device=True, use_distributed_sampler=False)
        
        train_time = time.perf_counter()
        logger = ExperimentLogger(fabric, io_args.log_dir, io_args.log_interval)
        if fabric.is_global_zero:
             logger.log_hyperparams(config)

        fit(fabric, model, optimizer, train_dataloader, val_dataloader, logger, train_args)
        
        fabric.print(f"Training finished on device {fabric.global_rank} after {(time.perf_counter()-train_time):.2f}s")
        # if fabric.device.type == "cuda":
        #     fabric.print(f"Memory used: {cuda.max_memory_allocated() / 1e9:.02f} GB")
        
        #fabric.print(f"Creating job report for device {fabric.global_rank}..")
        logger.create_job_report()
    
def fit(fabric: Fabric, 
        model:nn.Module, 
        optimizer, 
        train_dataloader:DataLoader, 
        val_dataloader:DataLoader, 
        logger:ExperimentLogger, 
        train_args:TrainArgs):
     
    best_acc1_train = 0
    best_acc5_train = 0
    best_acc1_eval = 0
    best_acc5_eval = 0

    for epoch in range(1, train_args.epochs +1):
        if train_dataloader:
            max_iters = min(len(train_dataloader), train_args.epoch_max_iters(fabric.world_size)) 
            fabric.print(f"Starting training loop for epoch {epoch}")
            model.train(mode=True)
            loss_train, acc1_train, acc5_train = train_loop(fabric, epoch, model, optimizer, train_dataloader, max_iters, logger,train_args.dataload_only)
              # remember best acc@1 and acc@5
            best_acc1_train = max(acc1_train, best_acc1_train)
            best_acc5_train = max(acc5_train, best_acc5_train)


        if val_dataloader:
            max_iters = min(len(val_dataloader), train_args.epoch_max_iters(fabric.world_size)) 
            fabric.print(f"Starting validation loop for epoch {epoch}")
            model.eval()
            loss_eval, acc1_eval, acc5_eval = val_loop(epoch, model, val_dataloader,max_iters, logger) 
            best_acc1_eval = max(acc1_eval, best_acc1_eval)
            best_acc5_eval = max(acc5_eval, best_acc5_eval)
    
def val_loop(fabric: Fabric,
             epoch:int,
             model:nn.Module, 
             val_dataloader:DataLoader,
             max_iters:int,
             logger:ExperimentLogger):
        total_samples  = 0
        batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
        losses = AverageMeter('Loss', ':6.2f')
        top1 = AverageMeter('Acc1', ':6.2f', Summary.AVERAGE)
        top5 = AverageMeter('Acc5', ':6.2f', Summary.AVERAGE)
        progress = ProgressMeter(
        max_iters,[batch_time, losses, top1, top5],prefix='Test: ')
        end = time.perf_counter() 

        with no_grad():
            for batch_idx,(images, target) in enumerate(val_dataloader):
                batch_lebgth = images.size(0)
                output:Tensor = model(images)
                loss = nn.functional.cross_entropy(output, target)
                acc1, acc5 = accuracy(output, target, topk=(1, 5))

                losses.update(loss.item(), batch_lebgth)
                top1.update(acc1[0], batch_lebgth)
                top5.update(acc5[0], batch_lebgth)
                batch_time.update(time.perf_counter() - end)
                total_samples += len(batch_lebgth)

                if batch_idx % logger.log_freq == 0:
                    progress.display(batch_idx + 1, fabric)

                    logger.save_eval_batch_metrics(
                        epoch=epoch,
                        step=batch_idx+1,
                        global_step=(epoch*max_iters) + batch_idx+1,
                        num_sampels=batch_lebgth,
                        batch_time=batch_time.val,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg
                    )
                if batch_idx >= max_iters:
                    # end loop early as max number of minibatches have been processed 
                    break
                end = time.perf_counter()
            
            logger.save_eval_epoch_metrics(
                epoch=epoch,
                num_samples=total_samples,
                global_step=((epoch+1)*max_iters),
                num_batches = max_iters,
                total_time=batch_time.sum,
                loss=losses.avg,
                top1=top1.avg,
                top5=top5.avg
            )
        return top1.avg, top5.avg


def train_loop(fabric: Fabric,
                epoch:int,
                model:nn.Module, 
                optimizer:optim.Optimizer, 
                train_dataloader:DataLoader, 
                max_iters:int, 
                logger:ExperimentLogger, 
                dataload_only:bool):
        
        total_samples = 0
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        compute_time = AverageMeter('Compute', ':6.3f')
        losses = AverageMeter('Loss', ':6.2f')
        top1 = AverageMeter('Acc1', ':6.2f')
        top5 = AverageMeter('Acc5', ':6.2f')
        progress = ProgressMeter(max_iters,
            [batch_time, data_time, compute_time, losses, top1, top5],
            prefix="Epoch: [{}]".format(epoch))
        
        with ResourceMonitor() as monitor:
            end = time.perf_counter()
            for batch_idx,(images, target) in enumerate(train_dataloader):
                data_time.update(time.perf_counter() - end)
                batch_size = images.size(0)
                if not dataload_only:
                    is_accumulating = False
                    # Forward pass and loss calculation
                    with fabric.no_backward_sync(model, enabled=is_accumulating):
                        output:Tensor = model(images)
                        loss = nn.functional.cross_entropy(output, target)
                        fabric.backward(loss) # .backward() accumulates when .zero_grad() wasn't called
                    
                    if not is_accumulating:
                        # Step the optimizer after accumulation phase is over
                        optimizer.step()
                        optimizer.zero_grad()
                    # measure computation time
                    compute_time.update(time.perf_counter() - end - data_time.val)
                    
                    acc1, acc5 = accuracy(output.data, target, topk=(1, 5))        
                    losses.update(loss.item(), batch_size)
                    top1.update(acc1.item(), batch_size)
                    top5.update(acc5.item(), batch_size)
                batch_time.update(time.perf_counter() - end)
                total_samples += batch_size
                
                if batch_idx % logger.log_freq == 0:
                    #progress.display(batch_idx + 1, fabric)
                    progress.display(batch_idx + 1)

                    logger.save_train_batch_metrics(
                        epoch=epoch,
                        step=batch_idx+1,
                        global_step=(epoch*max_iters) + batch_idx+1,
                        num_sampels=batch_size,
                        total_time=batch_time.val,
                        data_time=data_time.val,
                        compute_time=compute_time.val,
                        loss=losses.val,
                        acc1=top1.val,
                        acc5=top5.val,
                        avg_cpu= monitor.resource_data['cpu_util'].summarize()['mean'],
                        max_cpu= monitor.resource_data['cpu_util'].summarize()['max'],
                        avg_gpu=monitor.resource_data['gpu_util'].summarize()['mean'] if monitor.monitor_gpu else 0,
                        max_gpu=monitor.resource_data['gpu_util'].summarize()['max'] if monitor.monitor_gpu else 0,
                        )

                if batch_idx+1 >= max_iters:
                    # end loop early as max number of minibatches have been processed 
                    break
                end = time.perf_counter()
            
            logger.save_train_epoch_metrics(
                epoch=epoch,
                num_samples=total_samples,
                global_step=((epoch+1)*max_iters),
                num_batches = max_iters,
                total_time=batch_time.sum,
                data_time=data_time.sum,
                compute_time=compute_time.sum,
                loss=losses.avg,
                acc1=top1.avg,
                acc5=top5.avg,
                avg_cpu= monitor.resource_data['cpu_util'].summarize()['mean'],
                max_cpu= monitor.resource_data['cpu_util'].summarize()['max'],
                avg_gpu= monitor.resource_data['gpu_util'].summarize()['mean'] if monitor.monitor_gpu else 0,
                max_gpu= monitor.resource_data['gpu_util'].summarize()['max'] if monitor.monitor_gpu else 0,
            )
            return losses.avg, top1.avg, top5.avg
    

def make_dataloader(job_id:int, 
                    dataloader_kind:str, 
                    data_dir:str, 
                    shuffle: bool, 
                    batch_size:int, 
                    num_workers:int,
                    world_size:int,
                    super_address:str = None, 
                    cache_address:str = None, 
                    simulate_data_delay = None):
    
    dataloader = None

    if dataloader_kind == 'vanilla_pytorch':
        dataset =  VanillaTorchImageDataset(data_dir = data_dir, transform=transform())
        sampler = RandomSampler(data_source=dataset) if shuffle else SequentialSampler(data_source=dataset)
        dataloader = DataLoader(dataset=dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers)

    elif dataloader_kind == 'super_image':
        dataset =  SUPERDataset(
            job_id=job_id,
            data_dir=data_dir,
            batch_size=batch_size,
            transform=transform(),
            world_size=world_size,
            super_address=super_address,
            cache_address=cache_address,
            simulate_delay=simulate_data_delay)   
        dataloader = SUPERDataLoader(dataset=dataset, num_workers=num_workers, batch_size=None)
    else:
        raise Exception(f"unknown dataloader_kind {dataloader_kind}")
    return dataloader
   
def make_model(fabric:Fabric, model_name:str):
    if model_name in torchvision.models.list_models():
        with fabric.init_module(empty_init=True):
            return torchvision.models.get_model(model_name)                                               
    else:
        raise Exception(f"unknown model {model_name}")
    
def transform():
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transformations = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), normalize])
    return transformations

def accuracy(output: Tensor, target:Tensor, topk=(1,))-> List[Tensor]:
        """Computes the accuracy over the k top predictions for the specified values of k."""
        with no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res 



# @hydra.tmain(version_base=None, config_path="conf", config_name="config")
# def run_experiment(config: DictConfig):
   
#     start_time = time.perf_counter()

#     precision = get_default_supported_precision(training=True)
#     fabric = Fabric(accelerator=config.accelerator, devices=config.devices, strategy="auto", precision=precision)
#     exp_version = get_next_exp_version(config.log_dir,config.dataset.name)
#     config.log_dir = os.path.join(config.log_dir, config.dataset.name, str(exp_version))

#     if not config.training.max_minibatches_per_epoch:
#          config.training.max_minibatches_per_epoch = Infinity
   
#     result = fabric.launch(main, config=config)
    
#     fabric.print(f"Creating overall report for experiment")

#     create_exp_summary_report(config.log_dir)

#     fabric.print(f"Exeperiment completed. Total Duration: {time.perf_counter() - start_time}")





if __name__ == "__main__":
    setup()