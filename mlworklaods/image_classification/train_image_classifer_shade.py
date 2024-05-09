from typing import List
from omegaconf import DictConfig
from lightning.fabric import Fabric
import time
from torch import nn, optim, Tensor, no_grad
from torch.utils.data import DataLoader
import torchvision
from mlworklaods.args import *
from mlworklaods.utils import ResourceMonitor, get_default_supported_precision, num_model_parameters
from mlworklaods.log_utils import AverageMeter, ProgressMeter, Summary, ExperimentLogger, get_next_exp_version, create_exp_summary_report
from shade.shadedataset import ShadeDataset
from shade.shadesampler import ShadeSampler
import redis
import heapdict
import torchvision.transforms as transforms
from torch_lru.batch_sampler_with_id import BatchSamplerWithID
import math

# #Initialization of local cache, PQ and ghost cache
#Initialization of local cache, PQ and ghost cache
red_local = redis.StrictRedis(host='172.17.0.2', port='6379')
PQ = heapdict.heapdict()
ghost_cache = heapdict.heapdict()
key_counter  = 0


def launch_shade_job(pid:int, config: DictConfig, train_args: TrainArgs, io_args: IOArgs):
    start_time = time.perf_counter()
    precision = get_default_supported_precision(training=True)
    fabric = Fabric(accelerator=train_args.accelerator, devices=train_args.devices, strategy="auto", precision=precision)

    fabric.launch(train_model, train_args.seed, config, train_args, io_args)

    fabric.print(f"Creating overall report for experiment")
    
    output_file_path = create_exp_summary_report(io_args.log_dir)

    fabric.print(f"Job Ended. Total Duration {(time.perf_counter()-start_time):.2f}s")

def train_model(fabric: Fabric, seed: int, config: DictConfig, train_args: TrainArgs, io_args: IOArgs,) -> None:        
    fabric.seed_everything(seed)  # same seed for every process to init model (FSDP)
    #setup model 
    t0 = time.perf_counter()
    model:nn.Module = make_model(fabric, train_args.model_name) 
    fabric.print(f"Time to instantiate {train_args.model_name} model: {time.perf_counter() - t0:.02f} seconds")
    fabric.print(f"Total parameters in {train_args.model_name} model: {num_model_parameters(model):,}")
    optimizer = optim.Adam(model.parameters(), lr=train_args.learning_rate)

    model, optimizer = fabric.setup(model,optimizer, move_to_device=True)
    train_dataloader, val_dataloader = None, None
    if train_args.run_training:
        train_dataloader = make_dataloader(
            fabric,
            io_args.train_data_dir, 
            train_args.global_batch_size, 
            train_args.num_pytorch_workers,
            io_args.cache_address,
            io_args.working_set_size,
            io_args.replication_factor)
        
        train_dataloader = fabric.setup_dataloaders(train_dataloader, move_to_device=True, use_distributed_sampler=True)
    if train_args.run_evaluation:
        val_dataloader = make_dataloader(
            train_args.job_id,
            io_args.val_data_dir, 
            train_args.global_batch_size, 
            train_args.num_pytorch_workers,
            io_args.cache_address,
            io_args.replication_factor)
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
    
    batch_wts = []
    for j in range(train_args.global_batch_size):
        batch_wts.append(math.log(j+10))

    for epoch in range(1, train_args.epochs +1):
        if train_dataloader:
            max_iters = min(len(train_dataloader), train_args.epoch_max_iters(fabric.world_size)) 
            fabric.print(f"Starting training loop for epoch {epoch}")
            model.train(mode=True)
            loss_train, acc1_train, acc5_train = train_loop(fabric, epoch, model, optimizer, train_dataloader, max_iters, logger,train_args.dataload_only,batch_wts)
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
                dataload_only:bool,
                batch_wts):
        global PQ
        
        global key_id_map
        global key_counter
        global red_local
        global ghost_cache
        key_id_map = redis.StrictRedis(host='172.17.0.2', port='6379')
        epoch_cache_hits_count = 0
        total_samples = 0
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        compute_time = AverageMeter('Compute', ':6.3f')
        losses = AverageMeter('Loss', ':6.2f')
        top1 = AverageMeter('Acc1', ':6.2f')
        top5 = AverageMeter('Acc5', ':6.2f')
        
        cache_hit_ratio = AverageMeter('hit%', ':6.2f')

        progress = ProgressMeter(max_iters,
            [batch_time, data_time, compute_time, losses, cache_hit_ratio],
            prefix="Epoch: [{}]".format(epoch))
        
        with ResourceMonitor() as monitor:
            end = time.perf_counter()
            for batch_idx,(images, target, cache_hit_count, batch_id) in enumerate(train_dataloader):
                data_time.update(time.perf_counter() - end)
                epoch_cache_hits_count +=cache_hit_count
                cache_hit_ratio.update(cache_hit_count/images.size(0), 1)
               
                batch_size = images.size(0)
                if not dataload_only:
                    is_accumulating = False
                    # Forward pass and loss calculation
                    with fabric.no_backward_sync(model, enabled=is_accumulating):
                        output:Tensor = model(images)
                        criterion = nn.CrossEntropyLoss(reduce = False).cuda(0)
                        item_loss = criterion(output,target)
                        # item_loss = nn.functional.cross_entropy(input=output, target=target, reduce = False).cpu()
                        
                        loss = item_loss.mean()
                        fabric.backward(loss) # .backward() accumulates when .zero_grad() wasn't called
                        
                        train_dataloader.sampler.sampler.pass_batch_important_scores(item_loss)

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

                key_counter = red_local.dbsize()
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
                batch_time.update(time.perf_counter() - end)
                total_samples += batch_size
                train_dataloader.dataset.set_PQ(PQ)
                train_dataloader.dataset.set_ghost_cache(ghost_cache)
                train_dataloader.dataset.set_num_local_samples(key_counter)

                if batch_idx % logger.log_freq == 0:
                    #progress.display(batch_idx + 1, fabric)
                    progress.display(batch_idx + 1)
                    logger.save_train_batch_metrics(
                        epoch=epoch,
                        step=batch_idx+1,
                        global_step=((epoch-1)*max_iters) + batch_idx+1,
                        num_sampels=batch_size,
                        total_time=batch_time.val,
                        data_time=data_time.val,
                        compute_time=compute_time.val,
                        cache_hits=cache_hit_count,
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
                global_step=((epoch)*max_iters),
                num_batches = max_iters,
                total_time=batch_time.sum,
                data_time=data_time.sum,
                compute_time=compute_time.sum,
                loss=losses.avg,
                acc1=top1.avg,
                acc5=top5.avg,
                cache_hits=epoch_cache_hits_count,
                avg_cpu= monitor.resource_data['cpu_util'].summarize()['mean'],
                max_cpu= monitor.resource_data['cpu_util'].summarize()['max'],
                avg_gpu= monitor.resource_data['gpu_util'].summarize()['mean'] if monitor.monitor_gpu else 0,
                max_gpu= monitor.resource_data['gpu_util'].summarize()['max'] if monitor.monitor_gpu else 0,
            )
            train_dataloader.sampler.sampler.on_epoch_end(losses.avg)
            # indices_for_process = train_dataloader.sampler.get_indices_for_process()
            # indices_for_process_per_epoch = indices_for_process_per_epoch.append({
            #      'epoch': epoch,
            #      'indices': indices_for_process,
            #      }
            #      , ignore_index=True
            #      )
            return losses.avg, top1.avg, top5.avg
    

def make_dataloader(
        fabric:Fabric,
        data_dir:str, 
        batch_size:int, 
        num_workers:int,
        cache_address:str = None, 
        working_set_size = None,
        replication_factor = None):

        dataloader = None

        dataset =  ShadeDataset(data_dir = data_dir, 
                                transform=transform(),
                                target_transform=None,
                                cache_data=True if cache_address is not None else False,
                                PQ=PQ,
                                ghost_cache=ghost_cache,
                                key_counter=key_counter,
                                wss=working_set_size,
                                cache_address=cache_address
                                )
        sampler = ShadeSampler(dataset,
                               num_replicas=fabric.world_size,
                               rank=fabric.node_rank, 
                               batch_size=batch_size, 
                               seed = 41, 
                               cache_address = cache_address, 
                               rep_factor=replication_factor
                               )
        
        batch_sampler = BatchSamplerWithID(sampler=sampler, batch_size=batch_size, drop_last=False)

        dataloader = DataLoader(dataset=dataset, sampler=batch_sampler, batch_size=None, num_workers=num_workers)

        return dataloader
   
def make_model(fabric:Fabric, model_name:str):
    if model_name in torchvision.models.list_models():
        with fabric.init_module(empty_init=True):
            return torchvision.models.get_model(model_name)                                               
    else:
        raise Exception(f"unknown model {model_name}")
    
def transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # transformations = transforms.Compose([torchvision.transforms.ToTensor(), normalize])
    transformations = transforms.Compose([
                # transforms.RandomResizedCrop(224),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])

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
    launch_job()