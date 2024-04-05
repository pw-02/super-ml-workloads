# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import math
import os
import time
from typing import  Any, Iterable
import lightning as L
import torch
import torch.nn as nn
from lightning.fabric.utilities.throughput import measure_flops
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig
from lightning.fabric import Fabric
import mlworklaods.utils as utils
import mlworklaods.super_dl.s3utils as s3utils
from mlworklaods.super_dl.dataset.s3_text_iterable import S3TextIterableDataset
from mlworklaods.super_dl.dataset.redis_text_iterable import RedisTextIterableDataset

import tiktoken
from  mlworklaods.utils import  AverageMeter, ProgressMeter, Summary, ExperimentLogger, ResourceMonitor, create_exp_summary_report

from litgpt import Config
from litgpt.args import EvalArgs, IOArgs, TrainArgs
from litgpt.model import GPT, Block
from litgpt.utils import chunked_cross_entropy
import functools
import mlworklaods.super_dl.s3utils as s3utils
from  mlworklaods.super_dl.s3utils import S3Url
import functools
from typing import List, Tuple, Dict
from torch.utils.data import SequentialSampler, IterableDataset, RandomSampler, DataLoader
import torch.nn.functional as F
import tiktoken


class CycleIterator:
    """An iterator that cycles through an iterable indefinitely.

    Example:
        >>> iterator = CycleIterator([1, 2, 3])
        >>> [next(iterator) for _ in range(5)]
        [1, 2, 3, 1, 2]

    Note:
        Unlike ``itertools.cycle``, this iterator does not cache the values of the iterable.
    """

    def __init__(self, iterable: Iterable) -> None:
        self.iterable = iterable
        self.epoch = 0
        self._iterator = None

    def __next__(self) -> Any:
        if self._iterator is None:
            self._iterator = iter(self.iterable)
        try:
            return next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.iterable)
            self.epoch += 1
            return next(self._iterator)

    def __iter__(self):
        return self


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def setup(config: DictConfig):

    start_time = time.perf_counter()
    precision = utils.get_default_supported_precision(training=True)
    fabric = Fabric(accelerator=config.accelerator, devices=config.devices, strategy="auto", precision=precision)
    exp_version = utils.get_next_exp_version(config.log_dir,config.dataset.name)
    config.log_dir = os.path.join(config.log_dir, config.dataset.name, str(exp_version))

    train: TrainArgs = TrainArgs(
        model_name=config.training.model_name,
        dataloader_kind=config.dataloader.kind,
        global_batch_size=config.training.global_batch_size,
        micro_batch_size=config.training.micro_batch_size,
        lr_warmup_steps=config.training.lr_warmup_steps,
        epochs=config.training.epochs,
        epoch_size=config.training.epoch_size,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        beta1=config.training.beta1,
        beta2=config.training.beta2,
        max_norm=config.training.max_norm,
        min_lr=config.training.min_lr,
        num_pytorch_workers=config.dataloader.num_pytorch_workers,
        shuffle=config.dataloader.shuffle,
        max_tokens= config.training.max_tokens
    )
    io: IOArgs = IOArgs(
        train_data_dir=config.dataset.train_dir, 
        val_data_dir=config.dataset.val_dir, 
        log_dir=config.log_dir,
        log_interval=config.log_interval)
    
    eval: EvalArgs = EvalArgs(interval=config.training.eval_iterval, max_iters=config.training.max_eval_iters)
    fabric.launch(main,config.seed, config, train,io, eval)
    fabric.print(f"Creating overall report for experiment")
    create_exp_summary_report(io.log_dir)
    fabric.print(f"Experiement Ended. Total Duration {(time.perf_counter()-start_time):.2f}s")



def main(fabric: Fabric, seed: int, config: DictConfig, train: TrainArgs, io: IOArgs,eval:EvalArgs) -> None:
    fabric.seed_everything(seed, workers=True)  # same seed for every process to init model (FSDP)
    t0 = time.perf_counter()

    model = make_model(fabric, train.model_name)
    block_size = model.config.block_size
    fabric.print(f"Time to instantiate {train.model_name} model: {time.perf_counter() - t0:.02f} seconds")
    fabric.print(f"Total parameters in {train.model_name} model: {utils.num_model_parameters(model):,}")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train.learning_rate,
        weight_decay=train.weight_decay,
        betas=(train.beta1, train.beta2),
        foreach=False,
    )
    model, optimizer = fabric.setup(model,optimizer, move_to_device=True)

    tokenizer = tiktoken.get_encoding("gpt2")

    train_dataloader, val_dataloader = None, None
    if config.run_training:
        train_dataloader = make_dataloader(train.dataloader_kind, io.train_data_dir, train.shuffle, train.micro_batch_size, train.num_pytorch_workers, tokenizer, block_size)
        train_dataloader = fabric.setup_dataloaders(train_dataloader, move_to_device=True, use_distributed_sampler=True)
    if config.run_evaluation:
        train_dataloader = make_dataloader(train.dataloader_kind, io.train_data_dir, False, train.micro_batch_size, train.num_pytorch_workers, tokenizer, block_size)
        val_dataloader = fabric.setup_dataloaders(val_dataloader, move_to_device=True, use_distributed_sampler=True)
    
    
    state = {"model": model, "optimizer": optimizer, "iter_num": 0, "step_count": 0}

    train_time = time.perf_counter()
    logger = ExperimentLogger(fabric, io.log_dir, io.log_interval)
    if fabric.is_global_zero:
        logger.log_hyperparams(config)

    fit(fabric, model, optimizer, train_dataloader, val_dataloader, io, train, eval, logger)
    
    fabric.print(f"Training Finished on device {fabric.global_rank}. Training time: {(time.perf_counter()-train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")
    
    fabric.print(f"Creating job report for device {fabric.global_rank}..")
    logger.create_job_report()


def fit(fabric: L.Fabric,model:nn.Module, optimizer,train_dataloader: DataLoader,val_dataloader: DataLoader,io: IOArgs,train: TrainArgs,eval: EvalArgs, logger:ExperimentLogger) -> None:
    epoch_count = 0
    iter_num = 0
    devices = fabric.world_size

    with torch.device("meta"):
        meta_model = GPT(model.config)
        x = torch.randint(0, 1, (train.micro_batch_size, meta_model.max_seq_length))
        model_fwd = lambda: meta_model(x)
        model_loss = lambda y: chunked_cross_entropy(y, x, chunk_size=0)
        measured_flops = measure_flops(meta_model, model_fwd, model_loss)
        fabric.print(f"Measured TFLOPs: {measured_flops * fabric.world_size / 1e12:.2f}")
        del meta_model, x

    max_tokens_per_device = train.max_tokens // fabric.world_size
    tokens_per_iter = train.micro_batch_size * model.max_seq_length
    max_iters = max_tokens_per_device // tokens_per_iter
    log_iter_interval = io.log_interval

    train_iterator = CycleIterator(train_dataloader)
    step_count = 0
    total_samples = 0
    total_tokens = 0
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    compute_time = AverageMeter('Compute', ':6.3f')
    losses = AverageMeter('Loss', ':6.2f')

    progress = ProgressMeter(max_iters,
            [batch_time, data_time, compute_time, losses],
            prefix="Epoch: [{}]".format(train_iterator.epoch))


    with ResourceMonitor() as monitor:
        end = time.perf_counter()
        for input_ids, targets in train_iterator:
            data_time.update(time.perf_counter() - end)
            
            #determine and set the learning rate for this iteration
            lr = get_lr(train.learning_rate, iter_num, 0, max_iters, train.min_lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            
            is_accumulating = iter_num % train.gradient_accumulation_iters(devices) != 0
        
            with fabric.no_backward_sync(model, enabled=is_accumulating):
                logits = model(input_ids)
                loss = chunked_cross_entropy(logits, targets)
                fabric.backward(loss / train.gradient_accumulation_iters(devices))
        
            if not is_accumulating:
                fabric.clip_gradients(model, optimizer, max_norm=train.max_norm)
                optimizer.step()
                optimizer.zero_grad()
                step_count+=1
            
            # measure computation time
            compute_time.update(time.perf_counter() - end - data_time.val)

            loss = loss.item()  # expensive device-to-host synchronization
            losses.update(loss, input_ids.size(0))
            batch_time.update(time.perf_counter() - end)

            total_samples += input_ids.size(0)
            total_tokens += input_ids.size(0) * model.max_seq_length

            if iter_num % log_iter_interval == 0:
                progress.display(iter_num + 1, fabric)
                logger.save_train_batch_metrics(
                    epoch=train_iterator.epoch,
                    step=iter_num+1,
                    global_step=(train_iterator.epoch * max_iters) + iter_num+1,
                    num_sampels=input_ids.size(0),
                    num_tokens = input_ids.size(0) * model.max_seq_length,
                    total_time=batch_time.val,
                    data_time=data_time.val,
                    compute_time=compute_time.val,
                    loss=losses.val,
                    avg_cpu= monitor.resource_data['cpu_util'].summarize()['mean'],
                    max_cpu= monitor.resource_data['cpu_util'].summarize()['max'],
                    avg_gpu= monitor.resource_data['gpu_util'].summarize()['mean'],
                    max_gpu= monitor.resource_data['gpu_util'].summarize()['max'],)
        
            if train_iterator.epoch > epoch_count or iter_num +1 >= max_iters:
                logger.save_train_epoch_metrics(
                    epoch=epoch_count,
                    num_samples=total_samples,
                    num_tokens= total_tokens,
                    global_step=((epoch_count+1)*max_iters),
                    num_batches = max_iters,
                    total_time=batch_time.sum,
                    data_time=data_time.sum,
                    compute_time=compute_time.sum,
                    loss=losses.avg,
                    avg_cpu= monitor.resource_data['cpu_util'].summarize()['mean'],
                    max_cpu= monitor.resource_data['cpu_util'].summarize()['max'],
                    avg_gpu= monitor.resource_data['gpu_util'].summarize()['mean'],
                    max_gpu= monitor.resource_data['gpu_util'].summarize()['max'])
                
                epoch_count = train_iterator.epoch
                batch_time.reset()
                data_time.reset()
                compute_time.reset()
                losses.reset()

            if val_dataloader is not None and not is_accumulating and step_count % eval.interval == 0:
                    t0 = time.perf_counter()
                    val_loss = validate(fabric, model, val_dataloader, max_iters=eval.max_iters)
                    val_loss = val_loss.item()
                    td = time.perf_counter() - t0
                    fabric.print(f"iter {iter_num}: val loss {val_loss:.4f}, val time: {td * 1000:.2f} ms")
                    metrics = {"val_loss": val_loss, "val_ppl": math.exp(val_loss)}
                    fabric.log_dict(metrics, step=step_count - 1)
                    fabric.barrier()
            
            iter_num +=1
            if iter_num >= max_iters:
                break
            end = time.perf_counter()


# FSDP has issues with `inference_mode`
@torch.no_grad()
def validate(fabric: L.Fabric, model: torch.nn.Module, val_dataloader: DataLoader, max_iters: int,  logger:ExperimentLogger, epoch:int):
    fabric.barrier()
    fabric.print("Validating ...")
    model.eval()
    losses = []

    total_samples  = 0
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':6.2f')
    progress = ProgressMeter(
    max_iters,[batch_time, losses],prefix='Test: ')
    end = time.perf_counter() 
     
    for k, (input_ids, targets) in enumerate(val_dataloader):
        batch_length = input_ids.size(0)
        logits = model(input_ids)
        loss = chunked_cross_entropy(logits, targets)
        losses.update(loss.item(), batch_length)
        batch_time.update(time.perf_counter() - end)
        total_samples += len(batch_length)

        if k % logger.log_freq == 0:
            progress.display(k + 1, fabric)
            logger.save_eval_batch_metrics(
                        epoch=epoch,
                        step=k+1,
                        global_step=(epoch*max_iters) + k+1,
                        num_sampels=k,
                        batch_time=batch_time.val,
                        loss=losses.avg,
                    )
            
        if k+1 >= max_iters:
            break
        end = time.perf_counter()
    
    logger.save_eval_epoch_metrics(
                epoch=epoch,
                num_samples=total_samples,
                global_step=((epoch+1)*max_iters),
                num_batches = max_iters,
                total_time=batch_time.sum,
                loss=losses.avg,
            )

    val_loss = losses.avg
    model.train()
    fabric.barrier()
    return val_loss 

def make_dataloader(dataloader_kind:str, date_dir:str,shuffle: bool, batch_size:int, num_workers:int, tokenizer, block_size):
        dataloader = None
        if dataloader_kind == 's3_text_iterable':
            dataset =  RedisTextIterableDataset(date_dir, tokenizer, block_size,batch_size, shuffle)
            dataloader = DataLoader(dataset=dataset, batch_size=None,num_workers=num_workers)
        else:
            raise Exception(f"unknown dataloader_kind {dataloader_kind}")
        return dataloader


def make_model(fabric:Fabric, model_name:str):
    with fabric.init_module(empty_init=True):
        model =GPT(Config.from_name(name=model_name))
    model.apply(model._init_weights)
    return model





# learning rate decay scheduler (cosine with linear warmup)
def get_lr(learning_rate: float, it: int, warmup_iters: int, max_iters: int, min_lr: float) -> float:
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > max_iters, return min learning rate
    if it > max_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


if __name__ == "__main__":
   setup()
