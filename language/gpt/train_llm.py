import time
from lightning.fabric import Fabric
from argparse import Namespace
import torch.optim as optim
from super_client import SuperClient
from super_dl.logger import SUPERLogger
from super_dl.utils import *
import torch.optim as optim
import torch.nn
from torch.utils.data import DataLoader
from  super_dl.utils import chunked_cross_entropy

def run_llm_training(fabric: Fabric, model:torch.nn.Module, optimizer:optim.Optimizer, scheduler:optim.lr_scheduler.LRScheduler,
                train_dataloader: DataLoader, val_dataloader: DataLoader, hparams:Namespace,
                 logger:SUPERLogger, super_client:SuperClient = None) -> None:
    
    for epoch in range(hparams.epochs):

        if hparams.run_evaluate:
            fabric.print(f"Running validation loop for epoch {epoch}")

            if hparams.max_minibatches_per_epoch:
                total_batches = min(hparams.max_minibatches_per_epoch, len(val_dataloader))
            else:
                total_batches = len(val_dataloader)

            process_data(fabric,
                         dataloader=val_dataloader,
                         global_step=epoch * total_batches,
                         model=model,
                         block_size=hparams.block_Size,
                         optimizer=optimizer,
                         logger=logger,
                         epoch=epoch,
                         hparams=hparams,
                         is_training=False,
                         super_client=super_client,
                         total_batches=total_batches)
            
        if hparams.run_training:
            fabric.print(f"Running training loop for epoch {epoch}")

            if hparams.max_minibatches_per_epoch:
                total_batches = min(hparams.max_minibatches_per_epoch, len(train_dataloader))
            else:
                total_batches= len(train_dataloader)

            process_data(fabric=fabric,
                         dataloader=train_dataloader,
                         global_step=epoch * total_batches,
                         model=model,
                         optimizer=optimizer,
                         logger=logger,
                         epoch=epoch,
                         hparams=hparams,
                         is_training=True,
                         super_client=super_client,
                         total_batches=total_batches)

    logger.job_end()
  


def process_data(fabric: Fabric, dataloader: DataLoader,
                 global_step:int, model:torch.nn.Module, 
                 optimizer:optim.SGD, logger:SUPERLogger, epoch, hparams:Namespace,
                 total_batches:int, is_training, super_client:SuperClient): 
    
    logger.epoch_start(epoch_length=total_batches,is_training=is_training)
    model.train(is_training)
    end = time.perf_counter()
    start_time = time.time()
    for iteration, (inputs, targets,) in enumerate(dataloader):
        cache_hit = False
        batch_id = iteration
        data_transform_time = 0
        num_sampels = inputs.size(0)
        data_time = time.perf_counter() - end
        data_fetch_time = data_time/inputs.shape[0]
        # Accumulate gradient x batches at a time
            
        is_accumulating = hparams.grad_acc_steps is not None and iteration % hparams.grad_acc_steps != 0

        if hparams.profile:
            torch.cuda.synchronize()

        # Forward pass and loss calculation
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits:torch.Tensor = model(inputs)
            loss = chunked_cross_entropy(logits, targets, chunk_size=0)
            #loss = chunked_cross_entropy(logits[..., :-1, :], targets[..., 1:])
            if is_training:
                fabric.backward(loss/1) # .backward() accumulates when .zero_grad() wasn't called
        if not is_accumulating and is_training:
            # Step the optimizer after accumulation phase is over
            optimizer.step()
            optimizer.zero_grad()
        
        if hparams.profile:
            torch.cuda.synchronize()    
          
        iteration_time = time.perf_counter()-end
        compute_time = iteration_time - data_time
        # measure accuracy and record loss
        # prec1, prec5 = accuracy(logits.data, targets, topk=(1, 5))
  
        metrics_dict = logger.record_iteration_metrics(
            epoch=epoch,
            step=iteration,
            global_step = global_step,
            num_sampels=num_sampels,
            iteration_time=iteration_time,
            data_time=data_time,
            compute_time=compute_time,
            compute_ips=  calc_throughput_per_second(num_sampels,compute_time),
            total_ips=calc_throughput_per_second(num_sampels,iteration_time),
            loss = to_python_float(loss.detach()),
            top1=0,
            top5=0,
            batch_id=batch_id,
            is_training=is_training,
            cache_hit = cache_hit,
            data_fetch_time=data_fetch_time,
            data_transform_time=data_transform_time
            )
    
        global_step+=1

        if hparams.max_minibatches_per_epoch and iteration >= hparams.max_minibatches_per_epoch - 1:
            # end epoch early based on num_minibatches that have been processed 
            break

        if iteration == len(dataloader) - 1:
            break

        end = time.perf_counter()
        start_time = time.time()
    
    logger.epoch_end(epoch, is_training=is_training)

def cross_entropy(output, labels, _fp16=False):
    """From pretrain_gpt2:forward_step()"""
    """
    if self.fp16_lm_cross_entropy:
        assert output.dtype == torch.half
        loss = mpu.vocab_parallel_cross_entropy(output, labels)
    else:
        loss = mpu.vocab_parallel_cross_entropy(output.float(), labels)
        return loss
    """
    labels, loss_mask = labels[0], labels[1]
    if _fp16:
        assert output.dtype == torch.half and loss_mask.dtype == torch.half
        losses = mpu.vocab_parallel_cross_entropy(output.contiguous(), labels)
    else:
        losses = mpu.vocab_parallel_cross_entropy(output.float().contiguous(), labels)
    loss_mask = loss_mask.view(-1)
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
    return loss

def accuracy(output: torch.Tensor, target:torch.Tensor, topk=(1,))-> List[torch.Tensor]:
    """Computes the accuracy over the k top predictions for the specified
    values of k."""
    with torch.no_grad():
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



