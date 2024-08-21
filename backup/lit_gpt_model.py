
import abc
from ast import Dict
import os
import random
import time
from functools import cached_property
from io import IOBase
from typing import Optional, Any, Tuple, Union, List
import torch
import torch.nn as nn
from PIL import Image
from omegaconf import DictConfig
# from s3torchconnector import S3Reader, S3Checkpoint
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import v2
from lightning.fabric import Fabric
from utils import ResourceMonitor, num_model_parameters
from logutil import ExperimentLogger, AverageMeter, ProgressMeter, Summary
from args import TrainArgs, EvalArgs, IOArgs
from mlworklaods.language.litgpt.config import Config
from mlworklaods.language.litgpt.model import GPT, Block

class LitGPTModel():
    def __init__(self, fabric: Fabric, model_name:str, learning_rate,weight_decay,beta1, beta2 ):
        self.model_name = model_name

        with fabric.init_module(empty_init=True):
            model:nn.Module =GPT(Config.from_name(name=model_name))
        model.apply(model._init_weights)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(beta1,beta2),
            foreach=False,
        )
        self._model, self._optimizer = fabric.setup(model, optimizer, move_to_device=True)
        self._num_parameters = num_model_parameters(self._model)

    @property
    def model(self)->nn.Module:
        return self._model
    
    @property
    def optimizer(self):
        return self._optimizer
     
    @property
    def num_parameters(self):
        return self._num_parameters
    
    def chunked_cross_entropy(self,
                              logits: Union[torch.Tensor, List[torch.Tensor]],
                              targets: torch.Tensor,
                              chunk_size: int = 128,
                              ignore_index: int = -1,
                              ) -> torch.Tensor:
        # with large max_sequence_lengths, the beginning of `backward` allocates a large memory chunk which can dominate
        # the memory usage in fine-tuning settings with low number of parameters.
        # as a workaround hack, the cross entropy computation is chunked to force it to deallocate on the go, reducing
        # the memory spike's magnitude

        # lm_head was chunked (we are fine-tuning)
        if isinstance(logits, list):
            # don't want to chunk cross entropy
            if chunk_size == 0:
                logits = torch.cat(logits, dim=1)
                logits = logits.reshape(-1, logits.size(-1))
                targets = targets.reshape(-1)
                return torch.nn.functional.cross_entropy(logits, targets, ignore_index=ignore_index)

            # chunk cross entropy
            logit_chunks = [logit_chunk.reshape(-1, logit_chunk.size(-1)) for logit_chunk in logits]
            target_chunks = [target_chunk.reshape(-1) for target_chunk in targets.split(logits[0].size(1), dim=1)]
            loss_chunks = [
                torch.nn.functional.cross_entropy(logit_chunk, target_chunk, ignore_index=ignore_index, reduction="none")
                for logit_chunk, target_chunk in zip(logit_chunks, target_chunks)
            ]
            non_masked_elems = (targets != ignore_index).sum()
            return torch.cat(loss_chunks).sum() / max(1, non_masked_elems)

        # no chunking at all
        logits = logits.reshape(-1, logits.size(-1))
        targets = targets.reshape(-1)
        if chunk_size == 0:
            return torch.nn.functional.cross_entropy(logits, targets, ignore_index=ignore_index)

        # lm_head wasn't chunked, chunk cross entropy
        logit_chunks = logits.split(chunk_size)
        target_chunks = targets.split(chunk_size)
        loss_chunks = [
            torch.nn.functional.cross_entropy(logit_chunk, target_chunk, ignore_index=ignore_index, reduction="none")
            for logit_chunk, target_chunk in zip(logit_chunks, target_chunks)
        ]
        non_masked_elems = (targets != ignore_index).sum()
        return torch.cat(loss_chunks).sum() / max(1, non_masked_elems)
    
   
    def fit(self,fabric: Fabric, devices:int, state:Dict,train_dataloader: DataLoader, val_dataloader: DataLoader, train: TrainArgs, eval: EvalArgs, logger: ExperimentLogger):
        
        total_t0 = time.perf_counter()
        train_iter = iter(train_dataloader)        
        for state["iter_num"] in range(state["iter_num"], train.max_iters(devices)):
            iter_num = state["iter_num"] + 1
            iter_t0 = time.perf_counter()
            input_ids, targets = next(train_iter)
            #fabric.print(time.perf_counter()-iter_t0)
            
            is_accumulating = iter_num % train.gradient_accumulation_iters(devices) != 0

            with fabric.no_backward_sync(model, enabled=is_accumulating):
                logits = self.model(input_ids)
                loss = self.chunked_cross_entropy(logits, targets, chunk_size=0)
                fabric.backward(loss / train.gradient_accumulation_iters(devices))

            if not is_accumulating:
                fabric.clip_gradients(self.model, self.optimizer, max_norm=train.max_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()
                state["step_count"] += 1
            
            



  

        
        
        
        for epoch in range(0, train.epochs):
            if train_dataloader:
                self.model.train(mode=True)
                fabric.print(f"Stating training loop for epoch {epoch}")
                self.train(epoch=epoch, fabric=fabric,devices=devices,train_dataloader=train_dataloader, train=train, logger=logger)
            if val_dataloader:
                self.model.eval(mode=True)
                fabric.print(f"Stating evaluation loop for epoch {epoch}")
                self.validate(epoch=epoch, val_dataloader=val_dataloader, eval=eval, logger=logger)

    def validate(self,epoch:int, val_dataloader: DataLoader, eval: EvalArgs, logger: ExperimentLogger):
        batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
        losses = AverageMeter('Loss', ':6.2f')
        top1 = AverageMeter('Acc1', ':6.2f', Summary.AVERAGE)
        top5 = AverageMeter('Acc5', ':6.2f', Summary.AVERAGE)
        progress = ProgressMeter(
        eval.max_iters,[batch_time, losses, top1, top5],prefix='Test: ')
        end = time.perf_counter()  
        total_samples  = 0       
        with torch.no_grad():
            for iter_num,(data, target) in enumerate(val_dataloader):
                # NOTE: no need to call `.to(device)` on the data, target
                output:torch.Tensor = self.model(data)
                loss_eval = torch.nn.functional.cross_entropy(output, target)
                # measure accuracy and record loss
                acc1, acc5 = self.accuracy(output, target, topk=(1, 5))
                losses.update(loss_eval.item(), data.size(0))
                top1.update(acc1[0], data.size(0))
                top5.update(acc5[0], data.size(0))
                batch_time.update(time.perf_counter() - end)
                
                total_samples += len(data)

                if iter_num % logger.print_freq == 0:
                    progress.display(iter_num + 1)
            
                if iter_num % logger.log_freq == 0:
                    logger.save_eval_batch_metrics(
                        epoch=epoch,
                        step=iter_num+1,
                        global_step=(epoch*eval.max_iters) + iter_num+1,
                        num_sampels=len(data),
                        batch_time=batch_time.val,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg
                    )
                if  iter_num >= eval.max_iters:
                    # end loop early as max number of minibatches have been processed 
                    break
            end = time.perf_counter()
        
        logger.save_eval_epoch_metrics(
                epoch=epoch,
                num_samples=total_samples,
                global_step=((epoch+1)*eval.max_iters),
                num_batches = eval.max_iters,
                total_time=batch_time.sum,
                loss=losses.avg,
                top1=top1.avg,
                top5=top5.avg
            )
        return top1.avg, top5.avg

    def train(self,
            epoch:int,
            fabric: Fabric, 
            devices: int, 
            train_dataloader: DataLoader, 
            train: TrainArgs,
            logger: ExperimentLogger):
        
        total_samples  = 0
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        compute_time = AverageMeter('Compute', ':6.3f')
        losses = AverageMeter('Loss', ':6.2f')
        top1 = AverageMeter('Acc1', ':6.2f')
        top5 = AverageMeter('Acc5', ':6.2f')
        progress = ProgressMeter(
            train.max_iters,
            [batch_time, data_time, compute_time, losses, top1, top5],
            prefix="Epoch: [{}]".format(epoch))
        
        with ResourceMonitor() as monitor:
            end = time.perf_counter()   
            for iter_num,(data, target) in enumerate(train_dataloader):
                # measure data loading delay
                data_time.update(time.perf_counter() - end)

                is_accumulating = iter_num % train.gradient_accumulation_iters(devices) != 0

                #train model on next batch
                with self.fabric.no_backward_sync(self.model, enabled=is_accumulating):
                    output:torch.Tensor = self.model(data)
                    loss_train = torch.nn.functional.cross_entropy(output, target)
                    fabric.backward(loss_train) # .backward() accumulates when .zero_grad() wasn't called

                if not is_accumulating:
                    # Step the optimizer after accumulation phase is over
                    self.optimizer.step()
                    self.optimizer.zero_grad()
        
                acc1, acc5 = self.accuracy(output.data, target, topk=(1, 5))
                compute_time.update(time.perf_counter() - end - data_time.val)   # measure compute tim

                losses.update(loss_train.item(), data.size(0))
                top1.update(acc1, data.size(0))
                top5.update(acc5, data.size(0))
                
                batch_time.update(time.perf_counter() - end)
                total_samples += len(data)

                if iter_num % logger.print_freq == 0:
                    progress.display(iter_num + 1)
                
                if iter_num % logger.log_freq == 0:
                    logger.save_train_batch_metrics(
                        epoch=epoch,
                        step=iter_num+1,
                        global_step=(epoch*train.max_iters) + iter_num+1,
                        num_sampels=len(data),
                        total_time=batch_time.val,
                        data_time=data_time.val,
                        compute_time=compute_time.val,
                        loss=losses.val,
                        avg_cpu= monitor.resource_data['cpu_util'].summarize()['mean'],
                        max_cpu= monitor.resource_data['cpu_util'].summarize()['max'],
                        avg_gpu= monitor.resource_data['gpu_util'].summarize()['mean'],
                        max_gpu= monitor.resource_data['gpu_util'].summarize()['max'],
                        )

                if iter_num+1 >= train.max_iters:
                    # end loop early as max number of minibatches have been processed 
                    break
                end = time.perf_counter()

            logger.save_train_epoch_metrics(
                epoch=epoch,
                num_samples=total_samples,
                global_step=((epoch+1)*train.max_iters),
                num_batches = train.max_iters,
                total_time=batch_time.sum,
                data_time=data_time.sum,
                compute_time=compute_time.sum,
                loss=losses.avg,
                acc1=top1.avg,
                acc5=top5.avg,
                avg_cpu= monitor.resource_data['cpu_util'].summarize()['mean'],
                max_cpu= monitor.resource_data['cpu_util'].summarize()['max'],
                avg_gpu= monitor.resource_data['gpu_util'].summarize()['mean'],
                max_gpu= monitor.resource_data['gpu_util'].summarize()['max'],
            )
        return top1.avg, top5.avg
    
    def accuracy(self, output: torch.Tensor, target:torch.Tensor, topk=(1,))-> List[torch.Tensor]:
        """Computes the accuracy over the k top predictions for the specified values of k."""
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



class TorchVisionModel():
    def __init__(self, fabric: Fabric, model_name):
        super().__init__()
        self.name = model_name
        self.fabric:Fabric = fabric

        self._model, self._optimizer = fabric.setup(model, optimizer, move_to_device=True)
    
    def train(self, epoch, train_dataloader: DataLoader, logger: ExperimentLogger, max_minibatches: int, grad_acc_steps: int):
        return super().train(epoch, train_dataloader, logger, max_minibatches, grad_acc_steps)
        

    @property
    def model(self):
        return self._model
    
    @property
    def optimizer(self):
        return self._optimizer
    
    @cached_property
    def transform(self):
        normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transformations = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), normalize])
        return transformations

    def train_batch(self, data, target, is_accumulating:bool):
        # Forward pass and loss calculation
        with self.fabric.no_backward_sync(self.model, enabled=is_accumulating):
            output:torch.Tensor = self.model(data)
            loss = torch.nn.functional.cross_entropy(output, target)
            self.fabric.backward(loss) # .backward() accumulates when .zero_grad() wasn't called

        if not is_accumulating:
            # Step the optimizer after accumulation phase is over
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        acc1, acc5 = self.accuracy(output.data, target, topk=(1, 5))

        return loss.item(), acc1[0], acc5[0]
    
    
    def eval_batch(self, int, data, target):
        pass

    def accuracy(self, output: torch.Tensor, target:torch.Tensor, topk=(1,))-> List[torch.Tensor]:
        """Computes the accuracy over the k top predictions for the specified values of k."""
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
    def save(self):
        raise NotImplementedError
