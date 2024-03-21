import abc
import os
import random
import time
from functools import cached_property
from io import IOBase
from typing import Optional, Any, Tuple, Union, List

from numpy import Infinity
import torch
import torch.nn as nn
from PIL import Image
from omegaconf import DictConfig
# from s3torchconnector import S3Reader, S3Checkpoint
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import v2
from lightning.fabric import Fabric
import torchvision
from utils import ResourceMonitor, num_model_parameters
from logutil import ExperimentLogger, AverageMeter, ProgressMeter, Summary

from mlworklaods.language.lit_gpt.config import Config
from mlworklaods.language.lit_gpt.model import GPT, Block
import tiktoken
# from lightning.fabric.utilities import measure_flops

class ModelInterface(metaclass=abc.ABCMeta):
    def __init__(self):
        self.name = self.__class__.__name__  


    
    @abc.abstractmethod
    def train_batch(self, batch_idx: int, data, target, is_accumulating) -> Optional[Any]:
        raise NotImplementedError
    
    @abc.abstractmethod
    def eval_batch(self, batch_idx: int, data, target) -> Optional[Any]:
        raise NotImplementedError
    

    def validate(self, epoch:int,val_dataloader: DataLoader, logger:ExperimentLogger, max_minibatches: int):
        totoal_batches = min(len(val_dataloader), max_minibatches)
        batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
        losses = AverageMeter('Loss', ':6.2f')
        top1 = AverageMeter('Acc1', ':6.2f', Summary.AVERAGE)
        top5 = AverageMeter('Acc5', ':6.2f', Summary.AVERAGE)
        progress = ProgressMeter(
        totoal_batches,[batch_time, losses, top1, top5],prefix='Test: ')
        end = time.perf_counter()  
        total_samples  = 0
        for batch_idx,(data, target) in enumerate(val_dataloader):
            loss, acc1, acc5 = self.eval_batch(batch_idx, data, target)
            losses.update(loss.item(), data.size(0))
            top1.update(acc1[0], data.size(0))
            top5.update(acc5[0], data.size(0))
            # measure elapsed time
            batch_time.update(time.perf_counter() - end)

            total_samples += len(data)

            if batch_idx % logger.print_freq == 0:
                    progress.display(batch_idx + 1)
            
            if batch_idx % logger.log_freq == 0:
                    logger.save_eval_batch_metrics(
                        epoch=epoch,
                        step=batch_idx+1,
                        global_step=(epoch*totoal_batches) + batch_idx+1,
                        num_sampels=len(data),
                        batch_time=batch_time.val,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg
                    )
            if max_minibatches and batch_idx >= max_minibatches:
                    # end loop early as max number of minibatches have been processed 
                    break
            end = time.perf_counter()
        
        logger.save_eval_epoch_metrics(
                epoch=epoch,
                num_samples=total_samples,
                global_step=((epoch+1)*totoal_batches),
                num_batches = totoal_batches,
                total_time=batch_time.sum,
                loss=losses.avg,
                top1=top1.avg,
                top5=top5.avg
            )
        return top1.avg, top5.avg


    def train(self, epoch, train_dataloader: DataLoader, logger:ExperimentLogger, max_minibatches:int, grad_acc_steps:int ):
        totoal_batches = min(len(train_dataloader), max_minibatches)
        total_samples  = 0
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        compute_time = AverageMeter('Compute', ':6.3f')
        losses = AverageMeter('Loss', ':6.2f')
        top1 = AverageMeter('Acc1', ':6.2f')
        top5 = AverageMeter('Acc5', ':6.2f')
        progress = ProgressMeter(
            totoal_batches,
            [batch_time, data_time, compute_time, losses, top1, top5],
            prefix="Epoch: [{}]".format(epoch))
        
        with ResourceMonitor() as monitor:
            end = time.perf_counter()   
            for batch_idx,(data, target) in enumerate(train_dataloader):
                # measure data loading delay
                data_time.update(time.perf_counter() - end)

                is_accumulating = grad_acc_steps is not None and batch_idx % grad_acc_steps != 0
                
                #train model on next batch
                result = self.train_batch(data, target, is_accumulating)
                if result:
                    loss, acc1, acc5 = result
                    losses.update(loss, data.size(0))
                    top1.update(acc1, data.size(0))
                    top5.update(acc5, data.size(0))
                # measure computation time
                compute_time.update(time.perf_counter() - end - data_time.val)
                
                batch_time.update(time.perf_counter() - end)
                total_samples += len(data)

                if batch_idx % logger.print_freq == 0:
                    progress.display(batch_idx + 1)
                
                if batch_idx % logger.log_freq == 0:
                    logger.save_train_batch_metrics(
                        epoch=epoch,
                        step=batch_idx+1,
                        global_step=(epoch*totoal_batches) + batch_idx+1,
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

                if max_minibatches and batch_idx+1 >= max_minibatches:
                    # end loop early as max number of minibatches have been processed 
                    break
                end = time.perf_counter()

            logger.save_train_epoch_metrics(
                epoch=epoch,
                num_samples=total_samples,
                global_step=((epoch+1)*totoal_batches),
                num_batches = totoal_batches,
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
        
    @property
    @abc.abstractmethod
    def model(self):
        """Property representing the model"""
        raise NotImplementedError

    @property
    def num_parameters(self):
        if self.name == 'EmptyModel':
            return 0
        return num_model_parameters(self.model)
    
    @abc.abstractmethod
    def save(self, **kwargs):
        """Save checkpoint"""
        raise NotImplementedError
    
    @property
    def block_size(self) -> Optional[Any]:
        raise NotImplementedError
    

class TorchVisionModel(ModelInterface):
    def __init__(self, fabric: Fabric, model_name):
        super().__init__()
        self.name = model_name
        self.fabric:Fabric = fabric
        
        with fabric.init_module(empty_init=True):
            model = torchvision.models.get_model(model_name)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        self._model, self._optimizer = fabric.setup(model, optimizer, move_to_device=True)

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

class EmptyModel(ModelInterface):
    """
    This is not really a training model as it does not train anything. Instead, this model simply reads the binary
    object data from S3, so that we may identify the max achievable throughput for a given dataset.
    """
    def __init__(self, num_labels: int = None):
        super().__init__()
        self.num_labels = num_labels

    def train_batch(self, data, target, is_accumulating:bool):
        pass
    
    def eval_batch(self, batch_idx: int, data, target):
        pass

    def model(self):
        pass

    @cached_property
    def transform(self):
        return torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    
    def save(self):
        raise NotImplementedError


class LitGPTModel(ModelInterface):

    def __init__(self, fabric: Fabric, model_name:str, learning_rate, weight_decay, beta1, beta2 ):
        super().__init__()
        self.name = model_name
   
        self.fabric:Fabric = fabric
        
        with fabric.init_module(empty_init=True):
            model =GPT(Config.from_name(name=model_name))
        model.apply(model._init_weights)

        optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(beta1,beta2),
        foreach=False,
        )

        self._model, self._optimizer = fabric.setup(model, optimizer, move_to_device=True)

    @property
    def model(self):
        return self._model
    
    @property
    def optimizer(self):
        return self._optimizer
    
    @property
    def block_size(self):
        return self.model.config.block_size
    
    @cached_property
    def transform(self):
        normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transformations = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), normalize])
        return transformations
    
    @cached_property
    def tokenizer(self):
       
        return tiktoken.get_encoding("gpt2")

    def train_batch(self, input_ids, targets, is_accumulating:bool):

        
        # Forward pass and loss calculation
        with self.fabric.no_backward_sync(self.model, enabled=is_accumulating):
            output:torch.Tensor = self.model(input_ids)
            loss = self.chunked_cross_entropy(output, targets, chunk_size=0)
            self.fabric.backward(loss) # .backward() accumulates when .zero_grad() wasn't called

        if not is_accumulating:
            # Step the optimizer after accumulation phase is over
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        # acc1, acc5 = self.accuracy(output.data, target, topk=(1, 5))

        return loss.item(), 0, 0
    
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






