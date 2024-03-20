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

class ModelInterface(metaclass=abc.ABCMeta):
    def __init__(self):
        self.name = self.__class__.__name__    

    @abc.abstractmethod
    def transform(self) -> Optional[Any]:
        raise NotImplementedError

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
                    losses.update(loss.item(), data.size(0))
                    top1.update(acc1[0], data.size(0))
                    top5.update(acc5[0], data.size(0))
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
                        acc1=top1.val,
                        acc5=top5.val)

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
                acc5=top5.avg
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

        return loss, acc1, acc5
    
    
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