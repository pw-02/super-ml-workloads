# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch.nn as nn

import argparse
from collections import OrderedDict
import datetime
import math
import os
import random
import time
from typing import List, Tuple, Union
import hydra
from omegaconf import DictConfig
from lightning.fabric import Fabric, seed_everything
from lightning.fabric.loggers import CSVLogger
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from multi_modal.albef.model import albef_model_for_retrieval
# from model import albef_model_for_retrieval
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from lightning.fabric.strategies import FSDPStrategy
from torch.utils.data import DataLoader
from dataloading.s3_redis.s3redis_retrieval_dataset import S3RedisRetrievalTrainingDataset
from torch.utils.data import RandomSampler, SequentialSampler
import torch.optim as optim
from transformers.models.bert.tokenization_bert import BertTokenizer
from torch.nn.utils.rnn import pad_sequence

from torchtext.transforms import PadTransform, Sequential, ToTensor, Truncate
import re
from torchvision import transforms
import yaml

from litgpt.utils import (
    get_default_supported_precision,
    num_parameters,
    parse_devices,
    save_hyperparameters,
)

# mean and standard deviation from the ALBEF repo:
# https://github.com/salesforce/ALBEF/blob/main/dataset/__init__.py#L16
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD_DEV = (0.26862954, 0.26130258, 0.27577711)

class ALBEFTextTransform:
    """
    Remove punctuations and trailing spaces in input text and transform it into
    a Tensor of token ids using BERTTokenizer.

    Args:
        pretrained_tokenizer (str): Pretrained tokenizer to use.
            Default: "bert-base-uncased"
        do_pre_process (bool): Whether to pre-process input text.
            Defaults to True.
        truncate (bool): Whether to truncate input text to max_seq_length.
            Defaults to False.
        pad_to_max_seq_len (bool): Whether to pad the sequence to max_seq_length.
        add_end_token (bool): Whether to add the end-of-sentence token.
            Defaults to True.
        max_seq_len (int): The max sequence length after truncating or padding.
            Defaults to 25.
        cls_token_id (int): Value to represent the start of each text.
            Defaults to 101, Hugging Face's BERT cls token id.
        sep_token_id (int): Value to represent the end of each text.
            Defaults to 102, Hugging Face's BERT sep token id.
        pad_token_id (int): Value with which to pad each text so that all texts are the same length.
            Defaults to 0, Hugging Face's BERT pad token id.

    Inputs:
        text (Union[List[str], str]): Input text to transform.
    """

    def __init__(
        self,
        pretrained_tokenizer: str = "bert-base-uncased",
        do_pre_process: bool = True,
        truncate: bool = False,
        pad_to_max_seq_len: bool = False,
        add_end_token: bool = True,
        max_seq_len: int = 25,
        cls_token_id: int = 101,
        sep_token_id: int = 102,
        pad_token_id: int = 0,
    ):
        self.do_pre_process = do_pre_process
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        self.pad_token_id = pad_token_id
        self.add_end_token = add_end_token

        self.tokenizer = BertTokenizer.from_pretrained(pretrained_tokenizer)
        self.transform = Sequential(
            Truncate(max_seq_len=max_seq_len) if truncate else torch.nn.Identity(),
            ToTensor(padding_value=self.pad_token_id),
            (
                PadTransform(max_length=max_seq_len, pad_value=self.pad_token_id)
                if pad_to_max_seq_len
                else torch.nn.Identity()
            ),
        )

    def pre_process(self, text: str) -> str:
        text = (
            re.sub(
                r"([,.'!?\"()*#:;~])",
                "",
                text,
            )
            .replace("-", " ")
            .replace("/", " ")
        )
        text = text.rstrip(" ")

        return text

    def __call__(self, text: Union[List[str], str]) -> torch.Tensor:
        if self.do_pre_process:
            if isinstance(text, str):
                text = self.pre_process(text)
            else:
                text = [self.pre_process(t) for t in text]
        tokens = self.tokenizer(text)["input_ids"]
        if not self.add_end_token and tokens[-1] == self.sep_token_id:
            tokens = tokens[:-1]
        input_ids = self.transform(tokens)

        return input_ids


def parse_devices(devices: Union[str, int]) -> int:
    if devices in (-1, "auto"):
        return torch.cuda.device_count() or 1
    if isinstance(devices, int) and devices > 0:
        return devices
    raise ValueError(f"Devices must be 'auto' or a positive integer, got: {devices!r}")

def get_default_supported_precision(training: bool) -> str:
    """Return default precision that is supported by the hardware: either `bf16` or `16`.

    Args:
        training: `-mixed` or `-true` version of the precision to use

    Returns:
        default precision that is suitable for the task and is supported by the hardware
    """
    from lightning.fabric.accelerators import MPSAccelerator

    if MPSAccelerator.is_available() or (torch.cuda.is_available() and not torch.cuda.is_bf16_supported()):
        return "16-mixed" if training else "16-true"
    return "bf16-mixed" if training else "bf16-true"


@torch.no_grad()
def encode_text(model, text_dataloader, device):
    text_embeds = []
    text_feats = []
    text_atts = []
    for text, text_att in text_dataloader:
        text = text.to(device)
        text_att = text_att.to(device)
        text_embed, text_feat = model(
            text=text, text_atts=text_att, input_type="text", is_train=False
        )
        text_embeds.append(text_embed)
        text_feats.append(text_feat)
        text_atts.append(text_att)
    text_embeds = torch.cat(text_embeds, dim=0)
    text_feats = torch.cat(text_feats, dim=0)
    text_atts = torch.cat(text_atts, dim=0)
    return text_embeds, text_feats, text_atts


@torch.no_grad()
def encode_image(model, image_dataloader, device):
    image_embeds = []
    image_feats = []
    for image in image_dataloader:
        image = image.to(device)
        image_embed, image_feat = model(image=image, input_type="image", is_train=False)
        image_embeds.append(image_embed)
        image_feats.append(image_feat)
    image_embeds = torch.cat(image_embeds, dim=0)
    image_feats = torch.cat(image_feats, dim=0)
    return image_embeds, image_feats


def launch_finetune(config: DictConfig, train_logger: CSVLogger, val_logger: CSVLogger):
    devices = parse_devices(config.workload.devices)
    precision = config.workload.precision or get_default_supported_precision(training=True)
    strategy = "auto"
    fabric = Fabric(
        devices=devices,
        num_nodes=config.workload.num_nodes,
        strategy=strategy,
        precision=precision,
        plugins=None,
        accelerator=config.accelerator,
    )

    fabric.launch(main, devices, config, train_logger, val_logger)


def training_image_transform(
    image_size: int = 384,
    scale: Tuple[float, float] = (0.5, 1.0),
    image_interpolation=transforms.InterpolationMode.BICUBIC,
    mean: Tuple[float, float, float] = MEAN,
    std_dev: Tuple[float, float, float] = STD_DEV,
) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size, scale=scale, interpolation=image_interpolation
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(2, 7),
            transforms.ToTensor(),
            transforms.Normalize(mean, std_dev),
        ]
    )

def get_dataloaders(
    fabric: Fabric, config:DictConfig) -> Tuple[DataLoader, DataLoader]:
    train_dataloader = None
    val_dataloader = None

    if config.dataloader.name == 'pytorch':
        if config.workload.run_training:
            train_dataset = S3RedisRetrievalTrainingDataset(
                annotation_file=config.workload.train_annotation_file,
                s3_data_dir=config.workload.s3_train_prefix,
                image_transform= training_image_transform(),
                text_transform=ALBEFTextTransform(truncate=True, pad_to_max_seq_len=True, max_seq_len=30, add_end_token=False),
                cache_address=config.dataloader.cache_address,
            )
            sampler = RandomSampler(train_dataset)

            train_dataloader = DataLoader(
                train_dataset,
                batch_size=config.workload.batch_size,
                sampler=sampler,
                num_workers=config.workload.num_pytorch_workers,
                pin_memory=True,
                collate_fn=retrieval_train_collate_fn
            )
            train_dataloader = fabric.setup_dataloaders(train_dataloader, move_to_device=True)

    return train_dataloader, val_dataloader

def train_loop(fabric: Fabric, job_id: str, train_logger: CSVLogger, model, optimizer, train_dataloader, train_start_time, current_epoch, global_step_count, max_steps, limit_train_batches, criterion, batch_wts):
    model.train()
    total_samples = 0
    total_train_loss = 0.0
    correct_preds = 0
    alpha = 0.4
    end = time.perf_counter()
    for batch_idx, (image,text,text_atts, idx, data_load_time, transformation_time, is_cache_hit, cached_on_miss) in enumerate(train_dataloader):

        wait_for_data_time = time.perf_counter() - end
        
        if limit_train_batches is not None and batch_idx >= limit_train_batches:
                break
        # Forward pass: Compute model output and loss
        gpu_processing_started = time.perf_counter()
       
        loss = model(image, text, text_atts, idx, alpha, is_train=True)
        # Backpropagation and optimization
        optimizer.zero_grad()  # Clear previous gradients
        fabric.backward(loss)  # Backpropagation
        optimizer.step()  # Update weights
        total_train_loss += loss.item() * image.size(0)
        if fabric.device.type == 'cuda':
                torch.cuda.synchronize()

        # Track time taken for GPU processing
        gpu_processing_time = time.perf_counter() - gpu_processing_started
        # Metrics calculation
        total_samples += image.size(0)
        avg_train_loss = total_train_loss / total_samples
        avg_train_acc = correct_preds / total_samples
        global_step_count +=1

        if isinstance(train_dataloader.dataset, S3RedisRetrievalTrainingDataset):
                data_load_time = sum(data_load_time)
                transformation_time = sum(transformation_time)
                cache_hit_samples = sum(is_cache_hit)
                cache_hit_bacth = 1 if cache_hit_samples == len(is_cache_hit) else 0

        metrics= OrderedDict({
                            "Elapsed Time (s)": time.perf_counter() - train_start_time,
                            "Num Torch Workers": train_dataloader.num_workers,
                            "Device": fabric.global_rank,
                            "Epoch Index": current_epoch,
                            "Batch Index": batch_idx+1,
                            "Batch Size": image.size(0),
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
    return global_step_count

def retrieval_train_collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, int]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    image_list = []
    text_list = []
    idx_list = []
    transformation_time_list = []
    fetch_duration_list = []
    cache_hit_list = []
    cached_after_fetch_list = []
    for image, text, idx, fetch_duration, transformation_time, cache_hit, cached_after_fetch in batch:
        image_list.append(image)
        text_list.append(text)
        idx_list.append(idx)
        transformation_time_list.append(transformation_time)
        fetch_duration_list.append(fetch_duration)
        cache_hit_list.append(cache_hit)
        cached_after_fetch_list.append(cached_after_fetch)
        # print(f'Image shape: {image.shape}, Text shape: {text.shape}, Index: {idx}')
    images = torch.stack(image_list, dim=0)
    text = pad_sequence(text_list, batch_first=True)  # You can specify your padding value
    text_atts = (text != 0).type(torch.long)
    idx = torch.Tensor(idx_list).type(torch.long)
    return (
        images,
        text,
        text_atts,
        idx,
        fetch_duration_list,
        transformation_time_list,
        cache_hit_list,
        cached_after_fetch_list
    )
    

def main(fabric: Fabric, devices: int, config: DictConfig,train_logger: CSVLogger, val_logger: CSVLogger) -> None:
     
    if config.seed is not None:
        seed_everything(config.seed) # instead of torch.manual_seed(...)
    else:
        seed_everything(config.job_id) # instead of torch.manual_seed(...)

    
    model = albef_model_for_retrieval(config.workload, pretrained=True)

    # optimizer = AdamW(optimizer_params, lr=args["lr"])
    # scheduler = CosineAnnealingWarmRestarts(
    #     optimizer, T_0=args["max_epochs"], eta_min=args["min_lr"]
    # )

    optimizer = optim.Adam(model.parameters(), lr=config.workload.lr)
    model, optimizer = fabric.setup(model, optimizer)

    fabric.print(f"Number of trainable parameters: {num_parameters(model, requires_grad=True):,}")
    fabric.print(f"Number of non-trainable parameters: {num_parameters(model, requires_grad=False):,}")
    train_dataloader, val_dataloader = get_dataloaders(fabric, config)
    
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
        # if isinstance(train_dataloader.sampler, ShadeSampler):
        #     train_dataloader.sampler.set_epoch(current_epoch)

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
        
        if config.workload.max_steps is not None and global_train_step_count >= config.workload.max_steps:
            should_stop = True
        if config.workload.max_epochs is not None and current_epoch >= config.workload.max_epochs:
            should_stop = True

    # if isinstance(train_dataloader.sampler, SUPERSampler):
    #     train_dataloader.sampler.send_job_ended_notfication()

    elapsed_time = time.perf_counter() - train_start_time

    fabric.print(f"Training completed in {elapsed_time:.2f} seconds")



@hydra.main(version_base=None, config_path="./conf", config_name="config")
def run(config: DictConfig):

    log_dir = f"{config.log_dir}/{config.workload.name}/{config.dataloader.name}/{config.exp_id}/{config.job_id}".lower()
    log_dir = os.path.normpath(log_dir)  # Normalize path for Windows
    # config.workload.checkpoint_dir = 'mlworkloads\checkpoints\TinyLlama-1.1B-Chat-v1.0'
    train_logger = CSVLogger(root_dir=log_dir, name="train", prefix='', flush_logs_every_n_steps=config.log_interval)
    val_logger = CSVLogger(root_dir=log_dir, name="val", prefix='', flush_logs_every_n_steps=config.log_interval)
    launch_finetune(config, train_logger,val_logger)


if __name__ == "__main__":
    run()
