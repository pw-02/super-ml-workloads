# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import math
import pprint
import time
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Optional, Tuple, Union

import lightning as L
import torch
import torch.nn as nn
from lightning.fabric.strategies import FSDPStrategy
from lightning.fabric.utilities.throughput import ThroughputMonitor, measure_flops
from torch.utils.data import DataLoader
from torchmetrics.aggregation import RunningMean
from typing_extensions import Literal
from mlworklaods.args import LLMTrainArgs, EvalArgs
from mlworklaods.llm.config import name_to_config
from lightning.fabric.loggers import Logger
from typing import Any, Iterable, List, Literal, Optional, Tuple, Union, cast
from mlworklaods.dataloaders.torch_lru.torch_lru_text_dataset import TorchLRUTextDataset
from lightning.fabric.accelerators import Accelerator
from lightning.fabric.strategies import Strategy

# from data.tiny_llama import TinyLlama
from mlworklaods.llm.model import GPT, Block, CausalSelfAttention, Config, LLaMAMLP
from mlworklaods.utils import (
    CycleIterator,
    chunked_cross_entropy,
    num_parameters,
    reset_parameters,
    get_default_supported_precision
)


class LLMPretrainer():
    def __init__(
        self,
        train:LLMTrainArgs,
        accelerator: Union[str, Accelerator] = "auto",
        strategy: Union[str, Strategy] = "auto",
        devices: Union[List[int], str, int] = "auto",
        precision: Literal["bf16-true", "bf16-mixed", "32-true", None] = None,
        loggers: Optional[Union[Logger, List[Logger]]] = None,
        model_name: Optional[str] = None,
        eval: EvalArgs = EvalArgs(interval=1000, max_iters=100),
        seed: int = None
    ) -> None:
    
        available_models = "\n".join(sorted(name_to_config))
        if model_name not in available_models:
            raise ValueError(f"Please specify --model_name <model_name>. Available values:\n{available_models}")
        
        self.devices = devices
        self.model_config = Config.from_name(model_name)
        self.train: LLMTrainArgs = train
        self.eval: EvalArgs = eval
        self.fabric = L.Fabric(
            accelerator=accelerator,
            strategy=strategy,
            devices=devices,
            precision=precision,
            loggers=loggers,
            )
        
        if seed:
            self.fabric.seed_everything(seed)
        
        self.fabric.launch()
        

    def initialize_model_and_optimizer(self):
        t0 = time.perf_counter()
        with self.fabric.init_module(empty_init=True):
            model = GPT(self.model_config)

        initialize_weights(self.fabric, model, n_layer=self.model_config.n_layer, n_embd=self.model_config.n_embd)

        if self.train.tie_embeddings:
            model.transformer.wte.weight = model.lm_head.weight
        if self.train.max_seq_length:
            model.max_seq_length = self.train.max_seq_length

        self.fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
        self.fabric.print(f"Total parameters: {num_parameters(model):,}")

        model = torch.compile(model)
        model = self.fabric.setup(model)
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.train.learning_rate,
            weight_decay=self.train.weight_decay,
            betas=(self.train.beta1, self.train.beta2),
            fused=self.fabric.device.type == "cuda",
            )
        optimizer = self.fabric.setup_optimizers(optimizer)

        return model, optimizer

    

    def fit(self, model,optimizer,train_dataloader, val_dataloader):
        
        train_dataloader = self.fabric.setup_dataloaders(train_dataloader, use_distributed_sampler=False)
        
        if val_dataloader is not None:
            val_dataloader = self.fabric.setup_dataloaders(val_dataloader, use_distributed_sampler=False)
        
        state = {
            "model": model,
            "optimizer": optimizer,
            "train_dataloader": train_dataloader,
            "iter_num": 0,
            "step_count": 0,
        }
        
        train_time = time.perf_counter()
        
        if self.eval.initial_validation and val_dataloader:
            val_loss = validate(self.fabric, model, val_dataloader, max_iters=eval.max_iters)
            val_loss = f"{val_loss:.3f}"
        elif val_dataloader:
            validate(self.fabric, model, val_dataloader, max_iters=2)   # sanity check
            val_loss = "n/a"
        else:
            val_loss = "n/a"
        
        throughput = ThroughputMonitor(self.fabric, window_size=5)

        with torch.device("meta"):
            meta_model = GPT(model.config)
            x = torch.randint(0, 1, (self.train.micro_batch_size, meta_model.max_seq_length))
            model_fwd = lambda: meta_model(x)
            model_loss = lambda y: chunked_cross_entropy(y, x, chunk_size=0)
            measured_flops = measure_flops(meta_model, model_fwd, model_loss)
            self.fabric.print(f"Measured TFLOPs: {measured_flops * self.fabric.world_size / 1e12:.2f}")
            del meta_model, x

        max_tokens_per_device = self.train.max_tokens // self.fabric.world_size
        tokens_per_iter = self.train.micro_batch_size * model.max_seq_length
        max_iters = max_tokens_per_device // tokens_per_iter
        log_iter_interval = self.train.log_interval * self.train.gradient_accumulation_iters(self.devices)
        initial_iter = state["iter_num"]
        train_iterator = CycleIterator(train_dataloader)

        running_loss = RunningMean(window=self.train.gradient_accumulation_iters(self.devices), 
                                   sync_on_compute=False).to(self.fabric.device)
        self.fabric.barrier()
        total_t0 = time.perf_counter()

        warmup_iters = self.train.warmup_iters(self.devices, max_iters, train_dataloader)

        for input_ids, targets  in train_iterator:
            if state["iter_num"] >= max_iters:
                break

            # determine and set the learning rate for this iteration
            lr = get_lr(self.train.learning_rate, state["iter_num"], warmup_iters, max_iters, self.train.min_lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            state["iter_num"] += 1
            iter_t0 = time.perf_counter()

            is_accumulating = state["iter_num"] % self.train.gradient_accumulation_iters(self.devices) != 0
            with self.fabric.no_backward_sync(model, enabled=is_accumulating):
                logits = model(input_ids)
                loss = chunked_cross_entropy(logits, targets)
                self.fabric.backward(loss / self.train.gradient_accumulation_iters(self.devices))

            running_loss.update(loss.detach())

            if not is_accumulating:
                self.fabric.clip_gradients(model, optimizer, max_norm=self.train.max_norm)
                optimizer.step()
                optimizer.zero_grad()
                state["step_count"] += 1

            if state["iter_num"] % log_iter_interval == 0:
                loss = running_loss.compute().item()  # expensive device-to-host synchronization
                t1 = time.perf_counter()
                throughput.update(
                    time=(t1 - total_t0),
                    flops=(measured_flops * log_iter_interval),
                    batches=state["iter_num"],
                    samples=(state["iter_num"] * self.train.micro_batch_size),
                    lengths=(state["iter_num"] * self.train.micro_batch_size * model.max_seq_length),
                )
                metrics = {
                    "loss": loss,
                    "iter": state["iter_num"],
                    "step": state["step_count"],
                    "epoch": train_iterator.epoch,
                    "iter_time": t1 - iter_t0,
                    "remaining_time": (
                        (t1 - total_t0) / (state["iter_num"] - initial_iter) * (max_iters - state["iter_num"])
                    ),
                    "tokens": state["iter_num"] * self.train.micro_batch_size * model.max_seq_length,
                    "total_tokens": (state["iter_num"] * self.train.micro_batch_size * model.max_seq_length * self.fabric.world_size),
                    "learning_rate": lr,
                }
                if isinstance(val_loss, float):
                    val_loss = f"{val_loss:.3f}"

                self.fabric.print(
                    f"Epoch {metrics['epoch']+1} | iter {metrics['iter']} step {metrics['step']} |"
                    f" loss train: {metrics['loss']:.3f},"
                    f" val: {val_loss} |"
                    f" iter time: {metrics['iter_time'] * 1000:.2f} ms"
                    f"{' (step)' if not is_accumulating else ''}"
                    f" remaining time: {timedelta(seconds=int(metrics['remaining_time']))!s}"
                )

                throughput_metrics = throughput.compute()
                metrics.update(throughput_metrics)
                self.fabric.log_dict(metrics, step=state["iter_num"] - 1)

            if val_dataloader is not None and not is_accumulating and state["step_count"] % eval.interval == 0:
                t0 = time.perf_counter()
                val_loss = validate(self.fabric, model, val_dataloader, max_iters=self.eval.max_iters)
                val_loss = val_loss.item()
                td = time.perf_counter() - t0

                self.fabric.print(f"iter {state['iter_num']}: val loss {val_loss:.4f}, val time: {td * 1000:.2f} ms")
                metrics = {"val_loss": val_loss, "val_ppl": math.exp(val_loss)}
                self.fabric.log_dict(metrics, step=state["iter_num"] - 1)
                self.fabric.barrier()

        self.fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
        if  self.fabric.device.type == "cuda":
         self.fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")
        
    
@torch.no_grad()
def validate(fabric: L.Fabric, model: nn.Module, val_dataloader: DataLoader, max_iters: int) -> torch.Tensor:
    fabric.barrier()
    fabric.print("Validating ...")
    model.eval()

    losses = []
    for k, (input_ids, targets) in enumerate(val_dataloader):
        if k >= max_iters:
            break
        # input_ids = batch[:, 0 : model.max_seq_length].contiguous().long()
        # targets = batch[:, 1 : (model.max_seq_length + 1)].contiguous().long()
        logits = model(input_ids)
        loss = chunked_cross_entropy(logits, targets)
        losses.append(loss)

    val_loss = torch.stack(losses).mean()
    model.train()
    fabric.barrier()
    return val_loss


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


def initialize_weights(fabric: L.Fabric, model: GPT, n_layer: int, n_embd: int) -> None:
    """GPT-NeoX weight initialization (https://arxiv.org/abs/2204.06745)."""
    # Adapted from https://github.com/jzhang38/TinyLlama

    def init_weights(module, std):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)

    for mod in model.modules():
        if isinstance(mod, (nn.Embedding, nn.Linear)):
            mod.reset_parameters = partial(init_weights, mod, std=math.sqrt(2.0 / 5 / n_embd))

    # need a separate loop because `mod.proj` below is a `nn.Linear` too
    for mod in model.modules():
        if isinstance(mod, (LLaMAMLP, CausalSelfAttention)):
            mod.proj.reset_parameters = partial(init_weights, mod.proj, std=(1 / math.sqrt(n_embd) / n_layer))

    if not isinstance(fabric.strategy, FSDPStrategy):
        reset_parameters(model)



def validate_args(train: LLMTrainArgs, eval: EvalArgs, initial_checkpoint_dir, resume) -> None:
    issues = []
    unsupported = [(train, ["max_steps"]), (eval, ["max_new_tokens"])]
    for args, names in unsupported:
        for name in names:
            if getattr(args, name) is not None:
                issues.append(f"{__file__} doesn't support the {name!r} argument. This is set in {args}")
    required = [(train, ["max_tokens", "max_norm"])]
    for args, names in required:
        for name in names:
            if getattr(args, name) is None:
                issues.append(f"{__file__} requires the {name!r} argument. This is set in {args}")
    if initial_checkpoint_dir and resume:
        issues.append("Can't provide both `--resume` and `--initial_checkpoint_dir`. Choose one.")
    if issues:
        raise ValueError("\n".join(issues))

