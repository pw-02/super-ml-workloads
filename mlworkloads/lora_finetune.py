import torch
import dataclasses

import sys
print(sys.path)

from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig
from lightning.fabric import Fabric, seed_everything
from lightning.fabric.loggers import CSVLogger
import math
from litgpt.lora import GPT, Block, Config, lora_filter, mark_only_lora_as_trainable
from lightning.fabric.plugins import BitsandbytesPrecision
from lightning_utilities.core.imports import RequirementCache
from lightning.fabric.strategies import FSDPStrategy
from litgpt.tokenizer import Tokenizer
from pathlib import Path
import warnings
from typing import Dict, List, Literal, Optional, Tuple, Union
from litgpt.data import DataModule
from torch.utils.data import DataLoader, ConcatDataset
from lightning.fabric.utilities import ThroughputMonitor
from torchmetrics import RunningMean
from litgpt.generate.base import generate
from litgpt.prompts import save_prompt_style
import os
import time
from collections import OrderedDict
# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
import math
from dataclasses import dataclass
from typing import Optional
import warnings
from dataloading.litdata.custom_streming_dataset import CustomStreamingDataset
warnings.filterwarnings("ignore", category=UserWarning)

from litgpt.utils import (
    auto_download_checkpoint,
    check_nvlink_connectivity,
    CycleIterator,
    check_valid_checkpoint_dir,
    choose_logger,
    chunked_cross_entropy,
    copy_config_files,
    get_default_supported_precision,
    load_checkpoint,
    init_out_dir,
    instantiate_torch_optimizer,
    instantiate_bnb_optimizer,
    num_parameters,
    parse_devices,
    save_hyperparameters,
)

@dataclass
class TrainArgs:
    """Training-related arguments"""
    save_interval: Optional[int] = 1000
    """Number of optimizer steps between saving checkpoints"""
    log_interval: int = 1
    """Number of iterations between logging calls"""
    global_batch_size: int = 64
    """Number of samples between optimizer steps across data-parallel ranks"""
    micro_batch_size: int = 4
    """Number of samples per data-parallel rank"""
    lr_warmup_steps: Optional[int] = 100
    """Number of iterations with learning rate warmup active"""
    lr_warmup_fraction: Optional[float] = None
    """The fraction of an epoch to use for learning rate warmup"""
    epochs: Optional[int] = None
    """Number of epochs to train on"""
    # TODO: `pretrain` is the only script using `max_tokens` explicitly. replace it with epoch_size*epochs?
    max_tokens: Optional[int] = None
    """Total number of tokens to train on"""
    max_steps: Optional[int] = None
    """Limits the number of optimizer steps to run"""
    max_seq_length: Optional[int] = None
    """Limits the length of samples"""
    tie_embeddings: Optional[bool] = None
    """Whether to tie the embedding weights with the language modeling head weights"""
    
    run_training: Optional[bool] = True

    # Optimization args
    max_norm: Optional[float] = None
    min_lr: float = 6e-5

    def __post_init__(self) -> None:
        if self.lr_warmup_fraction and self.lr_warmup_steps:
            raise ValueError(
                "Can't provide both `--train.lr_warmup_fraction` and `--train.lr_warmup_steps`. Choose one."
            )
        if self.lr_warmup_fraction and not (0 <= self.lr_warmup_fraction <= 1):
            raise ValueError("`--train.lr_warmup_fraction` must be between 0 and 1.")

        if self.lr_warmup_steps and self.max_steps and (self.lr_warmup_steps >= self.max_steps):
            warnings.warn(
                "`--train.lr_warmup_steps` should be less than `--train.max_steps`."
                f" Got {self.lr_warmup_steps} lr_warmup_steps and {self.max_steps} max_steps.", UserWarning)

    def gradient_accumulation_iters(self, devices: int) -> int:
        """Number of iterations between gradient synchronizations"""
        gradient_accumulation_iters = self.batch_size(devices) // self.micro_batch_size
        assert gradient_accumulation_iters > 0
        return gradient_accumulation_iters

    def batch_size(self, devices: int) -> int:
        """Number of samples between optimizer steps per data-parallel rank"""
        batch_size = self.global_batch_size // devices
        assert batch_size > 0
        return batch_size

    def warmup_iters(self, devices: int, max_iters: int, train_dataloader) -> int:
        """Number of iterations to warm up the learning rate."""
        if self.lr_warmup_fraction:
            return min(max_iters, math.ceil(self.lr_warmup_fraction * len(train_dataloader)))
        if self.lr_warmup_steps:
            return min(max_iters, self.lr_warmup_steps * self.gradient_accumulation_iters(devices))
        return 0


@dataclass
class EvalArgs:
    """Evaluation-related arguments"""

    interval: int = 600
    """Number of optimizer steps between evaluation calls"""
    max_new_tokens: Optional[int] = None
    """Number of tokens to generate"""
    max_iters: int = 100
    """Number of iterations"""
    initial_validation: bool = False
    """Whether to evaluate on the validation set at the beginning of the training"""
    final_validation: bool = True
    """Whether to evaluate on the validation set at the end of the training"""
    run_validation: Optional[bool] = True




def launch_finetune(config: DictConfig, train_logger: CSVLogger, val_logger: CSVLogger):
    # config.workload.checkpoint_dir = Path(config.workload.checkpoint_dir)
    checkpoint_dir = auto_download_checkpoint(model_name=Path(config.workload.checkpoint_dir), access_token=None)
    devices = parse_devices(config.workload.devices)
    # out_dir = init_out_dir(config.workload.out_dir)
    check_valid_checkpoint_dir(checkpoint_dir)
    lora_config = Config.from_file(
        os.path.join(checkpoint_dir, "model_config.yaml"),
        lora_r=config.workload.lora_r,
        lora_alpha=config.workload.lora_alpha,
        lora_dropout=config.workload.lora_dropout,
        lora_query=config.workload.lora_query,
        lora_key=config.workload.lora_key,
        lora_value=config.workload.lora_value,
        lora_projection=config.workload.lora_projection,
        lora_mlp=config.workload.lora_mlp,
    )


    precision = config.workload.precision or get_default_supported_precision(training=True)

    train = TrainArgs(
        save_interval=config.workload.save_interval,
        log_interval=config.workload.log_interval,
        global_batch_size=config.workload.global_batch_size,
        micro_batch_size=config.workload.micro_batch_size,
        lr_warmup_steps=config.workload.lr_warmup_steps,
        epochs=config.workload.epochs,
        max_tokens=config.workload.max_tokens,
        max_steps=config.workload.max_steps,
        max_seq_length=config.workload.max_seq_length,
        tie_embeddings=config.workload.tie_embeddings,
        max_norm=config.workload.max_norm,
        min_lr=config.workload.min_lr,
        run_training=config.workload.run_training
    )
    eval = EvalArgs(
        interval=config.workload.eval_interval,
        max_new_tokens=config.workload.eval_max_new_tokens,
        max_iters=config.workload.eval_max_iters,
        initial_validation=config.workload.eval_final_validation,
        final_validation=config.workload.eval_final_validation,
        run_validation=config.workload.run_validation
    )

    plugins = None
    if config.workload.quantize is not None and config.workload.quantize.startswith("bnb."):
        if "mixed" in precision:
            raise ValueError("Quantization and mixed precision is not supported.")
        if RequirementCache("bitsandbytes != 0.42.0"):
            warnings.warn(
                "LitGPT only supports bitsandbytes v0.42.0. "
                "This may result in errors when using quantization."
            )
        dtype = {"16-true": torch.float16, "bf16-true": torch.bfloat16, "32-true": torch.float32}[precision]
        plugins = BitsandbytesPrecision(config.workload.quantize[4:], dtype)
        precision = None

    if devices * config.workload.num_nodes > 1:
        if config.workload.quantize:
            raise NotImplementedError(
                "Quantization is currently not supported for multi-GPU training. Please set devices=1 and num_nodes=1"
                " when using the --quantize flag."
            )
        strategy = FSDPStrategy(
            auto_wrap_policy={Block},
            activation_checkpointing_policy={Block},
            state_dict_type="full",
            limit_all_gathers=True,
            cpu_offload=False,
        )
    else:
        strategy = "auto"

    fabric = Fabric(
        devices=devices,
        num_nodes=config.workload.num_nodes,
        strategy=strategy,
        precision=precision,
        plugins=plugins,
    )

    fabric.launch(main, devices, config, lora_config, checkpoint_dir, train, eval, train_logger, val_logger)


def main(fabric: Fabric,
    devices: int,
    config: DictConfig,
    model_config: Config,
    checkpoint_dir: Path,
    train: TrainArgs,
    eval: EvalArgs,
    train_logger: CSVLogger, 
    val_logger: CSVLogger
) -> None:
    
    validate_args(train, eval)
    tokenizer = Tokenizer(checkpoint_dir)
    train_dataloader, val_dataloader = get_dataloaders(fabric, config, tokenizer, train, eval)
    steps_per_epoch = len(train_dataloader) // train.gradient_accumulation_iters(devices)
    lr_max_steps = train.max_steps or float("inf")

    if config.seed is not None:
        seed_everything(config.seed) # instead of torch.manual_seed(...)

    checkpoint_path = os.path.join(checkpoint_dir, "lit_model.pth")
    with fabric.init_module(empty_init=(fabric.world_size > 1)):
        model = GPT(model_config)
    mark_only_lora_as_trainable(model)
    fabric.print(f"Number of trainable parameters: {num_parameters(model, requires_grad=True):,}")
    fabric.print(f"Number of non-trainable parameters: {num_parameters(model, requires_grad=False):,}")
    model = fabric.setup_module(model)
    # optimizer =config.workload.optimizer
    # if isinstance(fabric.strategy.precision, BitsandbytesPrecision):
    #     optimizer = instantiate_bnb_optimizer(optimizer, model.parameters())
    # else:
    #     optimizer = instantiate_torch_optimizer(optimizer, model.parameters())
    
    optimizer = torch.optim.SGD(model.parameters(), lr=config.workload.learning_rate)

    optimizer = fabric.setup_optimizers(optimizer)
    scheduler = get_lr_scheduler(optimizer, warmup_steps=train.lr_warmup_steps, max_steps=lr_max_steps)
    # strict=False because missing keys due to LoRA weights not contained in state dict
    load_checkpoint(fabric, model, checkpoint_path, strict=False)
    train_time = time.perf_counter()
    fit(
        fabric=fabric,
        job_id=config.job_id,
        train_start_time=train_time,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        devices=devices,
        checkpoint_dir=checkpoint_dir,
        train=train,
        eval=eval,
        train_logger=train_logger,
        val_logger=val_logger,
    )
    fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")

    # Final evaluation
    if eval.final_validation:
        val_loss = validate(fabric, model, val_dataloader, dataclasses.replace(eval, max_iters=len(val_dataloader)))
        metrics = {"val_loss": val_loss, "val_ppl": math.exp(val_loss)}
        fabric.log_dict(metrics)
        fabric.print(f"Final evaluation | val loss: {val_loss.item():.3f} | val ppl: {math.exp(val_loss):.3f}")


def fit(
    fabric: Fabric,
    job_id: int,
    train_start_time: float,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    devices: int,
    checkpoint_dir: Path,
    train: TrainArgs,
    eval: EvalArgs,
    train_logger: CSVLogger,
    val_logger: CSVLogger
) -> None:
    
    # longest_seq_length, longest_seq_ix = get_longest_seq_length(ConcatDataset([train_dataloader.dataset, val_dataloader.dataset]))
    # model.max_seq_length = min(longest_seq_length, train.max_seq_length or float("inf"))
    # fabric.print(
    #     f"The longest sequence length in the train data is {longest_seq_length}, the model's maximum sequence length is"
    #     f" {model.max_seq_length} and context length is {model.config.block_size}"
    # )

    if eval.initial_validation and eval.run_validation:
        val_loss = validate(fabric, model, val_dataloader, dataclasses.replace(eval, max_iters=len(val_dataloader)))
        val_loss = f"{val_loss:.3f}"
    else:
        val_loss = "n/a"
    # else:
    #     fabric.print("Verifying settings ...")
    #     validate(fabric, model, val_dataloader, dataclasses.replace(eval, max_iters=2), verbose=False)  # sanity check
    #     val_loss = "n/a"

    train_iterator = CycleIterator(train_dataloader)
    throughput = ThroughputMonitor(fabric, window_size=50)
    running_loss = RunningMean(window=train.gradient_accumulation_iters(devices), sync_on_compute=False).to(
        fabric.device
    )
    max_steps = train.max_steps or float("inf")
    step_count = 0
    iter_num = 0
    total_lengths = 0
    total_t0 = time.perf_counter()

    while step_count < max_steps and train_iterator.epoch < train.epochs:
        iter_num += 1
        iter_t0 = time.perf_counter()
        batch, data_loading_time, transformation_time,is_cache_hit,cached_on_miss = next(train_iterator)
        input_ids, targets = batch
        wait_for_data_time = time.perf_counter() - iter_t0
        if isinstance(train_dataloader.dataset, CustomStreamingDataset):
                data_load_time = float(data_loading_time.sum())
                transformation_time = float(transformation_time.sum())
                cache_hit_samples = int(is_cache_hit.sum())
                cache_hit_bacth = 1 if cache_hit_samples == len(is_cache_hit) else 0
      
        is_accumulating = iter_num % train.gradient_accumulation_iters(devices) != 0

        gpu_processing_started = time.perf_counter()
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(input_ids, lm_head_chunk_size=128)
            # shift the targets such that output n predicts token n+1
            loss = chunked_cross_entropy(logits, targets)
            fabric.backward(loss / train.gradient_accumulation_iters(devices))

        running_loss.update(loss.detach())

        if not is_accumulating:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            step_count += 1
        
        loss = running_loss.compute().item()  # expensive device-to-host synchronization
        # Track time taken for GPU processing
        gpu_processing_time = time.perf_counter() - gpu_processing_started

        total_lengths += input_ids.numel()
        if iter_num % train.log_interval == 0:
            t1 = time.perf_counter()
            throughput.update(
                time=t1 - total_t0, batches=iter_num, samples=iter_num * train.micro_batch_size, lengths=total_lengths
            )
            
            # throughput.compute_and_log(step=iter_num)
            metrics= OrderedDict({
                            "Elapsed Time (s)": time.perf_counter() - train_start_time,
                            "Num Torch Workers": train_dataloader.num_workers,
                            "Device": fabric.global_rank,
                            "Epoch Index": train_iterator.epoch + 1,
                            "Batch Index": iter_num,
                            "Batch Size (Tokens)": train.micro_batch_size * model.config.block_size,
                            "Iteration Time (s)": t1 - iter_t0,
                            "Wait for Data Time (s)": wait_for_data_time,
                            "GPU Processing Time (s)": gpu_processing_time,
                            "Data Load Time (s)": data_load_time,
                            "Transformation Time (s)": transformation_time,
                            "Cache_Hit (Batch)": cache_hit_bacth,
                            "Cache_Hits (Samples)": cache_hit_samples,
                            "Train Loss": loss, #calculates the average training loss across all batches.
                            })
            train_logger.log_metrics(metrics,step=iter_num)
            fabric.print(
                    f" Job {job_id} | Epoch: {metrics['Epoch Index']} ({iter_num}) |"
                    f" iter:{metrics['Iteration Time (s)']:.2f}s |"
                    f" data_delay:{metrics['Wait for Data Time (s)']:.2f}s |"
                    f" gpu:{metrics['GPU Processing Time (s)']:.2f}s |"
                    f" data_fetch:{metrics['Data Load Time (s)']:.2f}s |"
                    f" transform:{metrics['Transformation Time (s)']:.2f}s |"
                    f" elapsed:{metrics['Elapsed Time (s)']:.2f}s |"
                    f" loss: {metrics['Train Loss']:.3f} |"
                    F" cache hit: {metrics['Cache_Hit (Batch)']} |"
                    )
         
            if isinstance(val_loss, torch.Tensor):
                val_loss = f"{val_loss:.3f}"
            
        if not is_accumulating and step_count % eval.interval == 0 and eval.run_validation:
            t0 = time.perf_counter()
            val_loss = validate(fabric, model, val_dataloader, eval)
            # generate_example(fabric, model, tokenizer, eval, data)
            t1 = time.perf_counter() - t0
            fabric.print(f"iter {iter_num}: val loss {val_loss.item():.4f}, val time: {t1 * 1000:.2f} ms")
            # metrics = {"val_loss": val_loss, "val_ppl": math.exp(val_loss)}
            # fabric.log_dict(metrics, step=iter_num)
            fabric.barrier()

        # if train.save_interval is not None and not is_accumulating and step_count % train.save_interval == 0:
        #     checkpoint_file = out_dir / f"step-{step_count:06d}" / "lit_model.pth.lora"
        #     checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        #     save_lora_checkpoint(fabric, model, checkpoint_file)
        #     if fabric.global_rank == 0:
        #         copy_config_files(checkpoint_dir, checkpoint_file.parent)
        #         save_hyperparameters(setup, checkpoint_file.parent)
        #         save_prompt_style(data.prompt_style, checkpoint_file.parent)

def save_lora_checkpoint(fabric: Fabric, model: torch.nn.Module, file_path: Path) -> None:
    fabric.print(f"Saving LoRA weights to {str(file_path)!r}")
    fabric.save(file_path, {"model": model}, filter={"model": lora_filter})

@torch.no_grad()
def generate_example(fabric: Fabric, model: GPT, tokenizer: Tokenizer, eval: EvalArgs, data: DataModule):
    instruction = "Recommend a movie for me to watch during the weekend and explain the reason."
    fabric.print(instruction)
    prompt = data.prompt_style.apply(instruction)
    encoded = tokenizer.encode(prompt, device=fabric.device)
    model.eval()

    max_returned_tokens = len(encoded) + eval.max_new_tokens

    if max_returned_tokens < model.max_seq_length:
        with fabric.init_tensor():
            # do not set `max_seq_length=max_returned_token` because memory is not a concern here
            model.set_kv_cache(batch_size=1)
        output = generate(
            model, encoded, max_returned_tokens=max_returned_tokens, temperature=0.8, eos_id=tokenizer.eos_id
        )
        model.clear_kv_cache()
        model.train()
        output = tokenizer.decode(output)
        fabric.print(output)
    else:
        print(
            f"Length of encoded instruction ({len(encoded)}) and eval.max_new_tokens ({eval.max_new_tokens}) "
            f"exceeds model.max_seq_length ({model.max_seq_length}) used for training. Skipping example generation for efficiency. "
            f"The model's supported context size (post-training) is {model.config.block_size}."
        )


# FSDP has issues with `inference_mode`
@torch.no_grad()
def validate(fabric: Fabric, model: GPT, val_dataloader: DataLoader, eval: EvalArgs, verbose: bool = True) -> torch.Tensor:
    if verbose:
        fabric.print("Validating ...")
    model.eval()
    losses = torch.zeros(min(len(val_dataloader), eval.max_iters))
    for k, batch in enumerate(val_dataloader):
        if k >= eval.max_iters:
            break
        input_ids, targets = batch["input_ids"], batch["labels"]
        logits = model(input_ids)
        losses[k] = chunked_cross_entropy(logits[..., :-1, :], targets[..., 1:], chunk_size=0)

    val_loss = losses.mean()

    model.train()
    return val_loss


def get_longest_seq_length(data: List[Dict]) -> Tuple[int, int]:
    # find out the minimum max_seq_length required during fine-tuning (saves memory!)
    lengths = [len(d["input_ids"]) for d in data]
    longest_seq_length = max(lengths)
    longest_seq_ix = lengths.index(longest_seq_length)
    return longest_seq_length, longest_seq_ix


def get_lr_scheduler(optimizer, warmup_steps: int, max_steps: int):
    # linear warmup followed by cosine annealing
    scheduler1 = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: step / warmup_steps)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(max_steps - warmup_steps))
    return torch.optim.lr_scheduler.SequentialLR(optimizer, [scheduler1, scheduler2], milestones=[warmup_steps])


def get_dataloaders(
    fabric: Fabric, config:DictConfig, tokenizer: Tokenizer, train: TrainArgs, val: EvalArgs) -> Tuple[DataLoader, DataLoader]:
    train_dataloader = None
    val_dataloader = None

    if config.dataloader.name == 'litgpt' or 'pytorch':
        from litdata.streaming import StreamingDataLoader, TokensLoader
        from dataloading.litdata.custom_streming_dataset import CustomStreamingDataset
        if train.run_training:
            train_dataset = CustomStreamingDataset(
            input_dir=config.workload.s3_train_prefix,
            item_loader=None,
            shuffle=True,
            max_cache_size="0GB",
            tokenizer=tokenizer,
            batch_size=train.micro_batch_size,
            seq_length=train.max_seq_length
            )

            train_dataloader = StreamingDataLoader(
                train_dataset, batch_size=train.batch_size(devices=config.workload.devices), 
                pin_memory=True, 
                num_workers=config.workload.num_pytorch_workers, 
                drop_last=True
            )
            train_dataloader = fabric.setup_dataloaders(train_dataloader, move_to_device=True)

        if val.run_validation:
            val_dataset = CustomStreamingDataset(
            input_dir=config.workload.s3_val_prefix,
            item_loader=TokensLoader(block_size=train.max_seq_length+1),
            shuffle=True,
            max_cache_size="0GB",
            tokenizer=tokenizer,
            batch_size=train.micro_batch_size,
            seq_length=train.max_seq_length
            )
            val_dataloader = StreamingDataLoader(
                val_dataset, batch_size=train.batch_size(devices=config.workload.devices), 
                pin_memory=True, 
                num_workers=config.workload.num_pytorch_workers, 
                drop_last=True
            )
            val_dataloader = fabric.setup_dataloaders(val_dataloader, move_to_device=True)


    # if dataloader_name == 'super':
    #     if run_training:
    #         train_dataset = SUPERMappedDataset(s3_data_dir=config.workload.s3_train_prefix, 
    #                                             transform=train_transform,
    #                                             cache_address=config.dataloader.cache_address)

    #         train_sampler = SUPERSampler(
    #                 dataset=train_dataset,
    #                 grpc_server_address=config.dataloader.grpc_server_address,
    #                 batch_size=config.workload.batch_size
    #                 )
    #         train_dataloader = DataLoader(train_dataset, batch_size=None, sampler=train_sampler, num_workers=config.workload.num_pytorch_workers)
    #         train_dataloader = fabric.setup_dataloaders(train_dataloader, move_to_device=True)
        
    #     if run_validation:
    #         val_dataset = SUPERMappedDataset(s3_data_dir=config.workload.s3_val_prefix, 
    #                                          transform=val_transform,
    #                                          cache_address=config.dataloader.cache_address)
    #         val_sampler = SUPERSampler(
    #             dataset=val_dataset,
    #             grpc_server_address=config.dataloader.grpc_server_address,
    #             batch_size=config.workload.batch_size
    #             )
    #         val_dataloader =  DataLoader(val_dataset, batch_size=None, sampler=val_sampler, num_workers=config.workload.num_pytorch_workers)
    #         val_dataloader = fabric.setup_dataloaders(val_dataloader,move_to_device=True)
    
    # elif dataloader_name == 'litgpt':
    #     litdata_data.connect(tokenizer=tokenizer, batch_size=train.micro_batch_size, max_seq_length=train.max_seq_length)
    #     with fabric.rank_zero_first():
    #         litdata_data.prepare_data()
    #         litdata_data.connect(tokenizer=tokenizer, batch_size=train.micro_batch_size, max_seq_length=train.max_seq_length)
    #         with fabric.rank_zero_first():
    #             litdata_data.prepare_data()
    #         litdata_data.setup()

    #         if run_training:
    #             train_dataloader = litdata_data.train_dataloader()
    #             train_dataloader = fabric.setup_dataloaders(train_dataloader)
    #         if run_validation:
    #             val_dataloader = litdata_data.val_dataloader()
    #             val_dataloader = fabric.setup_dataloaders(val_dataloader)
    
    return train_dataloader, val_dataloader



def validate_args(train: TrainArgs, eval: EvalArgs) -> None:
    issues = []
    unsupported = [(train, ["max_tokens", "max_norm", "tie_embeddings", "lr_warmup_fraction"])]
    for args, names in unsupported:
        for name in names:
            if getattr(args, name) is not None:
                issues.append(f"{__file__} doesn't support the {name!r} argument. This is set in {args}")
    required = [(train, ["epochs"]), (eval, ["max_new_tokens"])]
    for args, names in required:
        for name in names:
            if getattr(args, name) is None:
                issues.append(f"{__file__} requires the {name!r} argument. This is set in {args}")
    if not train.epochs and not train.max_steps:
        issues.append(f"{__file__} requires either epochs or max_steps to be set. This is set in {train}")
    if issues:
        raise ValueError("\n".join(issues))
    

@hydra.main(version_base=None, config_path="./conf", config_name="config")
def run(config: DictConfig):

    log_dir = f"{config.log_dir}/{config.workload.name}/{config.dataloader.name}/{config.exp_id}/{config.job_id}".lower()
    log_dir = os.path.normpath(log_dir)  # Normalize path for Windows
    
    train_logger = CSVLogger(root_dir=log_dir, name="train", prefix='', flush_logs_every_n_steps=config.log_interval)
    val_logger = CSVLogger(root_dir=log_dir, name="val", prefix='', flush_logs_every_n_steps=config.log_interval)
    launch_finetune(config, train_logger,val_logger)


if __name__ == "__main__":
    run()
