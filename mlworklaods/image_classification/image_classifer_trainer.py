import os
from collections.abc import Mapping
from functools import partial
from typing import Any, Iterable, List, Literal, Optional, Tuple, Union, cast
from lightning.fabric.strategies import FSDPStrategy
import lightning as L
import torch
from lightning.fabric.accelerators import Accelerator
from lightning.fabric.loggers import Logger
from lightning.fabric.strategies import Strategy
from lightning.fabric.wrappers import _unwrap_objects
from lightning.pytorch.utilities.model_helpers import is_overridden
from lightning_utilities import apply_to_collection
from tqdm import tqdm
import time
from mlworklaods.utils import ResourceMonitor
from collections import OrderedDict
import json
from utils import get_default_supported_precision
from lightning import LightningModule
from mlworklaods.utils import AverageMeter
from torch.utils.data import DataLoader
from  heapdict import heapdict
from mlworklaods.args import *

class ImageClassificationTrainer():
    def __init__(
        self,
        job_id,
        accelerator: Union[str, Accelerator] = "auto",
        strategy: Union[str, Strategy] = "auto",
        devices: Union[List[int], str, int] = "auto",
        precision: Literal["bf16-true", "bf16-mixed", "32-true", None] = None,
        callbacks: Optional[Union[List[Any], Any]] = None,
        loggers: Optional[Union[Logger, List[Logger]]] = None,
        max_epochs: Optional[int] = 1000,
        max_steps: Optional[int] = None,
        grad_accum_steps: int = 1,
        limit_train_batches: Union[int, float] = float("inf"),
        limit_val_batches: Union[int, float] = float("inf"),
        validation_frequency: int = 1,
        use_distributed_sampler: bool = True,
        checkpoint_dir: str = "./checkpoints",
        checkpoint_frequency: int = 1000,
        dataloader_args = None
    ) -> None:
        self.job_id = job_id
        self.global_step = 0
        self.grad_accum_steps: int = grad_accum_steps
        self.current_epoch = 0
        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.should_stop = False
        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        self.validation_frequency = validation_frequency
        self.use_distributed_sampler = use_distributed_sampler
        self._current_train_return: Union[torch.Tensor, Mapping[str, Any]] = {}
        self._current_val_return: Optional[Union[torch.Tensor, Mapping[str, Any]]] = {}
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_frequency = checkpoint_frequency
        self.train_start_time = time.perf_counter()
        self.dataloader_args = dataloader_args
  

        self.fabric = L.Fabric(
            accelerator=accelerator,
            strategy=strategy,
            devices=devices,
            precision=precision,
            callbacks=callbacks,
            loggers=loggers,
        )
    
    def is_using_shade(self):
        if isinstance(self.dataloader_args, SHADEArgs):
            return True
        else:
            False
    
    def fit(self,
            model: LightningModule,
            train_loader: DataLoader,
            val_loader: DataLoader,
            ckpt_path: Optional[str] = None,
            seed:int = None,
            ):
        
        if self.is_using_shade():
            self.batch_wts = []
            for j in range(train_loader.sampler.batch_size):
                self.batch_wts.append(math.log(j+10))

        if seed:
            self.fabric.seed_everything(seed)
        
        self.fabric.launch()

        # setup dataloaders
        train_loader = self.fabric.setup_dataloaders(train_loader, use_distributed_sampler=self.use_distributed_sampler)
        if val_loader is not None:
            val_loader = self.fabric.setup_dataloaders(val_loader, use_distributed_sampler=self.use_distributed_sampler)

        optimizer = model.configure_optimizers()
        
        model, optimizer = self.fabric.setup(model, optimizer)

        # assemble state (current epoch and global step will be added in save)
        state = {"model": model, "optim": optimizer}
        
        # load last checkpoint if available
        if ckpt_path is not None and os.path.isdir(ckpt_path):
            latest_checkpoint_path = self.get_latest_checkpoint(self.checkpoint_dir)
            if latest_checkpoint_path is not None:
                self.load(state, latest_checkpoint_path)
                # check if we even need to train here
                if self.max_epochs is not None and self.current_epoch >= self.max_epochs:
                    self.should_stop = True
        
        self.train_start_time = time.perf_counter()
        
        while not self.should_stop:
            self.train_loop(model, optimizer, train_loader, limit_batches=self.limit_train_batches)


            if self.should_validate:
                self.val_loop(model, val_loader, limit_batches=self.limit_val_batches)

            # self.step_scheduler(model, scheduler_cfg, level="epoch", current_value=self.current_epoch)

            self.current_epoch += 1

            # stopping condition on epoch level
            if self.max_epochs is not None and self.current_epoch >= self.max_epochs:
                self.should_stop = True
            
            if self.current_epoch % self.checkpoint_frequency == 0:
                self.save(state)
        # reset for next fit call
        self.should_stop = False
        
        self.fabric.logger.finalize(status='success')

        return model.losses.avg, model.top1.avg

    def train_loop(
        self,
        model: LightningModule,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        limit_batches: Union[int, float] = float("inf"),
        # using_shade:bool = False,
        # key_id_map = None,
        # batch_wts:List = []
    ):
        # self.fabric.call("on_train_epoch_start")
        # iterable = self.progbar_wrapper(
        #     train_loader, total=min(len(train_loader), limit_batches), desc=f"Epoch {self.current_epoch}"
        # )
        total_loss_for_epoch = 0.
        with ResourceMonitor() as monitor:
            end = time.perf_counter()
            for batch_idx, (batch, cache_hits, fetch_time, transform_time) in enumerate(train_loader):
                data_time = time.perf_counter() - end

                # end epoch if stopping training completely or max batches for this epoch reached
                if self.should_stop or batch_idx >= limit_batches:
                    break

                #self.fabric.call("on_train_batch_start")
                
                compute_start = time.perf_counter()
                # check if optimizer should step in gradient accumulation
                should_optim_step = self.global_step % self.grad_accum_steps == 0
                torch.cuda.synchronize()
                if should_optim_step:
                    # optimizer step runs train step internally through closure
                    loss, item_loss = self.training_step(model=model, batch=batch, batch_idx=batch_idx)
                    
                    # def get_loss():
                    #     return loss
                    
                    optimizer.step()
                    # self.fabric.call("on_before_zero_grad", optimizer
                    optimizer.zero_grad()
                else:
                    # gradient accumulation -> no optimizer step
                    loss, item_loss = self.training_step(model=model, batch=batch, batch_idx=batch_idx)

                torch.cuda.synchronize()

                compute_time = time.perf_counter() - compute_start

                if self.is_using_shade() and item_loss is not None:
                    train_loader.sampler.pass_batch_important_scores(item_loss)
                # only increase global step if optimizer stepped
                self.global_step += int(should_optim_step)

                if self.is_using_shade():
                    sorted_img_indices = train_loader.sampler.get_sorted_index_list()
                    track_batch_indx = 0

                    PQ:heapdict = train_loader.dataset.get_PQ()
                    ghost_cache = train_loader.dataset.get_ghost_cache()
                    key_id_map = train_loader.dataset.get_key_id_map()

                    for indx in sorted_img_indices:
                        if key_id_map.exists(indx.item()):
                            if indx.item() in PQ:
                                #print("Train_index: %d Importance_Score: %f Frequency: %d Time: %s N%dG%d" %(indx.item(),batch_loss,PQ[indx.item()][1]+1,insertion_time,args.nr+1,gpu+1))
                                PQ[indx.item()] = (self.batch_wts[track_batch_indx],PQ[indx.item()][1]+1)
                                ghost_cache[indx.item()] = (self.batch_wts[track_batch_indx],ghost_cache[indx.item()][1]+1)
                                track_batch_indx+=1
                            else:
                                #print("Train_index: %d Importance_Score: %f Time: %s N%dG%d" %(indx.item(),batch_loss,insertion_time,args.nr+1,gpu+1))
                                PQ[indx.item()] = (self.batch_wts[track_batch_indx],1)
                                ghost_cache[indx.item()] = (self.batch_wts[track_batch_indx],1)
                                track_batch_indx+=1
                        else:
                            if indx.item() in ghost_cache:
                                ghost_cache[indx.item()] = (self.batch_wts[track_batch_indx],ghost_cache[indx.item()][1]+1)
                                track_batch_indx+=1
                            else:
                                ghost_cache[indx.item()] = (self.batch_wts[track_batch_indx],1)
                                track_batch_indx+=1
                    train_loader.dataset.set_PQ(PQ)
                    train_loader.dataset.set_ghost_cache(ghost_cache)
                    train_loader.dataset.set_num_local_samples(key_id_map.dbsize())
                                

                metrics= OrderedDict({
                        "elapsed_time": time.perf_counter() - self.train_start_time,
                        "device": self.fabric.global_rank,
                        "epoch": self.current_epoch+1,
                        "batch_idx": batch_idx+1,
                        "batch_size": batch[0].size(0),
                        "batch_time": time.perf_counter() - end,
                        "data_time": data_time,
                        "fetch_time": fetch_time,
                        "transform_time": transform_time,
                        "compute_time": compute_time,
                        "cache_hits": cache_hits,
                        "loss_train": self._current_train_return['loss'],
                        "accuracy_train": self._current_train_return['top1'],
                        "cpu_usge": json.dumps(monitor.resource_data["cpu_util"].summarize()),
                        # "gpu_usge": json.dumps( monitor.resource_data["gpu_util"].summarize())   
                        })
                total_loss_for_epoch +=self._current_train_return['loss']

                self.fabric.log_dict(metrics,step=self.global_step)

                # # this guard ensures, we only step the scheduler once per global step
                # if should_optim_step:
                #     self.step_scheduler(model, scheduler_cfg, level="step", current_value=self.global_step)

                # # add output values to progress bar
                self._current_train_return["comp"] =compute_time
                self._current_train_return["data"] =data_time
                self._current_train_return["tform"] =transform_time

                # self._format_iterable(iterable, self._current_train_return, "t")
                
                self.fabric.print(
                f"{self.job_id} | Epoch {metrics['epoch']} | iter {metrics['batch_idx']}/{min(len(train_loader),self.limit_train_batches)} |"
                f" loss train: {metrics['loss_train']:.3f} |"
                # f" val: {val_loss} |"
                f" iter time: {metrics['batch_time']:.2f} |"
                f" data time: {metrics['data_time']:.2f} |"
                f" compute time: {metrics['compute_time']:.2f} |"
                f" fetch time: {metrics['fetch_time']:.2f} |"
                f" transform time: {metrics['transform_time']:.2f} |"
                f" elapsed time: {metrics['elapsed_time']:.2f} |"

                # f"{' (step)' if not is_accumulating else ''}"
                # f" remaining time: {timedelta(seconds=int(metrics['remaining_time']))!s}"
            )


                # stopping criterion on step level
                if self.max_steps is not None and self.global_step >= self.max_steps:
                    self.should_stop = True
                    break
                # if time.perf_counter() - self.train_start_time > 1000:
                #     print("Training time exceeded 10000 seconds. Exiting.")
                #     self.should_stop = True
                #     break
                end = time.perf_counter()

            # self.fabric.call("on_train_epoch_end")
            if self.is_using_shade():
                train_loader.sampler.on_epoch_end(total_loss_for_epoch/batch_idx+1)

    def val_loop(
        self,
        model: L.LightningModule,
        val_loader: Optional[torch.utils.data.DataLoader],
        limit_batches: Union[int, float] = float("inf"),
    ):
        
        # no validation if val_loader wasn't passed
        if val_loader is None:
            return

        # no validation but warning if val_loader was passed, but validation_step not implemented
        if val_loader is not None and not is_overridden("validation_step", _unwrap_objects(model)):
            L.fabric.utilities.rank_zero_warn(
                "Your LightningModule does not have a validation_step implemented, "
                "but you passed a validation dataloder. Skipping Validation."
            )
            return

        self.fabric.call("on_validation_model_eval")  # calls `model.eval()`

        torch.set_grad_enabled(False)

        self.fabric.call("on_validation_epoch_start")

        # iterable = self.progbar_wrapper(val_loader, total=min(len(val_loader), limit_batches), desc="Validation")

        for batch_idx, batch in enumerate(val_loader):
            # end epoch if stopping training completely or max batches for this epoch reached
            if self.should_stop or batch_idx >= limit_batches:
                break

            self.fabric.call("on_validation_batch_start", batch, batch_idx)

            out = model.validation_step(batch, batch_idx)
            # avoid gradients in stored/accumulated values -> prevents potential OOM
            out = apply_to_collection(out, torch.Tensor, lambda x: x.detach())

            self.fabric.call("on_validation_batch_end", out, batch, batch_idx)
            self._current_val_return = out

            self._format_iterable(iterable, self._current_val_return, "val")

        self.fabric.call("on_validation_epoch_end")

        self.fabric.call("on_validation_model_train")
        torch.set_grad_enabled(True)

    def training_step(self, model: L.LightningModule, batch: Any, batch_idx: int) -> torch.Tensor:
       
        outputs: Union[torch.Tensor, Mapping[str, Any]] = model.training_step(batch, batch_idx=batch_idx, is_shade=self.is_using_shade())
        
        loss = outputs if isinstance(outputs, torch.Tensor) else outputs["loss"]
        item_losss = outputs["item_loss"]

        self.fabric.backward(loss)

        self._current_train_return = apply_to_collection(outputs, dtype=torch.Tensor, function=lambda x: x.detach())

        return loss, item_losss

    def step_scheduler(
        self,
        model: L.LightningModule,
        scheduler_cfg: Optional[Mapping[str, Union[L.fabric.utilities.types.LRScheduler, bool, str, int]]],
        level: Literal["step", "epoch"],
        current_value: int,
    ) -> None:

        # no scheduler
        if scheduler_cfg is None:
            return

        # wrong interval (step vs. epoch)
        if scheduler_cfg["interval"] != level:
            return

        # right interval, but wrong step wrt frequency
        if current_value % cast(int, scheduler_cfg["frequency"]) != 0:
            return

        # assemble potential monitored values
        possible_monitor_vals = {None: None}
        if isinstance(self._current_train_return, torch.Tensor):
            possible_monitor_vals.update("train_loss", self._current_train_return)
        elif isinstance(self._current_train_return, Mapping):
            possible_monitor_vals.update({"train_" + k: v for k, v in self._current_train_return.items()})

        if isinstance(self._current_val_return, torch.Tensor):
            possible_monitor_vals.update("val_loss", self._current_val_return)
        elif isinstance(self._current_val_return, Mapping):
            possible_monitor_vals.update({"val_" + k: v for k, v in self._current_val_return.items()})

        try:
            monitor = possible_monitor_vals[cast(Optional[str], scheduler_cfg["monitor"])]
        except KeyError as ex:
            possible_keys = list(possible_monitor_vals.keys())
            raise KeyError(
                f"monitor {scheduler_cfg['monitor']} is invalid. Possible values are {possible_keys}."
            ) from ex

        # rely on model hook for actual step
        model.lr_scheduler_step(scheduler_cfg["scheduler"], monitor)

    @property
    def should_validate(self) -> bool:
        """Whether to currently run validation."""
        return self.current_epoch % self.validation_frequency == 0

    def progbar_wrapper(self, iterable: Iterable, total: int, **kwargs: Any):
        """Wraps the iterable with tqdm for global rank zero.

        Args:
            iterable: the iterable to wrap with tqdm
            total: the total length of the iterable, necessary in case the number of batches was limited.

        """
        if self.fabric.is_global_zero:
            return tqdm(iterable, total=total, **kwargs)
        return iterable

    def load(self, state: Optional[Mapping], path: str) -> None:
        """Loads a checkpoint from a given file into state.

        Args:
            state: a mapping contaning model, optimizer and lr scheduler
            path: the path to load the checkpoint from

        """
        if state is None:
            state = {}

        remainder = self.fabric.load(path, state)
        self.global_step = remainder.pop("global_step")
        self.current_epoch = remainder.pop("current_epoch")

        if remainder:
            raise RuntimeError(f"Unused Checkpoint Values: {remainder}")

    def save(self, state: Optional[Mapping]) -> None:
        """Saves a checkpoint to the ``checkpoint_dir``

        Args:
            state: A mapping containing model, optimizer and lr scheduler.

        """
        if state is None:
            state = {}

        state.update(global_step=self.global_step, current_epoch=self.current_epoch)

        self.fabric.save(os.path.join(self.checkpoint_dir, f"epoch-{self.current_epoch:04d}.ckpt"), state)

    @staticmethod
    def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
        """Returns the latest checkpoint from the ``checkpoint_dir``

        Args:
            checkpoint_dir: the directory to search for checkpoints

        """
        if not os.path.isdir(checkpoint_dir):
            return None

        items = sorted(os.listdir(checkpoint_dir))

        if not items:
            return None

        return os.path.join(checkpoint_dir, items[-1])

    def _parse_optimizers_schedulers(
        self, configure_optim_output
    ) -> Tuple[
        Optional[L.fabric.utilities.types.Optimizable],
        Optional[Mapping[str, Union[L.fabric.utilities.types.LRScheduler, bool, str, int]]],
    ]:
        """Recursively parses the output of :meth:`lightning.pytorch.LightningModule.configure_optimizers`.

        Args:
            configure_optim_output: The output of ``configure_optimizers``.
                For supported values, please refer to :meth:`lightning.pytorch.LightningModule.configure_optimizers`.

        """
        _lr_sched_defaults = {"interval": "epoch", "frequency": 1, "monitor": "val_loss"}

        # single optimizer
        if isinstance(configure_optim_output, L.fabric.utilities.types.Optimizable):
            return configure_optim_output, None

        # single lr scheduler
        if isinstance(configure_optim_output, L.fabric.utilities.types.LRScheduler):
            return None, _lr_sched_defaults.update(scheduler=configure_optim_output)

        # single lr scheduler config
        if isinstance(configure_optim_output, Mapping):
            _lr_sched_defaults.update(configure_optim_output)
            return None, _lr_sched_defaults

        # list or tuple
        if isinstance(configure_optim_output, (list, tuple)):
            if all(isinstance(_opt_cand, L.fabric.utilities.types.Optimizable) for _opt_cand in configure_optim_output):
                # single optimizer in list
                if len(configure_optim_output) == 1:
                    return configure_optim_output[0][0], None

                raise NotImplementedError("BYOT only supports a single optimizer")

            if all(
                isinstance(_lr_cand, (L.fabric.utilities.types.LRScheduler, Mapping))
                for _lr_cand in configure_optim_output
            ):
                # single scheduler in list
                if len(configure_optim_output) == 1:
                    return None, self._parse_optimizers_schedulers(configure_optim_output[0])[1]

            # optimizer and lr scheduler
            elif len(configure_optim_output) == 2:
                opt_cands, lr_cands = (
                    self._parse_optimizers_schedulers(configure_optim_output[0])[0],
                    self._parse_optimizers_schedulers(configure_optim_output[1])[1],
                )
                return opt_cands, lr_cands

        return None, None

    @staticmethod
    def _format_iterable(
        prog_bar, candidates: Optional[Union[torch.Tensor, Mapping[str, Union[torch.Tensor, float, int]]]], prefix: str
    ):
        """Adds values as postfix string to progressbar.

        Args:
            prog_bar: a progressbar (on global rank zero) or an iterable (every other rank).
            candidates: the values to add as postfix strings to the progressbar.
            prefix: the prefix to add to each of these values.

        """
        if isinstance(prog_bar, tqdm) and candidates is not None:
            postfix_str = ""
            float_candidates = apply_to_collection(candidates, torch.Tensor, lambda x: x.item())
            if isinstance(candidates, torch.Tensor):
                postfix_str += f" {prefix}_loss: {float_candidates:.3f}"
            elif isinstance(candidates, Mapping):
                for k, v in float_candidates.items():
                    postfix_str += f" {prefix}_{k}: {v:.3f}"

            if postfix_str:
                prog_bar.set_postfix_str(postfix_str)