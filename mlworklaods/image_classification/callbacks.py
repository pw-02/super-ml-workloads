from mlworklaods.log_utils import ExperimentLogger, AverageMeter, ProgressMeter, create_exp_summary_report
import time
from collections import OrderedDict
from typing import Any, Dict, List, Mapping, Optional, Set, Union
from datetime import datetime
from torch import Tensor
import csv
from lightning.fabric.utilities.cloud_io import _is_dir, get_filesystem
import os
import json

class LoggingCallback:
    def __init__(self, log_dir, log_freq):
        self.log_freq = log_freq
        self.local_rank = None
        self.log_dir = log_dir
        self.experiment_writer =None
        
        #batch metrics
        self.batch_time:AverageMeter = AverageMeter("Time", ":6.3f")
        self.data_time:AverageMeter = AverageMeter("Data", ":6.3f")
        self.fetch_time:AverageMeter = AverageMeter("Fetch", ":6.3f")
        self.transform_time:AverageMeter = AverageMeter("Transform", ":6.3f")
        self.compute_time:AverageMeter = AverageMeter("Compute", ":6.3f")
        self.losses:AverageMeter = AverageMeter("Loss", ":6.2f")
        self.top1:AverageMeter = AverageMeter("Acc1", ":6.2f")
        self.top5:AverageMeter = AverageMeter("Acc5", ":6.2f")
        self.cache_hits:AverageMeter = AverageMeter("hit%", ":6.2f")
        self.timer = time.perf_counter()
        self.current_epoch = None
        self.current_batch_idx = None
        self.total_samples = 0
        self.total_cache_hits =0
    
    def on_train_epoch_start(self, current_epoch=0, local_rank=0):
        
        if self.experiment_writer is None:
            self.experiment_writer = ExperimentWriter(log_dir=self.log_dir, rank=local_rank)

        self.current_epoch = current_epoch
        self.local_rank = local_rank 
        self.timer = time.perf_counter()
 
    def on_train_batch_end(self, 
        batch_idx,
        batch_size,
        batch_time,
        data_time,
        fetch_time,
        transform_time,
        compute_time,
        cache_hits,
        loss,
        acc5,
        acc1,
        cpu_usage,
        gpu_usage,
        global_step):
        
        self.batch_time.update(batch_time)
        self.data_time.update(data_time)
        self.fetch_time.update(fetch_time)
        self.transform_time.update(transform_time)
        self.compute_time.update(compute_time)
        self.losses.update(loss)
        self.top1.update(acc5)
        self.top5.update(acc1)
        self.total_cache_hits += cache_hits
        self.total_samples += batch_size
        
        if batch_idx % self.log_freq == 0:
            metrics = OrderedDict({
            "device": self.local_rank,
            "global_step": global_step,
            "epoch": self.current_epoch,
            "batch_idx": batch_idx,
            "batch_size": batch_size,
            "batch_time": batch_time,
            "data_time": data_time,
            "fetch_time": fetch_time,
            "transform_time": transform_time,
            "compute_time": compute_time,
            "cache_hits": cache_hits,
            "loss": loss,
            "acc5": acc1,
            "acc5": acc5,
            "cpu_usge": json.dumps(cpu_usage),
            "gpu_usge": json.dumps(gpu_usage)   
            })
            self.log_metrics(metrics = metrics,prefix='train.iteration',step=global_step,force_save=True)

    
    def log_metrics( self, metrics: Dict[str, Union[Tensor, float]],prefix:str, step:int = None, force_save:bool = False) -> None:
        metrics["timestamp"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        metrics["datetime"] = datetime.now().timestamp()
        if prefix:
            separator = '.'
            metrics =  OrderedDict({f"{prefix}{separator}{k}": v for k, v in metrics.items()})

        self.experiment_writer.log_metrics(metrics)
        if force_save:
            self.experiment_writer.save()
        elif  step is not None and (step + 1) % self.log_freq == 0:
            self.experiment_writer.save()


class ExperimentWriter:
    NAME_METRICS_FILE = "metrics.csv"
    NAME_HPARAMS_FILE = "hparams.yaml"

    def __init__(self, log_dir: str, rank:int) -> None:
        self.metrics: List[Dict[str, float]] = []
        self.metrics_keys: List[str] = []
        self.hparams: Dict[str, Any] = {}
        self._fs = get_filesystem(log_dir)
        self.log_dir = log_dir
        self._fs.makedirs(self.log_dir, exist_ok=True)
        self.metrics_file_path = os.path.join(self.log_dir, f'rank_{rank}_{self.NAME_METRICS_FILE}')
        
    def log_metrics(self, metrics_dict: Dict[str, float]) -> None:
        def _handle_value(value: Union[Tensor, Any]) -> Any:
            if isinstance(value, Tensor):
                return value.item()
            return value
        metrics = OrderedDict({k: _handle_value(v) for k, v in metrics_dict.items()})
        self.metrics.append(metrics)

    def save(self) -> None:
        """Save recorded metrics into files."""
        if not self.metrics:
            return
        new_keys = self._record_new_keys()
        file_exists = os.path.exists(self.metrics_file_path)

        if new_keys and file_exists:
            # we need to re-write the file if the keys (header) change
            self._rewrite_with_new_header(self.metrics_keys)

        with self._fs.open(self.metrics_file_path, mode=("a" if file_exists else "w"), newline="") as file:
            writer = csv.DictWriter(file, fieldnames=self.metrics_keys)
            if not file_exists:
                # only write the header if we're writing a fresh file
                writer.writeheader()
            writer.writerows(self.metrics)

        self.metrics = []  # reset

    def _record_new_keys(self) -> Set[str]:
        """Records new keys that have not been logged before."""
        current_keys = list(OrderedDict.fromkeys(key for keys_set in self.metrics for key in keys_set))
        new_keys = [key for key in current_keys if key not in self.metrics_keys]
        self.metrics_keys.extend(new_keys)
        return new_keys

    def _rewrite_with_new_header(self, fieldnames: List[str]) -> None:
        with self._fs.open(self.metrics_file_path, "r", newline="") as file:
            metrics = list(csv.DictReader(file))

        with self._fs.open(self.metrics_file_path, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metrics)
    
    def log_hparams(self, params: Dict[str, Any]) -> None:
        from lightning.pytorch.core.saving import save_hparams_to_yaml
        """Save recorded hparams and metrics into files."""
        hparams_file = os.path.join(self.log_dir, self.NAME_HPARAMS_FILE)
        save_hparams_to_yaml(hparams_file, params)

