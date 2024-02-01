import csv
import logging
import os
from argparse import Namespace
from typing import Any, Dict, List, Mapping, Optional, Set, Union
from datetime import datetime
from torch import Tensor
from .utils import  AverageMeter,MinMeter, MaxMeter, calc_throughput_per_second

from lightning.fabric import Fabric
from lightning.fabric.loggers.logger import Logger
from lightning.fabric.utilities.cloud_io import _is_dir, get_filesystem
from lightning.fabric.utilities.rank_zero import rank_zero_only, rank_zero_warn

# A Python program to demonstrate working of OrderedDict
from collections import OrderedDict

# from image_classification.logger_config import configure_logger
# log = configure_logger()  # Initialize the logger


class BaseMetrics:
    def __init__(self):
        self.num_samples = AverageMeter("num_samples", ":6.3f")
        self.data_time = AverageMeter("data_time", ":6.3f")
        self.compute_time = AverageMeter("compute_time", ":6.3f")
        self.compute_ips = AverageMeter("compute_ips", ":6.3f")
        self.total_ips = AverageMeter("total_ips", ":6.3f")
        self.losses = AverageMeter("Loss", ":.4e")
        self.top1 = AverageMeter("Acc@1", ":6.2f")
        self.top5 = AverageMeter("Acc@5", ":6.2f")
        self.data_fetch_time = AverageMeter("data_fetch", ":6.3f")
        self.data_transform_time = AverageMeter("transform_time", ":6.3f")


class IterationMetrics(BaseMetrics):
    def __init__(self):
        super().__init__()
        self.iteration_time = AverageMeter("iteration_time", ":6.3f")
        self.cache_hits = 0
        self.cache_misses = 0

class EpochMetrics(BaseMetrics):
    def __init__(self):
        super().__init__()
        self.total_batches = AverageMeter("total_batches", ":6.3f")
        self.epoch_time = AverageMeter("epoch_time", ":6.3f")
        self.compute_bps = AverageMeter("compute_bps", ":6.3f")
        self.total_bps = AverageMeter("total_bps", ":6.3f")
        self.epoch_length = 0
        self.cache_hits = AverageMeter("total_batches", ":6.3f")
        self.cache_misses = AverageMeter("total_batches", ":6.3f")


class SUPERLogger(Logger):
    
    def __init__(self, fabric:Fabric, root_dir:str, flush_logs_every_n_steps:int, print_freq:int, exp_name: str = "logs"):
        self._root_dir = os.fspath(root_dir)
        self._name = exp_name
        self._version = self._get_next_exp_version()
        self._fs = get_filesystem(root_dir)
        self._flush_logs_every_n_steps = flush_logs_every_n_steps
        self._fabric = fabric
        self._experiment: Optional[_ExperimentWriter] = None
        self._print_freq = print_freq
        self.iteration_aggregator = IterationMetrics()
        self.epoch_aggregator  = {'train':EpochMetrics(), 'val':EpochMetrics()}
        self._log_rank_0_only = True


    def record_iteration_metrics(self,epoch,step, global_step, num_sampels ,iteration_time, data_time,
                        compute_time, compute_ips, total_ips, 
                        loss, top1, top5, batch_id, data_fetch_time, data_transform_time, is_training:bool, cache_hit:bool):
        dataset_type = 'train' if is_training else 'val'

        self.iteration_aggregator.iteration_time.update(iteration_time)
        self.iteration_aggregator.data_time.update(data_time)
        self.iteration_aggregator.compute_time.update(compute_time)
        self.iteration_aggregator.compute_ips.update(compute_ips)
        self.iteration_aggregator.total_ips.update(total_ips)
        self.iteration_aggregator.losses.update(loss)
        self.iteration_aggregator.top1.update(top1)
        self.iteration_aggregator.top5.update(top5)
        self.iteration_aggregator.data_fetch_time.update(data_fetch_time)
        self.iteration_aggregator.data_transform_time.update(data_transform_time)
        self.epoch_aggregator[dataset_type].num_samples.update(num_sampels)

        if cache_hit:
            self.iteration_aggregator.cache_hits +=1
        else:
            self.iteration_aggregator.cache_misses +=1

        iteration_metrics_dict = OrderedDict(
                                {
                                    "device": self._fabric.local_rank,
                                    "epoch": epoch,
                                    "epoch_step": step,
                                    "global_step": global_step,
                                    "batch_id": batch_id,
                                    "num_samples": num_sampels,
                                    #"batch_size": batch_size,
                                    "total_time": self.iteration_aggregator.iteration_time.val,
                                    "data_time": self.iteration_aggregator.data_time.val,
                                    "data_fetch": self.iteration_aggregator.data_fetch_time.val,
                                    "data_transform": self.iteration_aggregator.data_transform_time.val,
                                    "compute_time": self.iteration_aggregator.compute_time.val,
                                    "total_ips": self.iteration_aggregator.total_ips.val,
                                    "compute_ips": self.iteration_aggregator.compute_ips.val,
                                    "loss": self.iteration_aggregator.losses.val,
                                    "top1": self.iteration_aggregator.top1.val,
                                    "top5": self.iteration_aggregator.top5.val,#
                                    "cache_hit": cache_hit
                                    }
                            )
       
        self.log_metrics(metrics = iteration_metrics_dict,
                         prefix=f'{dataset_type}.iteration',
                         step=global_step,
                         force_save = False)

        if step is not None and (step + 1) % self._print_freq == 0:
            if (self._fabric.is_global_zero and self.log_rank_zero_only) or (not self.log_rank_zero_only):
                # Filter keys to include only specific keys in the sub-dictionary
                new_keys_values = {"epoch": epoch, "step": f"{step}/{self.epoch_aggregator[dataset_type].epoch_length - 1}"}
                sub_dict_keys = ["total_time", "data_time", "data_fetch", "data_transform", "compute_time", "cache_hit"]
                sub_dict = {key: iteration_metrics_dict[key] for key in sub_dict_keys if key in iteration_metrics_dict}
                new_keys_values.update(sub_dict)
                self.display_progress(new_keys_values)
        return iteration_metrics_dict


    def display_progress(self,display_info:dict[str,float]):
        self._fabric.print(", ".join(f"{key}: {round(float(value), 4)}" if isinstance(value, float) else f"{key}: {value}" for key, value in display_info.items()))
        #print(", ".join(f"{key}: {value}" for key, value in display_info.items()))
        # reset traing metrics

    def epoch_start(self, epoch_length:int, is_training:bool):
        dataset_type = 'train' if is_training else 'val'
        self.epoch_aggregator[dataset_type].epoch_length = epoch_length


    def epoch_end(self,epoch, is_training:bool):
        
        dataset_type = 'train' if is_training else 'val'

        total_batches = self.iteration_aggregator.iteration_time.count
        epoch_time = self.iteration_aggregator.iteration_time.sum
        data_time = self.iteration_aggregator.data_time.sum  
        data_fetch_time = self.iteration_aggregator.data_fetch_time.sum
        data_transform_time = self.iteration_aggregator.data_transform_time.sum
        compute_time = self.iteration_aggregator.compute_time.sum
        total_samples = self.epoch_aggregator[dataset_type].num_samples.sum
        cache_hits = self.iteration_aggregator.cache_hits
        cache_misses = self.iteration_aggregator.cache_misses

        self.epoch_aggregator[dataset_type].total_batches.update(total_batches)
        self.epoch_aggregator[dataset_type].epoch_time.update(epoch_time)
        self.epoch_aggregator[dataset_type].data_time.update(data_time)
        self.epoch_aggregator[dataset_type].data_fetch_time.update(data_fetch_time)
        self.epoch_aggregator[dataset_type].data_transform_time.update(data_transform_time)
        self.epoch_aggregator[dataset_type].compute_time.update(compute_time)
        self.epoch_aggregator[dataset_type].losses.update(self.iteration_aggregator.losses.avg)
        self.epoch_aggregator[dataset_type].top1.update(self.iteration_aggregator.top1.avg)
        self.epoch_aggregator[dataset_type].top5.update(self.iteration_aggregator.top5.avg)
        
        self.epoch_aggregator[dataset_type].cache_misses.update(cache_misses)
        self.epoch_aggregator[dataset_type].cache_hits.update(cache_hits)

        self.epoch_aggregator[dataset_type].compute_ips.update(calc_throughput_per_second(total_samples,compute_time))
        self.epoch_aggregator[dataset_type].total_ips.update(calc_throughput_per_second(total_samples,epoch_time))
        
        self.epoch_aggregator[dataset_type].compute_bps.update(calc_throughput_per_second(total_batches,compute_time))
        self.epoch_aggregator[dataset_type].total_bps.update(calc_throughput_per_second(total_batches,epoch_time))
        
        epoch_metrics = OrderedDict(
                                {
                                    "device": self._fabric.local_rank,
                                    "epoch": epoch,
                                    "num_samples": total_samples,
                                    "num_batches": total_batches,
                                    "total_time": epoch_time,
                                    "data_time": self.epoch_aggregator[dataset_type].data_time.val,
                                    "data_fetch": self.epoch_aggregator[dataset_type].data_fetch_time.val,
                                    "data_transform": self.epoch_aggregator[dataset_type].data_transform_time.val,
                                    "compute_time": compute_time,
                                    "total_ips": self.epoch_aggregator[dataset_type].total_ips.val,
                                    "compute_ips": self.epoch_aggregator[dataset_type].compute_ips.val,
                                    "total_bps": self.epoch_aggregator[dataset_type].total_bps.val,
                                    "compute_bps": self.epoch_aggregator[dataset_type].compute_bps.val,
                                    "loss(avg)": self.epoch_aggregator[dataset_type].losses.val,
                                    "top1(avg)": self.epoch_aggregator[dataset_type].top1.val,
                                    "top5(avg)": self.epoch_aggregator[dataset_type].top5.val,#
                                    "cache_hits": self.epoch_aggregator[dataset_type].cache_hits.val,#
                                    "cache_misses": self.epoch_aggregator[dataset_type].cache_misses.val,#

                                    })
        
        self.log_metrics(metrics=epoch_metrics, prefix='train.epoch' if is_training else 'val.epoch', force_save = True)
        
        if (self._fabric.is_global_zero and self.log_rank_zero_only) or (not self.log_rank_zero_only):
                print_seperator_line()
                print(f"EPOCH {epoch} SUMMARY ({dataset_type}):")
                # Filter keys to include only specific keys in the sub-dictionary
                sub_dict_keys = ["device", "num_batches","total_time","data_time", "data_fetch", "data_transform", "compute_time" ,"total_bps"]
                sub_dict = {key: epoch_metrics[key] for key in sub_dict_keys if key in epoch_metrics}

                total_cache_accesses = cache_hits + cache_misses

                sub_dict["cache_hit_rate:"] = (cache_hits / total_cache_accesses) * 100

                self.display_progress(sub_dict)
                print_seperator_line()
        
        self.iteration_aggregator = IterationMetrics()
    

    def job_end(self):

        for dataset_type in self.epoch_aggregator.keys():
            if self.epoch_aggregator[dataset_type].compute_bps.count > 0:

                total_epochs =  self.epoch_aggregator[dataset_type].epoch_time.count
                total_batches = self.epoch_aggregator[dataset_type].num_samples.count
                total_samples = self.epoch_aggregator[dataset_type].num_samples.sum
                total_time = self.epoch_aggregator[dataset_type].epoch_time.sum
                compute_time =  self.epoch_aggregator[dataset_type].compute_time.sum
                data_time =  self.epoch_aggregator[dataset_type].data_time.sum
            
                data_fetch_time =  self.epoch_aggregator[dataset_type].data_fetch_time.sum
                data_transform_time =  self.epoch_aggregator[dataset_type].data_transform_time.sum


                cache_hits =  self.epoch_aggregator[dataset_type].cache_hits.sum
                cache_misses =  self.epoch_aggregator[dataset_type].cache_misses.sum

                job_metrics_dict = OrderedDict(
                    {
                        "device": self._fabric.local_rank,
                        "num_epochs": total_epochs,
                        "num_batches": total_batches,
                        "num_samples": total_samples,
                        "total_time": total_time,
                        "data_time": data_time,
                        "data_fetch": data_fetch_time,
                        "data_transform": data_transform_time,
                        "compute_time": compute_time,
                        "total_ips": calc_throughput_per_second(total_samples,total_time),
                        "compute_ips": calc_throughput_per_second(total_samples, compute_time),
                        "total_bps":calc_throughput_per_second(total_batches, total_time),
                        "compute_bps":calc_throughput_per_second(total_batches,compute_time),
                        "total_eps": calc_throughput_per_second(total_epochs,total_time),
                        "compute_eps": calc_throughput_per_second(total_epochs,compute_time),
                        "loss(avg)": self.epoch_aggregator[dataset_type].losses.val,
                        "top1(avg)": self.epoch_aggregator[dataset_type].top1.val,
                        "top5(avg)": self.epoch_aggregator[dataset_type].top5.val,
                        "cache_hits": self.epoch_aggregator[dataset_type].cache_hits.val,#
                        "cache_misses": self.epoch_aggregator[dataset_type].cache_misses.val,#
                        })
    
                self.log_metrics(
                            metrics=job_metrics_dict, 
                            step=1,
                            prefix='train.job' if dataset_type == 'train' else 'val.job',
                            force_save = True)
            
            if (self._fabric.is_global_zero and self.log_rank_zero_only) or (not self.log_rank_zero_only):
                print_seperator_line()
                print(f"JOB SUMMARY ({dataset_type}):")
                # Filter keys to include only specific keys in the sub-dictionary
                sub_dict_keys = ["device","num_epochs", "num_batches","total_time", "data_time","data_fetch","data_transform", "compute_time", "total_bps"]
                sub_dict = {key: job_metrics_dict[key] for key in sub_dict_keys if key in job_metrics_dict}

                total_cache_accesses = cache_hits + cache_misses
                sub_dict["cache_hit_rate:"] = (cache_hits / total_cache_accesses) * 100
                self.display_progress(sub_dict)
                print_seperator_line()
    

    def _get_next_exp_version(self):
        from lightning.fabric.utilities.cloud_io import _is_dir, get_filesystem

        versions_root = os.path.join(self.root_dir, self.name)
        fs = get_filesystem(self.root_dir)
        if not _is_dir(fs, versions_root, strict=True):
                #log.warning("Missing logger folder: %s", versions_root)
                return 0
        
        existing_versions = []
        for d in fs.listdir(versions_root):
            full_path = d["name"]
            name = os.path.basename(full_path)
            if _is_dir(fs, full_path) and name.startswith("version_"):
                dir_ver = name.split("_")[1]
                if dir_ver.isdigit():
                    existing_versions.append(int(dir_ver))
        if len(existing_versions) == 0:
            return 0
        return max(existing_versions) + 1
    
    @property
    def log_rank_zero_only(self) -> bool:
        """Gets the rank of the experiment.
        Returns:
            The rank of the experiment.
        """
        return self._log_rank_0_only

  
    @property
    def name(self) -> str:
        """Gets the name of the experiment.
        Returns:
            The name of the experiment.
        """
        return self._name
    
    @property
    def version(self) -> Union[int, str]:
        """Gets the version of the experiment.

        Returns:
            The version of the experiment if it is specified, else the next version.

        """
        if self._version is None:
            self._version = self._get_next_version()
        return self._version
    
    @property
    def root_dir(self) -> str:
        """Gets the save directory where the versioned experiments are saved."""
        return self._root_dir

    @property
    def log_dir(self) -> str:
        """The log directory for this run.

        By default, it is named ``'version_${self.version}'`` but it can be overridden by passing a string value for the
        constructor's version parameter instead of ``None`` or an int.

        """
        # create a pseudo standard path
        version = self.version if isinstance(self.version, str) else f"version_{self.version}"
        return os.path.join(self._root_dir, self.name, version)
    
    @property
    #@rank_zero_experiment
    def experiment(self) -> "_ExperimentWriter":
        """Actual ExperimentWriter object. To use ExperimentWriter features anywhere in your code, do the following.

        Example::

            self.logger.experiment.some_experiment_writer_function()

        """
        if self._experiment is not None:
            return self._experiment

        os.makedirs(self._root_dir, exist_ok=True)
        self._experiment = _ExperimentWriter(log_dir=self.log_dir, rank = self._fabric.local_rank)
        return self._experiment

    @rank_zero_only
    def log_hyperparams(self, params):
        params.pop('__default_config__')
        params.pop('config')
        self.experiment.log_hparams(params.as_flat())


    #@rank_zero_only
    def log_metrics(  # type: ignore[override]
        self, metrics: Dict[str, Union[Tensor, float]],prefix:str, step:int = None, force_save:bool = False
    ) -> None:
        
        metrics["timestamp"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        metrics["datetime"] = datetime.now().timestamp()

        if prefix:
            separator = '.'
            metrics =  OrderedDict({f"{prefix}{separator}{k}": v for k, v in metrics.items()})
        
        self.experiment.log_metrics(metrics)

        if force_save:
            self.save()
        elif step is not None and (step + 1) % self._flush_logs_every_n_steps == 0:
            self.save()

    #@rank_zero_only
    def save(self)-> None: 
        super().save()
        self.experiment.save()

    #@rank_zero_only
    def finalize(self, status: str) -> None:
        if self._experiment is None:
            # When using multiprocessing, finalize() should be a no-op on the main process, as no experiment has been
            # initialized there
            return
        self.save()

    def _get_next_version(self) -> int:
        versions_root = os.path.join(self._root_dir, self.name)

        if not _is_dir(self._fs, versions_root, strict=True):
            #log.warning("Missing logger folder: %s", versions_root)
            return 0

        existing_versions = []
        for d in self._fs.listdir(versions_root):
            full_path = d["name"]
            name = os.path.basename(full_path)
            if _is_dir(self._fs, full_path) and name.startswith("version_"):
                dir_ver = name.split("_")[1]
                if dir_ver.isdigit():
                    existing_versions.append(int(dir_ver))

        if len(existing_versions) == 0:
            return 0

        return max(existing_versions) + 1 


class _ExperimentWriter:
    r"""Experiment writer for CSVLogger.

    Args:
        log_dir: Directory for the experiment logs

    """
    NAME_METRICS_FILE = "metrics.csv"
    NAME_HPARAMS_FILE = "hparams.yaml"

    def __init__(self, log_dir: str, rank:int) -> None:
        self.metrics: List[Dict[str, float]] = []
        self.metrics_keys: List[str] = []
        self.hparams: Dict[str, Any] = {}
        self._fs = get_filesystem(log_dir)
        self.log_dir = log_dir
        if self._fs.exists(self.log_dir) and self._fs.listdir(self.log_dir):
            rank_zero_warn(
                f"Experiment logs directory {self.log_dir} exists and is not empty."
                " Previous log files in this directory will be deleted when the new ones are saved!"
            )
        self._fs.makedirs(self.log_dir, exist_ok=True)
        
        self.metrics_file_path = os.path.join(self.log_dir, f'rank_{rank}_{self.NAME_METRICS_FILE}')

    def log_metrics(self, metrics_dict: Dict[str, float]) -> None:
        """Record metrics."""

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
        file_exists = self._fs.isfile(self.metrics_file_path)

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


        #current_keys = set().union(*self.metrics)
        #new_keys = current_keys - set(self.metrics_keys)
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
        
        # params.pop('__default_config__')
        # params.pop('config')
        # c = params.as_flat()
        #"""Record hparams."""
        #self.hparams.update(params)
        """Save recorded hparams and metrics into files."""
        hparams_file = os.path.join(self.log_dir, self.NAME_HPARAMS_FILE)
        save_hparams_to_yaml(hparams_file, params)

def print_seperator_line():
            print('-' * 100)