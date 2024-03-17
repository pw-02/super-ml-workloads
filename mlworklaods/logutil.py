from enum import Enum
from utils import  AverageMeter
from lightning.fabric import Fabric
import os
from collections import OrderedDict
# from lightning.fabric.loggers.logger import Logger
from typing import Any, Dict, List, Mapping, Optional, Set, Union
from torch import Tensor
from lightning.fabric.utilities.cloud_io import _is_dir, get_filesystem
import csv
from datetime import datetime

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


class ExperimentLogger():
    def __init__(self, fabric:Fabric, log_dir:str, log_freq:int, print_freq:int):
        self.log_freq = log_freq
        self.print_freq = print_freq
        self.fabric = fabric
        self.print_rank_0_only = True
        self.experiment_writer = ExperimentWriter(log_dir=log_dir, rank=self.fabric.local_rank)

    def save_train_batch_metrics(self,epoch,step,global_step,num_sampels,total_time,data_time,compute_time,loss, cahce_hit = False):
        metrics = OrderedDict({
                "device": self.fabric.local_rank,
                "epoch": epoch,
                "epoch_step": step,
                "global_step": global_step,
                "num_samples": num_sampels,
                "total_time": total_time,
                "data_time": data_time,
                "compute_time": compute_time,
                "loss": loss,
            })
        self.log_metrics(metrics = metrics,prefix='train.iteration',step=global_step,force_save=True)
    
    def save_train_epoch_metrics(self,epoch,num_samples,global_step,num_batches,total_time,data_time,compute_time,loss, cahce_hit = False):
        metrics = OrderedDict({
                "device": self.fabric.local_rank,
                "epoch": epoch,
                "num_samples": num_samples,
                "num_batches": num_batches,
                "total_time": total_time,
                "data_time": data_time,
                "compute_time": compute_time,
                "loss": loss,
            })
        self.log_metrics(metrics = metrics,prefix='train.epoch',step=global_step,force_save=True)
    
    
    def save_eval_batch_metrics(self,epoch,step,global_step,num_sampels,total_time,loss, top1, top5):
        metrics = OrderedDict({
                "device": self.fabric.local_rank,
                "epoch": epoch,
                "epoch_step": step,
                "global_step": global_step,
                "num_samples": num_sampels,
                "total_time": total_time,
                "loss": loss,
                "top1": top1,
                "top5": top5,
            })
        self.log_metrics(metrics = metrics,prefix='eval.iteration',step=global_step,force_save=True)
    
        
    def save_eval_epoch_metrics(self,epoch,num_samples,global_step,num_batches,total_time,loss, top1, top5):
        metrics = OrderedDict({
                "device": self.fabric.local_rank,
                "epoch": epoch,
                "num_samples": num_samples,
                "num_batches": num_batches,
                "total_time": total_time,
                "loss": loss,
                "top1": top1,
                "top5": top5,
            })
        self.log_metrics(metrics = metrics,prefix='eval.epoch',step=global_step,force_save=True)
    


    # def job_end(self):
    #     for dataset_type in self.epoch_metrics.keys():
    #         if self.epoch_metrics[dataset_type].epoch_time.count > 0:
    #             total_epochs =  self.epoch_metrics[dataset_type].epoch_time.count
    #             total_batches = self.epoch_metrics[dataset_type].num_samples.count
    #             total_samples = self.epoch_metrics[dataset_type].num_samples.sum
    #             total_time = self.epoch_metrics[dataset_type].epoch_time.sum
    #             compute_time =  self.epoch_metrics[dataset_type].compute_time.sum
    #             data_time =  self.epoch_metrics[dataset_type].data_time.sum
    #             cache_hits =  self.epoch_metrics[dataset_type].cache_hits.sum

    #             job_metrics_dict = OrderedDict(
    #                 {
    #                     "device": self.fabric.local_rank,
    #                     "num_epochs": total_epochs,
    #                     "num_batches": total_batches,
    #                     "num_samples": total_samples,
    #                     "total_time": total_time,
    #                     "data_time": data_time,
    #                     # "data_fetch": data_fetch_time,
    #                     # "data_transform": data_transform_time,
    #                     "compute_time": compute_time,
    #                     # "total_ips": calc_throughput_per_second(total_samples,total_time),
    #                     # "compute_ips": calc_throughput_per_second(total_samples, compute_time),
    #                     # "total_bps":calc_throughput_per_second(total_batches, total_time),
    #                     # "compute_bps":calc_throughput_per_second(total_batches,compute_time),
    #                     # "total_eps": calc_throughput_per_second(total_epochs,total_time),
    #                     # "compute_eps": calc_throughput_per_second(total_epochs,compute_time),
    #                     "loss(avg)": self.epoch_metrics[dataset_type].loss.val,
    #                     # "top1(avg)": self.epoch_metrics[dataset_type].top1.val,
    #                     # "top5(avg)": self.epoch_metrics[dataset_type].top5.val,
    #                     "cache_hits": self.epoch_metrics[dataset_type].cache_hits.val,#
    #                     # "cache_misses": self.epoch_metrics[dataset_type].cache_misses.val,#
    #                     })
    
    #             self.log_metrics(metrics=job_metrics_dict, step=1,prefix='train.job' if dataset_type == 'train' else 'val.job', force_save = True)

    #             # # if (self._fabric.is_global_zero and self.log_rank_zero_only) or (not self.log_rank_zero_only):
    #             # print_seperator_line()
    #             # print(f"JOB SUMMARY ({dataset_type}):")
    #             # # Filter keys to include only specific keys in the sub-dictionary
    #             # sub_dict_keys = ["device","num_epochs", "num_batches","total_time", "data_time","data_fetch","data_transform", "compute_time", "total_bps"]
    #             # sub_dict = {key: job_metrics_dict[key] for key in sub_dict_keys if key in job_metrics_dict}

    #             # total_cache_accesses = cache_hits + cache_misses
    #             # sub_dict["cache_hit_rate:"] = (cache_hits / total_cache_accesses) * 100
    #             # self.display_progress(sub_dict)
    #             # print_seperator_line()

    #             self.iteration_aggregator = IterationMetrics()
    #             return self.epoch_metrics[dataset_type].loss.avg
            
                
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

    def should_display(self, current_step):
        if (current_step + 1) % self.print_freq == 0:
            if (self.fabric.is_global_zero and self.print_rank_0_only) or (not self.print_rank_0_only):
                return True
        else:
            return False
        
    def should_flush(self, current_step):
        if (current_step + 1) % self.log_freq == 0:
            return True
        else:
            return False

    def log_hyperparams(self, params):
        # params.pop('__default_config__')
        # params.pop('config')
        self.experiment_writer.log_hparams(params)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3
