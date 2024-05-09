from typing import Dict, Any
from collections import OrderedDict
from lightning.fabric.utilities.cloud_io import _is_dir, get_filesystem
from torch import Tensor
import os
import csv
from datetime import datetime
from lightning.fabric import Fabric
from typing import Any, Dict, List, Mapping, Optional, Set, Union
from mlworklaods.utils import calc_throughput, calculate_average
from enum import Enum


class ExperimentLogger():
    def __init__(self, fabric:Fabric, log_dir:str, log_freq:int):
        self.log_freq = log_freq
        self.fabric = fabric
        self.print_rank_0_only = False
        self.experiment_writer = ExperimentWriter(log_dir=log_dir, rank=self.fabric.local_rank)
        self.train_job_metrics = None
        self.eval_job_metrics = None
        self.total_tokens = None

    def save_train_batch_metrics(self,epoch,step,global_step,num_samples,total_time,data_time,compute_time,loss, avg_cpu, max_cpu,cache_hits, avg_gpu=0, max_gpu=0, num_tokens = None, acc1 = None, acc5 = None, ):
        metrics = OrderedDict({
                "device": self.fabric.local_rank,
                "epoch": epoch,
                "epoch_step": step,
                "global_step": global_step,
                "num_samples": num_samples,
                "total_time": total_time,
                "data_time": data_time,
                "compute_time": compute_time,
                "cache_hits": cache_hits,
                "loss": loss,
                "avg_cpu": avg_cpu,
                "max_cpu": max_cpu,
                "avg_gpu": avg_gpu,
                "max_gpu": max_gpu,   
            })
        
        if acc1 is not None:
            metrics['acc1'] = acc1

        if acc5 is not None:
            metrics['acc5'] = acc5

        if num_tokens is not None:
            metrics['num_tokens'] = num_tokens

        self.log_metrics(metrics = metrics,prefix='train.iteration',step=global_step,force_save=True)
    
    def save_train_epoch_metrics(self,epoch,num_samples,global_step,num_batches,total_time,data_time,compute_time,loss, 
                                 avg_cpu, max_cpu, cache_hits, avg_gpu=0, max_gpu=0, num_tokens = None,acc1 = None, acc5=None,):
        metrics = OrderedDict({
                "device": self.fabric.local_rank,
                "epoch": epoch,
                "num_samples": num_samples,
                "num_batches": num_batches,
                "total_time": total_time,
                "data_time": data_time,
                "compute_time": compute_time,
                "cache_hits": cache_hits,
                "loss": loss,
                "avg_cpu": avg_cpu,
                "max_cpu": max_cpu,
                "avg_gpu": avg_gpu,
                "max_gpu": max_gpu

            }) 
        
        if acc1 is not None:
            metrics['acc1'] = acc1

        if acc5 is not None:
            metrics['acc5'] = acc5

        if num_tokens is not None:
            metrics['num_tokens'] = num_tokens
            self.total_tokens = num_tokens

        if self.train_job_metrics == None:
            self.train_job_metrics = AggregatedEpochMetrics()
            
        self.train_job_metrics.cache_hits += cache_hits
        self.train_job_metrics.epoch_compute_times.update(compute_time)
        self.train_job_metrics.epoch_dataload_times.update(data_time)
        self.train_job_metrics.epoch_times.update(total_time)
        self.train_job_metrics.total_batches += num_batches
        self.train_job_metrics.total_samples +=num_samples
        self.train_job_metrics.epoch_losses.update(loss)
        self.train_job_metrics.epoch_acc1.update(loss)
        self.train_job_metrics.epoch_acc5.update(loss)
        self.train_job_metrics.cpu_util.update(avg_cpu)
        self.train_job_metrics.gpu_util.update(avg_gpu)

        self.log_metrics(metrics = metrics,prefix='train.epoch',step=global_step,force_save=True)
    
    def save_eval_batch_metrics(self,epoch,step,global_step,num_sampels,total_time,loss, acc1, acc5):
        metrics = OrderedDict({
                "device": self.fabric.local_rank,
                "epoch": epoch,
                "epoch_step": step,
                "global_step": global_step,
                "num_samples": num_sampels,
                "total_time": total_time,
                "loss": loss,
                "acc1" : acc1,
                "acc5" : acc5,
            })
        self.log_metrics(metrics = metrics,prefix='eval.iteration',step=global_step,force_save=True)
    
        
    def save_eval_epoch_metrics(self,epoch,num_samples,global_step,num_batches,total_time,loss, acc1, acc5):
        metrics = OrderedDict({
                "device": self.fabric.local_rank,
                "epoch": epoch,
                "num_samples": num_samples,
                "num_batches": num_batches,
                "total_time": total_time,
                "loss": loss,
                "acc1" : acc1,
                "acc5" : acc5,
            })
        if self.eval_job_metrics == None:
            self.eval_job_metrics = AggregatedEpochMetrics()
        self.eval_job_metrics.total_samples +=num_samples
        self.eval_job_metrics.total_batches += num_batches
        self.eval_job_metrics.epoch_times.update(total_time)
        self.eval_job_metrics.epoch_losses.update(loss)
        self.eval_job_metrics.epoch_acc5.update(acc5)
        self.eval_job_metrics.epoch_acc1.update(acc1)
        
        self.log_metrics(metrics = metrics,prefix='eval.epoch',step=global_step,force_save=True)
    
    def create_job_report(self):
        if self.train_job_metrics:
            metrics_dict = OrderedDict(
                {
                    "device": self.fabric.local_rank,
                    "total_epochs": self.train_job_metrics.epoch_times.count,
                    "total_batches": self.train_job_metrics.total_batches,
                    "total_samples": self.train_job_metrics.total_samples,
                    "total_time": self.train_job_metrics.epoch_times.sum,
                    "data_time": self.train_job_metrics.epoch_dataload_times.sum,
                    "compute_time": self.train_job_metrics.epoch_compute_times.sum,
                    "bps": calc_throughput(self.train_job_metrics.total_batches, self.train_job_metrics.epoch_times.sum),
                    "compute_bps": calc_throughput(self.train_job_metrics.total_batches,self.train_job_metrics.epoch_compute_times.sum),
                    "cache_hits": self.train_job_metrics.cache_hits,
                    "loss": self.train_job_metrics.epoch_losses.val,
                    "acc1" : self.train_job_metrics.epoch_acc1.val,
                    "acc5" :  self.train_job_metrics.epoch_acc5.val,
                    "cpu_util" :  self.train_job_metrics.cpu_util.val,
                    "gpu_util" :  self.train_job_metrics.gpu_util.val,
                })
        if self.total_tokens is not None:
            metrics_dict["total_tokens"] = self.total_tokens

        console_line = f'Total_Batches: {metrics_dict["total_batches"]}|Total_Time:{metrics_dict["total_time"]}|Data_time: {metrics_dict["data_time"]}|Compute_time: {metrics_dict["compute_time"]}|bps: {metrics_dict["bps"]}'
        
        if self.print_rank_0_only:
            self.fabric.print(console_line)
        else:
            print(console_line)
        
        self.log_metrics(metrics=metrics_dict, step=1,prefix='train.job', force_save = True)
    
        if self.eval_job_metrics:
            metrics_dict = OrderedDict(
                {
                    "device": self.fabric.local_rank,
                    "total_epochs": self.eval_job_metrics.epoch_times.count,
                    "total_batches": self.eval_job_metrics.total_batches,
                    "total_samples": self.eval_job_metrics.total_samples,
                    "total_time": self.eval_job_metrics.epoch_times.sum,
                    "loss": self.eval_job_metrics.epoch_losses.val,
                    "top5": self.eval_job_metrics.epoch_acc5.val,
                    "top1": self.eval_job_metrics.epoch_acc1.val,

                })
        self.log_metrics(metrics=metrics_dict, step=1,prefix='eval.job', force_save = True)

              
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


    def log_hyperparams(self, params):
        self.experiment_writer.log_hparams(params)

def create_exp_summary_report(folder_path):
    import pandas as pd
    import glob
    import yaml
    import os
    output_file_path = os.path.join(folder_path, f'exp_summary_report.xlsx')
    # Get a list of all CSV files in the specified folder
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    # Specify column categories
    categories = ['train.iteration', 'train.epoch', 'train.job', 'val.iteration', 'val.epoch', 'val.job']
    # Create a dictionary to store accumulated data for each category
    category_data = {}
    # Iterate through each CSV file
    for csv_file in csv_files:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_file)

        # Iterate through each category
        for category in categories:
            # Select columns starting with the current category
            selected_columns = [col for col in df.columns if col.startswith(category)]

            # Update the dictionary for the current category with the selected columns
            if selected_columns:
                data_dict = df[selected_columns].to_dict(orient='list')
                # Remove NaN values from the dictionary
                data_dict_no_nan = {key: [value for value in values if pd.notna(value)] for key, values in data_dict.items()}
                
                if len(data_dict_no_nan) >  0:
                     # Accumulate data for the current category across all CSV files
                    if category in category_data:
                        for key, values in data_dict_no_nan.items():
                            category_data[category][key].extend(values)
                    else:
                        category_data[category] = data_dict_no_nan

    # Read 'hparams.yaml' file
    hparams_file_path = os.path.join(folder_path, 'hparams.yaml')
    hparams_data = {}
    if os.path.isfile(hparams_file_path):
        with open(hparams_file_path, 'r') as hparams_file:
            hparams_data = yaml.safe_load(hparams_file)

    category_data["overall_summary"] = summarize_across_all_devices(category_data=category_data)

    # Create an Excel writer for the output file
    with pd.ExcelWriter(output_file_path, engine='xlsxwriter') as writer: 
        # df_hparams = pd.DataFrame.from_dict(hparams_data, orient='index')
        # df_hparams.to_excel(writer, sheet_name='hparams', header=False, index=True)

        # Iterate through each category
        for category, data_dict in  reversed(category_data.items()):
            # Skip creating sheets if the data for the current category is empty
            if data_dict:
                # Replace invalid characters in the sheet name
                df = pd.DataFrame.from_dict(data_dict, orient='columns')
                # for col in ['train.iteration.batch_id', 'val.iteration.batch_id']:
                #     if col in df.columns:
                #         df[col] = df[col].apply(lambda x: '{:.0f}'.format(x))
                
                df.to_excel(writer, sheet_name=category, index=False)
    
    return output_file_path

def summarize_across_all_devices(category_data):
    from collections import OrderedDict

    summary = OrderedDict({
        "dataset_type": [],
        "num_devices": [],
        "total_time": [],
        "data_time": [],
        "compute_time": [],
        "cpu_util": [],   
        "gpu_util": [],     
        "total_samples": [],
        "total_batches": [],
        "batches/sec": [],
        "total_epochs": [],
        "total_cache_hits": [],
        "loss": [],
        "acc1": [],
        "acc5": []
    })

    keys_to_check = ['train.job', 'val.job']

    for key in keys_to_check:
        if key in category_data:        
            summary['dataset_type'].append('Train' if key == 'train.job' else 'Validation')
            summary['num_devices'].append(len(category_data[key][f'{key}.device']))
            summary['total_time'].append(calculate_average(category_data[key][f'{key}.total_time']))                
            summary['data_time'].append(calculate_average(category_data[key][f'{key}.data_time']))
            summary['compute_time'].append(calculate_average(category_data[key][f'{key}.compute_time']))
            summary['total_samples'].append(sum(category_data[key][f'{key}.total_samples']))
            summary['total_batches'].append(sum(category_data[key][f'{key}.total_batches']))
            summary['total_cache_hits'].append(sum(category_data[key][f'{key}.cache_hits']))
            # summary['samples/sec'].append(calculate_average(category_data[key][f'{key}.total_ips']))                
            summary['batches/sec'].append(calculate_average(category_data[key][f'{key}.bps']))
            summary['total_epochs'].append(max(category_data[key][f'{key}.total_epochs']))
            # summary['epochs/sec'].append(calculate_average(category_data[key][f'{key}.total_eps']))                
            summary['loss'].append(calculate_average(category_data[key][f'{key}.loss']))
            summary['acc1'].append(calculate_average(category_data[key][f'{key}.acc1']))
            summary['acc5'].append(calculate_average(category_data[key][f'{key}.acc5']))
            summary['cpu_util'].append(calculate_average(category_data[key][f'{key}.cpu_util']))
            summary['gpu_util'].append(calculate_average(category_data[key][f'{key}.gpu_util']))
            
            if f'{key}.total_tokens' in category_data[key]:
            # if category_data[key][f'{key}.total_tokens']:
                if 'total_tokens' not in summary.keys():
                    summary['total_tokens'] = []  
                summary['total_tokens'].append(sum(category_data[key][f'{key}.total_tokens']))

    return summary


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        #fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        fmtstr = "{name}:{val" + self.fmt +"}"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, fabric:Fabric = None):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        # entries += [batch_id]
        return '\t'.join(entries)
        # if fabric:
        #     fabric.print('\t'.join(entries))
        # else:
        #     print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
    

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

class AggregatedEpochMetrics():
    def __init__(self):
        self.total_samples = 0
        self.total_batches = 0
        self.epoch_times = AverageMeter("epoch_times", ":6.3f")
        self.cache_hits = 0
        self.epoch_dataload_times = AverageMeter("data_times", ":6.3f")
        self.epoch_compute_times = AverageMeter("compute_times", ":6.3f")
        self.epoch_losses = AverageMeter('Loss', ':.4e')
        self.epoch_acc1 = AverageMeter('Acc@1', ':6.2f')
        self.epoch_acc5  = AverageMeter('Acc@5', ':6.2f')
        self.cpu_util  = AverageMeter('cpu_util', ':6.2f')
        self.gpu_util  = AverageMeter('gpu_util', ':6.2f')

        self.total_time = None

def get_next_exp_version(root_dir, name):
    from lightning.fabric.utilities.cloud_io import _is_dir, get_filesystem
    import os
    versions_root = os.path.join(root_dir, name)

    fs = get_filesystem(root_dir)
    if not _is_dir(fs, versions_root, strict=True):
            #log.warning("Missing logger folder: %s", versions_root)
            fs.makedirs(versions_root, exist_ok=True)

            f"version_{0}"
    
    existing_versions = []
    for d in fs.listdir(versions_root):
        full_path = d["name"]
        name = os.path.basename(full_path)
        if _is_dir(fs, full_path) and name.startswith("version_"):
            dir_ver = name.split("_")[1]
            if dir_ver.isdigit():
                existing_versions.append(int(dir_ver))
    if len(existing_versions) == 0:
        return f"version_{0}"
    return f"version_{max(existing_versions) + 1 }" 
