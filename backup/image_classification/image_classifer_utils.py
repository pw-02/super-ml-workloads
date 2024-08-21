import json
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from json import JSONEncoder
from typing import Dict, Any
import numpy as np
import psutil
import torch.cuda
from pynvml import (
    nvmlInit,
    nvmlDeviceGetUtilizationRates,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
)
from torch import nn
import math
from typing import  Optional
from enum import Enum
monitor_gpu = False
if torch.cuda.is_available():
    monitor_gpu = True
    nvmlInit()


class Distribution:
    def __init__(self, initial_capacity: int, precision: int = 4):
        self.initial_capacity = initial_capacity
        self._values = np.zeros(shape=initial_capacity, dtype=np.float32)
        self._idx = 0
        self.precision = precision

    def _expand_if_needed(self):
        if self._idx > self._values.size - 1:
            self._values = np.concatenate(
                self._values, np.zeros(self.initial_capacity, dtype=np.float32)
            )

    def add(self, val: float):

        self._expand_if_needed()
        self._values[self._idx] = val
        self._idx += 1

    def summarize(self) -> dict:
        window = self._values[: self._idx]
        if window.size == 0:
            return
        return {
            "n": window.size,
            "mean": round(float(window.mean()), self.precision),
            "min": round(np.percentile(window, 0), self.precision),
            "p50": round(np.percentile(window, 50), self.precision),
            "p75": round(np.percentile(window, 75), self.precision),
            "p90": round(np.percentile(window, 90), self.precision),
            "max": round(np.percentile(window, 100), self.precision),
        }

    def __repr__(self):
        summary_str = json.dumps(self.summarize())
        return "Distribution({0})".format(summary_str)


@dataclass(frozen=True)
class ExperimentResult:
    elapsed_time: float
    volume: float
    utilization: Dict[str, Distribution] = None

    @cached_property
    def throughput(self):
        return self.volume / self.elapsed_time
    
    
class ResourceMonitor:
    """
    Monitors CPU, GPU usage and memory.
    Set sleep_time_s carefully to avoid perf degradations.
    """

    def __init__(
        self, sleep_time_s: float = 0.05, gpu_device: int = 0, chunk_size: int = 25_000
    ):
        self.monitor_thread = None
        self._utilization = defaultdict(lambda: Distribution(chunk_size))
        self.stop_event = threading.Event()
        self.sleep_time_s = sleep_time_s
        self.gpu_device = gpu_device
        self.chunk_size = chunk_size

    def _monitor(self):
        while not self.stop_event.is_set():
            self._utilization["cpu_util"].add(psutil.cpu_percent())
            self._utilization["cpu_mem"].add(psutil.virtual_memory().percent)

            if monitor_gpu:
                gpu_info = nvmlDeviceGetUtilizationRates(
                    nvmlDeviceGetHandleByIndex(self.gpu_device)
                )
                gpu_mem_info = nvmlDeviceGetMemoryInfo(
                    nvmlDeviceGetHandleByIndex(self.gpu_device)
                )
                self._utilization["gpu_util"].add(gpu_info.gpu)
                self._utilization["gpu_mem"].add(
                    gpu_mem_info.used / gpu_mem_info.total * 100
                )
            time.sleep(self.sleep_time_s)

    @property
    def resource_data(self):
        return dict(self._utilization)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self):
        self.monitor_thread = threading.Thread(target=self._monitor)
        self.monitor_thread.start()

    def stop(self):
        self.stop_event.set()
        self.monitor_thread.join()


def num_model_parameters(module: nn.Module, requires_grad: Optional[bool] = None) -> int:
    total = 0
    for p in module.parameters():
        if requires_grad is None or p.requires_grad == requires_grad:
            if hasattr(p, "quant_state"):
                # bitsandbytes 4bit layer support
                total += math.prod(p.quant_state[1])
            else:
                total += p.numel()
    return total

def get_default_supported_precision(training: bool) -> str:
    from lightning.fabric.accelerators import MPSAccelerator
    if MPSAccelerator.is_available() or (torch.cuda.is_available() and not torch.cuda.is_bf16_supported()):
        return "16-mixed" if training else "16-true"
    return "bf16-mixed" if training else "bf16-true"

def test_resource_moinitor()-> ExperimentResult:
    with ResourceMonitor(sleep_time_s=4) as monitor:
        num_samples = 0
        start_time = time.perf_counter()
        for epoch in range(0, 50):
            time.sleep(0.2)
            num_samples += 1
        end_time = time.perf_counter()
        training_time = end_time - start_time
    
    return ExperimentResult(
            elapsed_time=training_time,
            volume=num_samples,
            utilization=monitor.resource_data,
        )

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
            # summary['samples/sec'].append(calculate_average(category_data[key][f'{key}.total_ips']))                
            summary['batches/sec'].append(calculate_average(category_data[key][f'{key}.bps']))
            summary['total_epochs'].append(max(category_data[key][f'{key}.total_epochs']))
            # summary['epochs/sec'].append(calculate_average(category_data[key][f'{key}.total_eps']))                
            summary['loss'].append(calculate_average(category_data[key][f'{key}.loss']))
            summary['acc1'].append(calculate_average(category_data[key][f'{key}.acc1']))
            summary['acc5'].append(calculate_average(category_data[key][f'{key}.acc5']))
            summary['cpu_util'].append(calculate_average(category_data[key][f'{key}.cpu_util']))
            summary['gpu_util'].append(calculate_average(category_data[key][f'{key}.gpu_util']))
    
    return summary

def calculate_average(values):
    if not values:
        raise ValueError("The input list is empty.")

    return sum(values) / len(values)

if __name__ == "__main__":
    result = test_resource_moinitor()
    print(result)