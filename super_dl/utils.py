from torch import nn
from typing import ContextManager, Dict, List, Mapping, Optional, TypeVar, Union
import math
from urllib.parse import urlparse
from enum import Enum
import os
import torch.distributed
import torch.distributed as dist
from lightning.fabric import Fabric
from collections import OrderedDict
import yaml
from lightning.fabric.utilities.rank_zero import rank_zero_only
import time


def chunked_cross_entropy(
    logits: Union[torch.Tensor, List[torch.Tensor]],
    targets: torch.Tensor,
    chunk_size: int = 128,
    ignore_index: int = -1,
) -> torch.Tensor:
    # with large max_sequence_lengths, the beginning of `backward` allocates a large memory chunk which can dominate
    # the memory usage in fine-tuning settings with low number of parameters.
    # as a workaround hack, the cross entropy computation is chunked to force it to deallocate on the go, reducing
    # the memory spike's magnitude

    # lm_head was chunked (we are fine-tuning)
    #if isinstance(logits, list):
        # don't want to chunk cross entropy
        if chunk_size == 0:
            logits = torch.cat(logits, dim=1)
            logits = logits.reshape(-1, logits.size(-1))
            targets = targets.reshape(-1)
            return torch.nn.functional.cross_entropy(logits, targets, ignore_index=ignore_index)

        # chunk cross entropy
        logit_chunks = [logit_chunk.reshape(-1, logit_chunk.size(-1)) for logit_chunk in logits]
        target_chunks = [target_chunk.reshape(-1) for target_chunk in targets.split(logits[0].size(1), dim=1)]
        loss_chunks = [
            torch.nn.functional.cross_entropy(logit_chunk, target_chunk, ignore_index=ignore_index, reduction="none")
            for logit_chunk, target_chunk in zip(logit_chunks, target_chunks)
        ]
        non_masked_elems = (targets != ignore_index).sum()
        return torch.cat(loss_chunks).sum() / max(1, non_masked_elems)

def find_multiple(n: int, k: int) -> int:
    assert k > 0
    if n % k == 0:
        return n
    return n + k - (n % k)


class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0  # noqa
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count  # noqa

    def __str__(self):
        #fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        fmtstr = "{name}:{val" + self.fmt +"}"
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"
    
class MinMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.min = None
        self.n = 0

    def update(self, val, n=1):
        if self.min is None:
            self.min = val
        else:
            self.min = min(self.min, val)
        self.n = n

    def get_val(self):
        return self.min, self.n

    def get_data(self):
        return self.min, self.n   

class MaxMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.max = None
        self.n = 0

    def update(self, val, n=1):
        if self.max is None:
            self.max = val
        else:
            self.max = max(self.max, val)
        self.n = n

    def get_val(self):
        return self.max, self.n

    def get_data(self):
        return self.max, self.n
    

class S3Url(object):
    def __init__(self, url):
        self._parsed = urlparse(url, allow_fragments=False)

    @property
    def bucket(self):
        return self._parsed.netloc

    @property
    def key(self):
        if self._parsed.query:
            return self._parsed.path.lstrip('/') + '?' + self._parsed.query
        else:
            return self._parsed.path.lstrip('/')

    @property
    def url(self):
        return self._parsed.geturl()




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

def get_next_exp_version(root_dir,name):
    from lightning.fabric.utilities.cloud_io import _is_dir, get_filesystem

    versions_root = os.path.join(root_dir, name)
    fs = get_filesystem(root_dir)
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

def to_python_float(t:torch.Tensor)-> float:
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]
    
def reduce_tensor(tensor:torch.Tensor, fabric:Fabric):
    rt = tensor.clone().detach()
    fabric.all_reduce(rt)
    rt /= fabric.world_size()
    return rt

def calc_throughput_per_second(num_sampels, time):
    #world_size = fabric.world_size() 
    world_size = 1
    tbs = world_size * num_sampels
    return tbs / time


import glob
import pandas as pd
import os

#@rank_zero_only
def create_job_report(job_id, folder_path):

    output_file_path = os.path.join(folder_path, f'job_{job_id}_summary_report.xlsx')

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
        df_hparams = pd.DataFrame.from_dict(hparams_data, orient='index')
        df_hparams.to_excel(writer, sheet_name='hparams', header=False, index=True)

        # Iterate through each category
        for category, data_dict in  reversed(category_data.items()):
            # Skip creating sheets if the data for the current category is empty
            if data_dict:
                # Replace invalid characters in the sheet name
                df = pd.DataFrame.from_dict(data_dict, orient='columns')
                for col in ['train.iteration.batch_id', 'val.iteration.batch_id']:
                    if col in df.columns:
                        df[col] = df[col].apply(lambda x: '{:.0f}'.format(x))
                
                df.to_excel(writer, sheet_name=category, index=False)

def summarize_across_all_devices(category_data):
    summary = OrderedDict({
        "dataset_type": [],
        "num_devices": [],
        "total_time": [],
        "data_load_time": [],
        "compute_time": [],      
        "samples_processed": [],
        "samples/sec": [],
        "batches_processed": [],
        "batches/sec": [],
        "epochs_processed": [],
        "epochs/sec": [],
        "loss": [],
        "acc(top1)": [],
        "acc(top5)": []
    })

    keys_to_check = ['train.job', 'val.job']

    for key in keys_to_check:
        if key in category_data:        
            summary['dataset_type'].append('Train' if key == 'train.job' else 'Validation')
            summary['num_devices'].append(len(category_data[key][f'{key}.device']))
            summary['total_time'].append(calculate_average(category_data[key][f'{key}.total_time']))                
            summary['data_load_time'].append(calculate_average(category_data[key][f'{key}.data_time']))
            summary['compute_time'].append(calculate_average(category_data[key][f'{key}.compute_time']))
            summary['samples_processed'].append(sum(category_data[key][f'{key}.num_samples']))
            summary['samples/sec'].append(calculate_average(category_data[key][f'{key}.total_ips']))                
            summary['batches_processed'].append(sum(category_data[key][f'{key}.num_batches']))
            summary['batches/sec'].append(calculate_average(category_data[key][f'{key}.total_bps']))
            summary['epochs_processed'].append(max(category_data[key][f'{key}.num_epochs']))
            summary['epochs/sec'].append(calculate_average(category_data[key][f'{key}.total_eps']))                
            summary['loss'].append(calculate_average(category_data[key][f'{key}.loss(avg)']))
            summary['acc(top1)'].append(calculate_average(category_data[key][f'{key}.top1(avg)']))
            summary['acc(top5)'].append(calculate_average(category_data[key][f'{key}.top5(avg)']))
    
    return summary

def calculate_average(values):
    """
    Calculate the average of a list of values.

    Parameters:
    - values: List of numeric values.

    Returns:
    - Average of the values.
    """
    if not values:
        raise ValueError("The input list is empty.")
    
    return sum(values) / len(values)




if __name__ == "__main__":
    create_job_report(1, '/workspaces/super-dl/mlworkloads/classification/reports/cifar10/version_0')
