import json
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
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

# @dataclass(frozen=True)
# class ExperimentResult:
#     elapsed_time: float
#     volume: float
#     utilization: Dict[str, Distribution] = None

#     @cached_property
#     def throughput(self):
#         return self.volume / self.elapsed_time
    
    
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
        self.monitor_gpu = monitor_gpu

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


def calculate_average(values):
    if not values:
        raise ValueError("The input list is empty.")
    return sum(values) / len(values)


def calc_throughput(count, time):
        if time >0 and count > 0:
            return count/time
        else:
            return 0

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


if __name__ == "__main__":
    pass
