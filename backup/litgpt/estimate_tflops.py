# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import math
import sys
import time
from pathlib import Path

import lightning as L

import torch

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from litgpt import Config
from litgpt.model import GPT
from litgpt.utils import  estimate_flops, num_parameters

def estimate_tflops_for_model(model_name,micro_batch_size =5 ):    
    config = Config.from_name(name=model_name)
    model = GPT(config)
    model.apply(model._init_weights)

    with torch.device("meta"):
        meta_model = GPT(model.config)
        # "estimated" is not as precise as "measured". Estimated is optimistic but widely used in the wild.
        # When comparing MFU or FLOP numbers with other projects that use estimated FLOPs,
        # consider passing `flops_per_batch=estimated_flops` instead
        estimated_flops = estimate_flops(meta_model, training=True) * micro_batch_size
        print(f"{model_name} - Total parameters {num_parameters(model):,}, Estimated TFLOPs: {estimated_flops * 1 / 1e12:.2f} ")

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    model_names = [
        'pythia-14m',
        'pythia-31m',
        'pythia-70m',
        'pythia-160m',
        'pythia-410m',
        'pythia-1b',
        'pythia-1.4b',
        'pythia-2.8b',
        'tiny-llama-1.1b',
        'open_llama_3b',
        'dolly-v2-3b',
        'open_llama_3b',
    ]

    for model in model_names:
        estimate_tflops_for_model(model_name=model, micro_batch_size=25)