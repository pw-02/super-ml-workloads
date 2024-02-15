from jsonargparse import CLI, ArgumentParser, ActionConfigFile
from typing import Any, Optional, Tuple, Union
from lightning.fabric.strategies import FSDPStrategy
from lightning.fabric import Fabric
import torch
from hpo_main import main
import os
from functools import partial
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from ray import tune
from ray.air import session
from ray.tune.schedulers import ASHAScheduler

#import cProfile

def get_next_exp_version(root_dir, name):
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

def initialize_parser(config_file: str) -> ArgumentParser:
    
    from torchvision.models import list_models
    data_backend_choices = ['superdl', 'classic_pytorch']
    gpt_model_choices = ['gpt2','gpt2-medium','gpt2-large','gpt2-xl']
    parser = ArgumentParser(prog="app", description="Description for my app", default_config_files=[config_file])
    
    #parser.add_argument('--config', action=ActionConfigFile)

    # PyTorch Configuration
    parser.add_argument('--num_workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('--accelerator', default='gpu', type=str, metavar='N', help='accelerator id to use')
    parser.add_argument('--devices', default=1, type=int, metavar='N', help='number of devices for distributed training')
    parser.add_argument('--training_seed', type=int, default=None)
    parser.add_argument('--workload_type', type=str, default='vision', required=True)

    parser.add_argument("--arch", type=str, default=None, required=True, choices=list_models() + gpt_model_choices, help="model architecture: {}".format(", ".join(list_models())))
    parser.add_argument("--weight_decay", default=1e-4, type=float, help="weight decay factor (default: 1e-4)")
    parser.add_argument("--lr", type=float, default=0.1, help="initial learning rate (default: 0.1)")
    parser.add_argument("--momentum", default=0.9, type=float, help='momentum (default: 0.9)')
    parser.add_argument('--optimizer', default="sgd", choices=['sgd','rmsprop'])
    parser.add_argument('--grad_acc_steps', type=int, default=None)

    # Data Configuration
    parser.add_argument('--dataloader_backend', default="super", choices=data_backend_choices, help="Choose data backend from {} (default: super-local)".format(", ".join(data_backend_choices)))
    parser.add_argument('--train_data_dir', type=str, default=None, required=False)
    parser.add_argument('--eval_data_dir', type=str, default=None, required=False)
    parser.add_argument('--epochs', type=int, default=3, help="number of epochs")
    parser.add_argument('--block_size', type=int, default=1024, help="number of blocks per batch")
    parser.add_argument('--batch_size', type=int, default=128, help="number of samples per batch")
    parser.add_argument('--max_minibatches_per_epoch', type=int, default=None)
    parser.add_argument('--shuffle', default=True, action="store_true")
    parser.add_argument('--drop_last', default=False, action="store_true")

    # #expiement/logging
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--print_freq', type=int, default=1, help="how many steps to wait before flushing logs")
    parser.add_argument('--flush_logs_every_n_steps', type=int, default=10, help="how many steps to wait before flushing logs")
    parser.add_argument('--run_training', default=True, action="store_true")
    parser.add_argument('--run_evaluate', default=False, action="store_true")
    parser.add_argument('--save_checkpoints', default=False)
    parser.add_argument('--save_checkpoint_dir', type=str, default=None)
    parser.add_argument('--report_dir', type=str, default=None, required=True)
    parser.add_argument('--profile', default=False, action="store_true")
    
    #SUPER/Caching Configuration
    parser.add_argument('--cache_adress', type=str, default='localhost:6379')
    parser.add_argument('--superdl_address', type=str, default='localhost:50051')
    parser.add_argument('--superdl_prefetch_lookahead', type=int, default=32)
    parser.add_argument("--config", action=ActionConfigFile)  
    return parser


def get_default_supported_precision(training: bool) -> str:
    from lightning.fabric.accelerators import MPSAccelerator
    if MPSAccelerator.is_available() or (torch.cuda.is_available() and not torch.cuda.is_bf16_supported()):
        return "16-mixed" if training else "16-true"
    return "bf16-mixed" if training else "bf16-true"

def setup(config_file: str, devices: int, precision: Optional[str]) -> None:
    parser: ArgumentParser = initialize_parser(config_file)
    hparams = parser.parse_args(["--config", config_file])
    precision = precision or get_default_supported_precision(training=True)
    strategy = FSDPStrategy(state_dict_type="full", limit_all_gathers=True, cpu_offload=False) if devices > 1 else "auto"
    fabric = Fabric(accelerator=hparams.accelerator, devices=hparams.devices, strategy=strategy, precision=precision)

    if hparams.max_minibatches_per_epoch:
        hparams.max_minibatches_per_epoch //= fabric.world_size
    hparams.job_id = os.getpid()
    hparams.exp_version = get_next_exp_version(hparams.report_dir,hparams.exp_name)
    fabric.print(hparams)
    #main(None,fabric=fabric, hparams=hparams)
    config = {
            "l1": tune.choice([2**i for i in range(9)]),
            "l2": tune.choice([2**i for i in range(9)]),
            "lr": tune.loguniform(1e-4, 1e-1),
            "batch_size": tune.choice([256]),
            }
    
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=1,
        grace_period=1,
        reduction_factor=2,
    )
    result = tune.run(
        partial(main, fabric=fabric, hparams=hparams),
        resources_per_trial={"cpu": 4, "gpu": 1},
        config=config,
        max_concurrent_trials=1,
        num_samples=2,
        scheduler=scheduler,
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")
        


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")

    defaults = {
       "config_file": 'configs/hpo-example-cfg.yaml',
        'devices': 1,
        'precision': None,
        }
      
    from jsonargparse import CLI
    CLI(setup, set_defaults=defaults)
