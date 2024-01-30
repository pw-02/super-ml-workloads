from jsonargparse import CLI, ArgumentParser, ActionConfigFile
from typing import Any, Optional, Tuple, Union
from lightning.fabric.strategies import FSDPStrategy
from lightning.fabric import Fabric
import torch
from main import main
import os
#import cProfile


def initialize_parser(config_file: str) -> ArgumentParser:
    
    from torchvision.models import list_models

    data_backend_choices = ['super', 'pytorch-vanillia','pytorch-batch']

    parser = ArgumentParser(prog="app", description="Description for my app", default_config_files=[config_file])
    
    #parser.add_argument('--config', action=ActionConfigFile)

    # PyTorch Configuration
    parser.add_argument('--workload.workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('--workload.accelerator', default='gpu', type=str, metavar='N', help='accelerator id to use')
    parser.add_argument('--workload.devices', default=1, type=int, metavar='N', help='number of devices for distributed training')
    parser.add_argument('--workload.epochs', type=int, default=3, help="number of epochs")
    parser.add_argument('--workload.max_minibatches_per_epoch', type=int, default=None)
    parser.add_argument('--workload.run_training', default=True, action="store_true")
    parser.add_argument('--workload.run_evaluate', default=False, action="store_true")
    parser.add_argument('--workload.save_checkpoints', default=False)
    parser.add_argument('--workload.save_dir', type=str, default=None)
    parser.add_argument('--workload.training_seed', type=int, default=None)

    # Model Configuration
    parser.add_argument("--model.arch", type=str, default=None, required=True, choices=list_models(), help="model architecture: {}".format(", ".join(list_models())))
    parser.add_argument("--model.weight_decay", default=1e-4, type=float, help="weight decay factor (default: 1e-4)")
    parser.add_argument("--model.lr", type=float, default=0.1, help="initial learning rate (default: 0.1)")
    parser.add_argument("--model.momentum", default=0.9, type=float, help='momentum (default: 0.9)')
    parser.add_argument('--model.optimizer', default="sgd", choices=['sgd','rmsprop'])
    parser.add_argument('--model.grad_acc_steps', type=int, default=None)


    # Data Configuration
    parser.add_argument('--data.dataloader_backend', default="super", choices=data_backend_choices, help="Choose data backend from {} (default: super-local)".format(", ".join(data_backend_choices)))
    parser.add_argument('--data.train_data_dir', type=str, default=None, required=False)
    parser.add_argument('--data.eval_data_dir', type=str, default=None, required=False)
    parser.add_argument('--data.batch_size', type=int, default=128, help="number of samples per batch")
    parser.add_argument('--data.shuffle', default=True, action="store_true")
    parser.add_argument('--data.drop_last', default=False, action="store_true")
    parser.add_argument('--data.cache_host', type=str, default='localhost') #only used when caching enabled
    parser.add_argument('--data.cache_port', type=int, default=6379) #only used when caching enabled
    parser.add_argument('--data.super_address', type=str, default='localhost:50051')
    parser.add_argument('--data.super_prefetch_lookahead', type=int, default=32)
    parser.add_argument('--data.use_cache', default=False, action="store_true")
    parser.add_argument('--data.sampler_seed', type=int, default=None)

    # Reporting Configuration
    parser.add_argument('--reporting.exp_name', type=str, default=None)
    parser.add_argument('--reporting.profile', default=False, action="store_true")
    parser.add_argument('--reporting.report_dir', type=str, default=None, required=True)
    parser.add_argument('--reporting.flush_logs_every_n_steps', type=int, default=10, help="how many steps to wait before flushing logs")
    parser.add_argument('--reporting.print_freq', type=int, default=1, help="how many steps to wait before flushing logs")

    parser.add_argument("--config", action=ActionConfigFile)  
    return parser


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


def setup(config_file: str, devices: int, precision: Optional[str]) -> None:

    
    parser: ArgumentParser = initialize_parser(config_file)
    hparams = parser.parse_args(["--config", config_file])

    precision = precision or get_default_supported_precision(training=True)

    strategy = FSDPStrategy(state_dict_type="full", limit_all_gathers=True, cpu_offload=False) if devices > 1 else "auto"

    fabric = Fabric(accelerator=hparams.workload.accelerator, devices=hparams.workload.devices, strategy=strategy, precision=precision)
    
    if hparams.workload.max_minibatches_per_epoch:
        hparams.workload.max_minibatches_per_epoch //= fabric.world_size
    hparams.job_id = os.getpid()
    #fabric.print(hparams)
    fabric.launch(main, hparams=hparams)

    

if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")

    defaults = {
        "config_file": 'exps/exp1/resnet_resnet18_super.yaml',
        'devices': 1,
        'precision': None,
    }
    
    from jsonargparse import CLI
    CLI(setup, set_defaults=defaults)
