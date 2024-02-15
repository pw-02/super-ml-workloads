from jsonargparse import CLI, ArgumentParser, ActionConfigFile
from typing import Any, Optional, Tuple, Union
from lightning.fabric.strategies import FSDPStrategy
from lightning.fabric import Fabric
import torch
from hpo_main import main
import os
from ray.air import session
from super_dl.utils import create_job_report
from super_dl.s3_tasks import S3Helper

#import cProfile

def initialize_parser(config_file: str) -> ArgumentParser:
    
    from torchvision.models import list_models
    print(config_file)
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

def launch_job(hpo_config:dict=None, job_config_file:str = None) -> None:
    precision = get_default_supported_precision(training=True)    
    parser: ArgumentParser = initialize_parser(job_config_file)
    hparams = parser.parse_args(["--config", job_config_file])

    if hpo_config is not None:
          #update the hparams here
         pass
       
    strategy = FSDPStrategy(state_dict_type="full", limit_all_gathers=True, cpu_offload=False) if hparams.devices > 1 else "auto"
    fabric = Fabric(accelerator=hparams.accelerator, devices=hparams.devices, strategy=strategy, precision=precision)
    
    if hparams.max_minibatches_per_epoch:
        hparams.max_minibatches_per_epoch //= fabric.world_size
    hparams.job_id = os.getpid()
    hparams.exp_version = get_next_exp_version(hparams.report_dir,hparams.exp_name)
    fabric.print(hparams)
    avg_loss, avg_acc, log_dir = fabric.launch(main, hparams=hparams)
    session.report({"loss": avg_loss, "accuracy": avg_acc})

    # if fabric.is_global_zero:
    #     logger.log_hyperparams(hparams)

    # exp_duration = time.time() - exp_start_time
    # fabric.print(f"Experiment ended. Duration: {exp_duration}")
    
    if fabric.is_global_zero:
        fabric.print(f"creating experiment report..")
        file_loaction = create_job_report(hparams.exp_name, log_dir)
        split_path = file_loaction.split('/reports/', 1)
        if len(split_path) > 1:
            trimmed_path = split_path[1]
            #S3Helper().upload_to_s3(file_loaction, 'superreports23',trimmed_path)
        else:
            pass #S3Helper().upload_to_s3(file_loaction, 'superreports23',file_loaction)

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

if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")
    config_file = '/workspaces/super-ml-workloads/configs/hpo-base-example-cfg.yaml'
    defaults = {
         "job_config_file": config_file}
    
    from jsonargparse import CLI
    CLI(launch_job, set_defaults=defaults)
    # launch_job(job_config_file=config_file)
