
from omegaconf import DictConfig
import hydra
from mlworklaods.args import *
from train_image_classifer_super import run_super_job
from train_image_classifer_shade import run_shade_job
from train_image_classifer_lru_torch import run_lru_torch_job
from mlworklaods.log_utils import  get_next_exp_version
import torch.multiprocessing as mp 
from torch.multiprocessing import Pool, Process, set_start_method 
from typing import List
from typing import Dict, Any
import os

# Helper function to prepare arguments for a job
def prepare_args(config: DictConfig):
    log_dir_base = f"{config.log_dir}/{config.dataset.name}/{config.training.model_name}"
    exp_version = get_next_exp_version(log_dir_base, config.dataloader.kind)
    full_log_dir = os.path.join(log_dir_base, config.dataloader.kind, str(exp_version))

    train_args = TrainArgs(
        job_id=os.getpid(),
        model_name=config.training.model_name,
        dataload_only=config.dataload_only,
        num_pytorch_workers=config.training.num_pytorch_workers,
        epochs=config.training.epochs,
        batch_size=config.training.batch_size,
        epoch_max_iters=config.training.iters_per_epoch,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        run_training=config.run_training,
        run_evaluation=config.run_evaluation,
        devices=config.num_devices_per_job,
        accelerator=config.accelerator,
        seed=config.seed,
        log_dir=full_log_dir,
        log_interval=config.log_interval,
        dataloader_kind=config.dataloader.kind
    )

    data_args = DataArgs(
        train_data_dir=config.dataset.train_dir,
        val_data_dir=config.dataset.val_dir,
    )

    if 'super' in train_args.dataloader_kind:
        super_args = SUPERArgs(
            super_address=config.dataloader.super_address,
            cache_address=config.dataloader.cache_address,
            simulate_data_delay=config.dataloader.simulate_data_delay)
        if super_args.simulate_data_delay is not None:
            train_args.dataload_only = True
        return train_args, data_args, super_args

    elif 'shade' in train_args.dataloader_kind:
        shade_args = SHADEArgs(
            cache_address=config.dataloader.cache_address,
            working_set_size=config.dataloader.working_set_size,
            replication_factor=config.dataloader.replication_factor)
        return train_args, data_args, shade_args

    elif 'torch_lru' in train_args.dataloader_kind:
        torchlru_args = LRUTorchArgs(
            cache_address=config.dataloader.cache_address,
            cache_granularity=config.dataloader.cache_granularity,
            shuffle=config.dataloader.shuffle)
        
        return train_args, data_args, torchlru_args

        


# Function to spawn multiple jobs
def spawn_multiple_jobs(config, train_args, data_args, dataloader_args):
    num_jobs = config.num_jobs
    processes = []
    lr_array = [0.01, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017]

    # for job in range(num_jobs):
    #     start_gpu_id = job * config.num_devices_per_job
    #     end_gpu_id = start_gpu_id + config.num_devices_per_job - 1
    for job in range(num_jobs):
        start_gpu_id = job
        end_gpu_id = job
        
        
        # Create a dictionary of process-specific arguments
        process_args = {
            "job_id": job,
            "config": config,
            "train_args": train_args,
            "data_args": data_args,
            "dataloader_args": dataloader_args,
            "start_gpu_id": start_gpu_id,
            "end_gpu_id": end_gpu_id,
            "lr": lr_array[job],
        }
        processes.append(Process(target=spawn_jobs, args=(process_args,)))
        # processes.append(Process(target=spawn_jobs, args=(job, config, train_args, io_args, start_gpu_id,  end_gpu_id, lr_array[job])))

    # Start all processes
    for process in processes:
        process.start()

    # Join all processes
    for process in processes:
        process.join()

# Function to spawn a single job
def spawn_jobs(process_args: Dict[str, Any]):
    job_id = process_args["job_id"]
    config = process_args["config"]
    train_args:TrainArgs = process_args["train_args"]
    data_args:DataArgs = process_args["data_args"]
    dataloader_args = process_args["dataloader_args"]
    start_gpu_id = process_args["start_gpu_id"]
    end_gpu_id = process_args["end_gpu_id"]
    lr = process_args["lr"]

    train_args.job_id = f"{train_args.job_id}_{job_id}"
    train_args.learning_rate = lr
    train_args.log_dir = f"{train_args.log_dir}_HPO/job_{job_id}"

    if train_args.accelerator == 'cpu':
        train_args.devices = config.num_devices_per_job
    else:
        # train_args.devices = [start_gpu_id, end_gpu_id]
        train_args.devices = start_gpu_id
    print(f"Spawning a {config.num_devices_per_job} GPU job starting at GPU#{start_gpu_id}")
    
    if 'super' in train_args.dataloader_kind:
        mp.spawn(run_super_job, nprocs=1, args=(config, train_args, data_args, dataloader_args))
    elif 'shade' in train_args.dataloader_kind:
        mp.spawn(run_shade_job, nprocs=1, args=(config, train_args, data_args, dataloader_args))
    elif 'torch_lru' in train_args.dataloader_kind:
        mp.spawn(run_lru_torch_job, nprocs=1, args=(config, train_args, data_args, dataloader_args))
    else:
        raise Exception(f"unknown dataloader_kind {train_args.dataloader_kind}")

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig):
    train_args, data_args, dataloader_args = prepare_args(config)

    if config.num_jobs == 1:
        
        print(f"Running a single job on {train_args.devices} GPUs")

        if 'super' in train_args.dataloader_kind:
            run_super_job(1, config, train_args, data_args,dataloader_args)
        elif 'shade' in train_args.dataloader_kind:
            run_shade_job(1, config, train_args, data_args,dataloader_args)
        elif 'torch_lru' in train_args.dataloader_kind:
            run_lru_torch_job(1, config, train_args, data_args,dataloader_args)
        else:
            raise Exception(f"unknown dataloader_kind {train_args.dataloader_kind}")
    else:
        print(f"Running HP with {config.num_jobs} jobs, each on {train_args.devices} GPUs")
        spawn_multiple_jobs(config, train_args, data_args, dataloader_args)

if __name__ == "__main__":
    main()