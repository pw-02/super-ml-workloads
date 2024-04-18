
from omegaconf import DictConfig
import hydra
from mlworklaods.args import *
import os
from train_image_classifer import launch_job
from mlworklaods.log_utils import  get_next_exp_version
import torch.multiprocessing as mp 
from torch.multiprocessing import Pool, Process, set_start_method 
from typing import List
import torch

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig):
    
    exp_version = get_next_exp_version(config.log_dir,config.dataset.name)

    train_args: TrainArgs = TrainArgs(
        job_id=os.getpid(),
        model_name = config.training.model_name,
        epochs = config.training.epochs,
        global_batch_size=config.training.batch_size,
        global_epoch_max_iters = config.training.iters_per_epoch,
        learning_rate = config.training.learning_rate,
        weight_decay = config.training.weight_decay,
        num_pytorch_workers = config.training.num_pytorch_workers,
        shuffle= config.dataloader.shuffle,
        run_training = config.run_training,
        run_evaluation = config.run_evaluation,
        dataload_only = config.dataload_only,
        devices = config.num_devices_per_job,
        accelerator = config.accelerator,
        seed = config.seed
        )
     
    io_args: IOArgs = IOArgs(
        dataloader_kind= config.dataloader.kind,
        train_data_dir=config.dataset.train_dir,
        val_data_dir=config.dataset.val_dir,
        log_dir = os.path.join(config.log_dir, config.dataset.name, str(exp_version)),
        log_interval = config.log_interval
        )
    
    if 'super' in io_args.dataloader_kind:
        io_args.super_address=config.dataloader.super_address,
        io_args.cache_address=config.dataloader.cache_address,
        train_args.simulate_data_delay = config.dataloader.simulate_data_delay,

    elif 'shade' in io_args.dataloader_kind:
        # #Initialization of local cache, PQ and ghost cache
        # red_local = redis.Redis()
        # PQ = heapdict.heapdict()
        # ghost_cache = heapdict.heapdict()
        # key_counter  = 0
        io_args.working_set_size = config.dataloader.working_set_size
        io_args.replication_factor =config.dataloader.replication_factor
        io_args.cache_address=config.dataloader.cache_address

    if config.num_jobs == 1:
        train_args.devices = config.num_devices_per_job
        print(f"Running single job on {train_args.devices} GPUS")
        launch_job(config, train_args, io_args)
    else:
        print(f"Running HP with {config.num_jobs} jobs each on {train_args.devices} GPUS")
        processes:List[Process] = []
        lr_array = [0.01, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017]
        
        total_gpu = torch.cuda.device_count() 
        num_procs = config.num_jobs * train_args.devices
        
        # if num_procs > total_gpu:
        #     raise Exception("error: Cannot run HP - more GPU requested than available") 
        
        for job in range(0, config.num_jobs):
            #The start ID of GPU for this multi-GPU HP job
            # Eg, if we start 4 2-GPU jobs, the IDs will be [0, 2, 4, 6]
            start_gpu_id = job*config.num_devices_per_job
            end_gpu_id = start_gpu_id + config.num_devices_per_job - 1                  
            #Change the port for each job - they are independent
            #print("Before creating process - port is {} for job:{}".format(args.master_port, job))
            processes.append(Process(target=spawn_jobs, args=(job, config, train_args, io_args, start_gpu_id,  end_gpu_id, lr_array[job])))

        for process in processes:
            process.start()

        for process in processes:
            process.join()

def spawn_jobs(job_id, config:DictConfig, train_args:TrainArgs, io_args:IOArgs, start_id, end_id, lr):
    train_args.job_id = f"{train_args.job_id}_{job_id}"
    train_args.learning_rate = lr
    io_args.log_dir =  f"{io_args.log_dir}_HPO"
    io_args.log_dir  = os.path.join(io_args.log_dir, f"job_{job_id}")
    # train_args.devices = start_id 
    if train_args.accelerator == 'cpu':
        train_args.devices = config.num_devices_per_job 
    else:
        train_args.devices = start_id
    #[start_id, end_id]
    # master_port = args.master_port + job_id
    print("Spawn a {} GPU job starting at GPU#{}".format(config.num_devices_per_job, start_id))

    mp.spawn(launch_job, nprocs=1, args=(config, train_args, io_args))


    
if __name__ == "__main__":
    main()