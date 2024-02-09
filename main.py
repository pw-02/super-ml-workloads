import time
from typing import Iterator
from torch import nn, optim
from torchvision import models, transforms
from jsonargparse._namespace import Namespace
import sys
from torch.utils.data.dataloader import default_collate
from lightning.fabric import Fabric
from super_client import SuperClient
from torch.utils.data import DataLoader
from super_dl.datasets.super_vision_dataset import SUPERVisionDataset
from super_dl.datasets.supet_text_dataset import SUPERTextDataset
import redis
from super_dl.samplers import *
from super_dl.utils import *
from super_dl.logger import *
from classification.train_image_classification import run_vision_training
from language.gpt.train_gpt import run_gpt_training
from super_dl.s3_tasks import S3Helper
import tiktoken
from language.gpt.model import GPT, GPTConfig
def main(fabric: Fabric, hparams: Namespace) -> None:

    exp_start_time = time.time()
    # Prepare for training
    model, optimizer, scheduler, train_dataloader, val_dataloader, logger = prepare_for_training(fabric=fabric, hparams=hparams)
    logger.log_hyperparams(hparams)

    if hparams.workload_type =='vision':
        # Run training
        run_vision_training(fabric,model,optimizer,scheduler,train_dataloader,val_dataloader,hparams=hparams,logger=logger,)
    elif hparams.workload_type =='language':
        run_gpt_training(fabric,model,optimizer,scheduler,train_dataloader,val_dataloader,hparams=hparams,logger=logger,)

    exp_duration = time.time() - exp_start_time
    fabric.print(f"Experiment ended. Duration: {exp_duration}")
    
    fabric.print(f"creating experiment report..")
    file_loaction = create_job_report(hparams.exp_name, logger.log_dir)
    split_path = file_loaction.split('/reports/', 1)
    if len(split_path) > 1:
        trimmed_path = split_path[1]
        S3Helper().upload_to_s3(file_loaction, 'superreports23',trimmed_path)

def prepare_for_training(fabric: Fabric, hparams: Namespace):
    # Set seed
    if hparams.training_seed is not None:
        fabric.seed_everything(hparams.training_seed, workers=True)

    # Load model
    t0 = time.perf_counter()
    model = initialize_model(fabric, hparams.arch,hparams.workload_type,hparams.block_size)
    fabric.print(f"Time to instantiate {hparams.arch} model: {time.perf_counter() - t0:.02f} seconds")
    fabric.print(f"Total parameters in {hparams.arch} model: {num_model_parameters(model):,}")

    # Initialize loss, optimizer, and scheduler
    optimizer = initialize_optimizer(hparams.optimizer, model.parameters(), hparams.lr, hparams.momentum, hparams.weight_decay)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.1 ** (epoch // 30))  # TODO: Add support for other scheduler
    
    # call `setup` to prepare for model / optimizer for distributed training. The model is moved automatically to the right device.
    model, optimizer = fabric.setup(model, optimizer, move_to_device=True)

    # Confirm the dataloader backend and access to super/cache
    verify_dataloader_backend_is_ok(fabric, hparams.dataloader_backend, hparams.cache_adress, hparams.superdl_address)

    # Initialize dataloaders
    eval_dataloader = None
    train_dataloader = None

    if hparams.run_training:
        train_dataset = initialize_dataset(fabric,hparams.job_id, hparams.workload_type, hparams.dataloader_backend, 
                                           hparams.train_data_dir, hparams.block_size, hparams.cache_adress)
        train_sampler = initialize_sampler(hparams.job_id, hparams.dataloader_backend,hparams.workload_type, train_dataset, fabric.world_size, fabric.global_rank,
                                            hparams.shuffle, hparams.batch_size,hparams.drop_last,
                                            hparams.superdl_address,hparams.superdl_prefetch_lookahead)
        
        if hparams.workload_type == 'vision':
            train_dataloader = DataLoader(dataset=train_dataset, sampler=train_sampler, batch_size=None, num_workers=hparams.num_workers)
        elif hparams.workload_type == 'language':
            train_dataloader = DataLoader(dataset=train_dataset, batch_size=hparams.batch_size, num_workers=hparams.num_workers)

        train_dataloader = fabric.setup_dataloaders(train_dataloader, move_to_device=True, use_distributed_sampler=False)


    if hparams.run_evaluate:
        eval_dataset = initialize_dataset(fabric,job_id, hparams.workload_type, hparams.dataloader_backend, 
                                          hparams.eval_data_dir, hparams.block_size, hparams.cache_adress)
        eval_sampler = initialize_sampler(hparams.job_id, hparams.dataloader_backend,hparams.workload_type, eval_dataset, fabric.world_size, fabric.global_rank,
                                            hparams.shuffle, hparams.batch_size,hparams.drop_last,
                                            hparams.superdl_address,hparams.superdl_prefetch_lookahead)
        if hparams.workload_type == 'vision':
            eval_dataloader = DataLoader(dataset=eval_dataset, sampler=eval_sampler, batch_size=None, num_workers=hparams.num_workers)
        elif hparams.workload_type == 'language':
            eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=hparams.batch_size, num_workers=hparams.num_workers)
        eval_dataloader = fabric.setup_dataloaders(eval_dataloader, move_to_device=True, use_distributed_sampler=False)

    # Register job and datasets with super if dataloader_backend is 'super-dl
    if hparams.dataloader_backend == 'superdl':
       register_job_and_datasets_with_superdl(hparams.superdl_address,hparams.job_id,train_dataset, eval_dataset)
    # Initialize logger
    logger = SUPERLogger(fabric, hparams.report_dir, hparams.flush_logs_every_n_steps, hparams.print_freq, hparams.exp_name)

    return model, optimizer, scheduler, train_dataloader, eval_dataloader, logger

def register_job_and_datasets_with_superdl(superdl_address, job_id, train_dataset = None, eval_dataset = None):
    superdl_client = SuperClient(superdl_address)   
    dataset_ids = []

    if train_dataset is not None:
            superdl_client.register_dataset(train_dataset.dataset_id, train_dataset.data_dir, train_dataset.convert_transforms_to_json_dict(), None)
            dataset_ids.append(train_dataset.dataset_id)
    if eval_dataset is not None:
            superdl_client.register_dataset(eval_dataset.dataset_id, eval_dataset.data_dir,eval_dataset.convert_transforms_to_json_dict(), None)
            dataset_ids.append(eval_dataset.dataset_id)
    superdl_client.register_new_job(job_id, dataset_ids)
    del superdl_client



def verify_dataloader_backend_is_ok(fabric: Fabric, dataloader_backend:str, cache_adress:str, super_address:str):
    #fabric.print(f"Checking that data loader backend '{dataloader_backend}' is ok..")

    if cache_adress is not None:
        cache_host, cache_port = cache_adress.split(':')
        fabric.print(f"confirming connection to cache at {cache_host}:{cache_port}..")
        # test connection to the cache, if the test fails, disables the use of cache and SUPER
        cache_client = redis.StrictRedis(host=cache_host, port=cache_port)
        try:
            cache_client.set('foo', 123456)
            cache_client.get('foo')
        except Exception as e:
            fabric.print(f"Failed to connect with cache -'{str(e)}'. Exiting job.")
            sys.exit()

    if dataloader_backend == 'superdl':
        fabric.print(f"confirming connection to super-dl server at '{super_address}'")
        super_client = SuperClient(server_address=super_address)
        connection_confirmed, message = super_client.ping_server()
        if not connection_confirmed:
            fabric.print(f"super-dl connection check failed with '{message}'. Exiting job.")
            sys.exit()
        del super_client
    
    fabric.print(f"Confirmed '{dataloader_backend}' as the data loader backend")



def initialize_model(fabric: Fabric, arch: str, workload_type, block_size) -> nn.Module:
    
    if workload_type == 'vision':
        with fabric.init_module(empty_init=True):  # model is instantiated with randomly initialized weights by default.
            model: nn.Module = models.get_model(arch)
    
    elif workload_type == 'language':
        with fabric.init_module(empty_init=True):
            gptconf = GPTConfig()    
            # n_layer, n_head and n_embd are determined from model_type
            config_args = {
                'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
                'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
                'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
                'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
            }
            gptconf = GPTConfig(**config_args[arch])
            model = GPT(gptconf)
            if block_size < model.config.block_size:
                 model.crop_block_size(block_size)
    return model


def initialize_optimizer(optimizer_type: str, model_parameters: Iterator[nn.Parameter], learning_rate, momentum, weight_decay):
    if optimizer_type == "sgd":
        optimizer = optim.SGD(params=model_parameters,lr=learning_rate,momentum=momentum, weight_decay=weight_decay)
    elif optimizer_type == "rmsprop":
        optimizer = optim.RMSprop(params=model_parameters,lr=learning_rate,momentum=momentum,weight_decay=weight_decay)
    return optimizer


def initialize_sampler(job_id, dataloader_backend,workload_type, dataset,world_size,global_rank, shuffle, batch_size, drop_last, super_address, superdl_prefetch_lookahead):
    if dataloader_backend == "super" and workload_type == 'vision':
        sampler = SuperBatchSampler(dataset, job_id, world_size, global_rank,batch_size, drop_last, shuffle, super_address, superdl_prefetch_lookahead)

    elif dataloader_backend == "classic_pytorch"  and workload_type == 'vision':
        sampler = SuperBatchSampler(dataset, job_id, world_size, global_rank,batch_size, drop_last, shuffle)
    
    elif dataloader_backend == "super" and workload_type == 'language':
        sampler = SUPERSequentialSample(dataset, job_id, world_size, global_rank,super_address, superdl_prefetch_lookahead)
    
    elif dataloader_backend == "classic_pytorch"  and workload_type == 'language':
        sampler = SUPERSequentialSample(dataset, job_id, world_size, global_rank)

    return sampler


def initialize_dataset(fabric:Fabric,job_id, workload_type, dataloader_backend: str, data_dir: str, max_sequence_length, cache_address:str):
    if dataloader_backend == "super" and workload_type == 'vision':
        dataset = SUPERVisionDataset(job_id, data_dir,initialize_transformations(),None, cache_address)  
    elif dataloader_backend == ("classic_pytorch" ) and workload_type == 'vision':
        dataset = SUPERVisionDataset(job_id, data_dir,initialize_transformations(),None, None)
    elif dataloader_backend == "super" and workload_type == 'language':
        dataset = SUPERTextDataset(job_id, data_dir,initialize_tokenizer(),max_sequence_length, cache_address)
    elif dataloader_backend == ("classic_pytorch" ) and workload_type == 'language':
        dataset = SUPERTextDataset(job_id, data_dir,initialize_tokenizer(),max_sequence_length, None)

    fabric.print(f"Dataset initialized: {data_dir}, size: {len(dataset)} files")
    return dataset

def initialize_transformations() -> transforms.Compose:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transformations = transforms.Compose([transforms.ToTensor(), normalize])
    return transformations

def initialize_tokenizer():
    return tiktoken.get_encoding("gpt2")

# def custom_collate(batch):

#     imgs, labels, indices, fetch_times, transform_times = zip(*batch)
#     # Convert images and labels to tensors using default_collate
#     img_tensor = default_collate(imgs)
#     label_tensor = default_collate(labels)

#     total_fetch_time = sum(fetch_times)
#     total_transform_time = sum(transform_times)

#     # Convert other information to tensors if needed
#     batch_id = abs(hash(tuple(indices)))

#     return img_tensor, label_tensor, batch_id, False, total_fetch_time, total_transform_time

# def custom_collate_batch(data):
#     return data



if __name__ == "__main__":
    pass
