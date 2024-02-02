import time
from typing import Iterator
from torch import nn, optim
from torchvision import models, transforms
from jsonargparse._namespace import Namespace
import sys
from torch.utils.data.dataloader import default_collate
from lightning.fabric import Fabric
from super_client import SuperClient
from image_classification.dataloader import DataLoader
from image_classification.utils import *
from image_classification.datasets import *
from image_classification.samplers import *
from image_classification.training import *


def main(fabric: Fabric, hparams: Namespace) -> None:
    exp_start_time = time.time()

    # Prepare for training
    model, optimizer, scheduler, train_dataloader, val_dataloader, logger, super_client = prepare_for_training(
        fabric=fabric, hparams=hparams)

    logger.log_hyperparams(hparams)

    # Run training
    run_training(
        fabric=fabric,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        hparams=hparams,
        logger=logger,
        super_client=super_client
    )

    exp_duration = time.time() - exp_start_time

    create_job_report(hparams.reporting.exp_name, logger.log_dir)

    fabric.print(f"Experiment ended. Duration: {exp_duration}")


def custom_collate(batch):
    imgs, labels, indices, fetch_times, transform_times = zip(*batch)

    # Convert images and labels to tensors using default_collate
    img_tensor = default_collate(imgs)
    label_tensor = default_collate(labels)

    total_fetch_time = sum(fetch_times)
    total_transform_time = sum(transform_times)

    # Convert other information to tensors if needed
    batch_id = abs(hash(tuple(indices)))

    return img_tensor, label_tensor, batch_id, False, total_fetch_time, total_transform_time


def custom_collate_batch(data):
    return data


def prepare_for_training(fabric: Fabric, hparams: Namespace):
    # Set seed
    if hparams.workload.training_seed is not None:
        fabric.seed_everything(hparams.workload.training_seed, workers=True)

    # Load model
    t0 = time.perf_counter()
    model = initialize_model(fabric, hparams.model.arch)
    fabric.print(f"Time to instantiate {hparams.model.arch} model: {time.perf_counter() - t0:.02f} seconds")
    fabric.print(f"Total parameters in {hparams.model.arch} model: {num_model_parameters(model):,}")

    # Initialize loss, optimizer, and scheduler
    optimizer = initialize_optimizer(optimizer_type=hparams.model.optimizer, model_parameters=model.parameters(),
                                     learning_rate=hparams.model.lr, momentum=hparams.model.momentum,
                                     weight_decay=hparams.model.weight_decay)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.1 ** (epoch // 30))  # TODO: Add support for other scheduler
    # call `setup` to prepare for model / optimizer for distributed training. The model is moved automatically to the right device.
    model, optimizer = fabric.setup(model, optimizer, move_to_device=True)

    # Confirm the dataloader backend
    fabric.print(f"Attempting to use '{hparams.data.dataloader_backend}' as the data loader backend")

    confirm_dataloader_backend(fabric=fabric, use_cache=hparams.data.use_cache,
                                              cache_host=hparams.data.cache_host if hparams.data.use_cache else None,
                                              cache_port=hparams.data.cache_port if hparams.data.use_cache else None,
                                              dataloader_backend=hparams.data.dataloader_backend,
                                              super_address=hparams.data.super_address if hparams.data.dataloader_backend == 'super' else None)

    fabric.print(f"Confirmed '{hparams.data.dataloader_backend}' as the data loader backend")

    # Initialize dataloaders
    eval_dataloader = None
    train_dataloader = None

    if hparams.workload.run_training:
        train_dataloader = initialize_dataloader(
            job_id=hparams.job_id,
            fabric=fabric,
            num_workers=hparams.workload.workers,
            dataloader_backend=hparams.data.dataloader_backend,
            data_dir=hparams.data.train_data_dir,
            shuffle=hparams.data.shuffle,
            sampler_seed=hparams.data.sampler_seed,
            batch_size=hparams.data.batch_size,
            drop_last=hparams.data.drop_last,
            super_address=hparams.data.super_address,
            super_prefetch_lookahead=hparams.data.super_prefetch_lookahead,
            cache_host=hparams.data.cache_host if hparams.data.use_cache else None,
            cache_port=hparams.data.cache_port if hparams.data.use_cache else None
        )
        train_dataloader = fabric.setup_dataloaders(train_dataloader, move_to_device=True)

    if hparams.workload.run_evaluate:
        train_dataloader = initialize_dataloader(
            job_id=hparams.job_id,
            fabric=fabric,
            num_workers=hparams.workload.workers,
            dataloader_backend=hparams.data.dataloader_backend,
            data_dir=hparams.data.eval_data_dir,
            shuffle=False,
            sampler_seed=hparams.data.sampler_seed,
            batch_size=hparams.data.batch_size,
            drop_last=hparams.data.drop_last,
            super_address=hparams.data.super_address if hparams.data.dataloader_backend == 'super' else None,
            super_prefetch_lookahead=hparams.data.super_prefetch_lookahead if hparams.data.dataloader_backend == 'super' else None,
            cache_host=hparams.data.cache_host if hparams.data.use_cache else None,
            cache_port=hparams.data.cache_port if hparams.data.use_cache else None
        )
        eval_dataloader = fabric.setup_dataloaders(eval_dataloader, move_to_device=True)

    # Register job and datasets with super if it's the dataloader_backend
    if hparams.data.dataloader_backend == 'super':
        super_client = SuperClient(server_address=hparams.data.super_address)

        job_dataset_ids = []
        if train_dataloader is not None:
            train_dataset: SUPERDataset = train_dataloader.dataset
            super_client.register_dataset(train_dataset.dataset_id, train_dataset.data_dir,
                                          json.dumps(train_dataset.transform_to_dict()), None)
            job_dataset_ids.append(train_dataset.dataset_id)

        if eval_dataloader is not None:
            eval_dataset: SUPERDataset = eval_dataloader.dataset
            super_client.register_dataset(eval_dataset.dataset_id, eval_dataset.data_dir,
                                          json.dumps(eval_dataset.transform_to_dict()), None)
            job_dataset_ids.append(eval_dataset.dataset_id)

        super_client.register_new_job(job_id=hparams.job_id, job_dataset_ids=job_dataset_ids)
        del super_client

    # Initialize logger
    logger = SUPERLogger(fabric=fabric, root_dir=hparams.reporting.report_dir,
                         flush_logs_every_n_steps=hparams.reporting.flush_logs_every_n_steps,
                         print_freq=hparams.reporting.print_freq,
                         exp_name=hparams.reporting.exp_name)

    return model, optimizer, scheduler, train_dataloader, eval_dataloader, logger, None


def confirm_dataloader_backend(fabric: Fabric, use_cache, cache_host, cache_port, dataloader_backend, super_address):

    if use_cache == True:
        fabric.print(f"confirming connection to cache at {cache_host}:{cache_port}..")
        # test connection to the cache, if the test fails, disables the use of cache and SUPER
        cache_client = redis.StrictRedis(host=cache_host, port=cache_port)
        try:
            cache_client.set('foo', 123456)
            cache_client.get('foo')
        except Exception as e:
            use_cache = False
            dataloader_backend = 'pytorch-batch'
            fabric.print(f"Failed to connect with cache -'{str(e)}'. Exiting job.")
            sys.exit()

    if dataloader_backend == 'super':
        fabric.print(f"Confirming connection to super at '{super_address}'")
        super_client = SuperClient(server_address=super_address)
        connection_confirmed, message = super_client.ping_server()
        if not connection_confirmed:
            fabric.print(f"super connection check failed with '{message}'. Exiting job.")
            sys.exit()
        del super_client
 


def initialize_model(fabric: Fabric, arch: str) -> nn.Module:
    with fabric.init_module(empty_init=True):  # model is instantiated with randomly initialized weights by default.
        model: nn.Module = models.get_model(arch)
    return model


def initialize_optimizer(optimizer_type: str, model_parameters: Iterator[nn.Parameter], learning_rate, momentum,
                         weight_decay):
    if optimizer_type == "sgd":
        optimizer = optim.SGD(params=model_parameters,
                              lr=learning_rate,
                              momentum=momentum,
                              weight_decay=weight_decay)
    elif optimizer_type == "rmsprop":
        optimizer = optim.RMSprop(params=model_parameters,
                                  lr=learning_rate,
                                  momentum=momentum,
                                  weight_decay=weight_decay)
    return optimizer


def initialize_dataloader(job_id, fabric: Fabric, num_workers, dataloader_backend, data_dir, shuffle, sampler_seed,
                          batch_size, drop_last, super_address, super_prefetch_lookahead, cache_host,
                          cache_port):
    dataset = initialize_dataset(job_id=job_id, dataloader_backend=dataloader_backend,
                                 transformations=initialize_transformations(), data_dir=data_dir,
                                 cache_host=cache_host, cache_port=cache_port)

    fabric.print(f"Dataset initialized: {data_dir}, size: {len(dataset)} files")

    sampler = initialize_sampler(job_id=job_id, dataset=dataset, dataloader_backend=dataloader_backend,
                                  shuffle=shuffle, batch_size=batch_size, drop_last=drop_last,
                                  super_prefetch_lookahead=super_prefetch_lookahead,
                                  sampler_seed=sampler_seed, super_address=super_address)

    if dataloader_backend == "pytorch-vanilla":
        return DataLoader(dataset=dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers,
                          collate_fn=custom_collate)
    else:
        return DataLoader(dataset=dataset, sampler=sampler, batch_size=None, num_workers=num_workers)


def initialize_sampler(job_id, dataset, dataloader_backend, shuffle, sampler_seed, batch_size, drop_last,
                        super_prefetch_lookahead=None, super_address=None):
    if dataloader_backend == "super":
        return SUPERSampler(job_id=job_id, data_source=dataset, batch_size=batch_size, drop_last=drop_last,
                            shuffle=shuffle, seed=sampler_seed,
                            super_prefetch_lookahead=super_prefetch_lookahead,
                            super_address=super_address)

    elif dataloader_backend == "pytorch-batch":
        return PytorchBatchSampler(data_source=dataset, batch_size=batch_size, drop_last=drop_last, shuffle=shuffle,
                                   seed=sampler_seed)

    elif dataloader_backend == "pytorch-vanilla":
        return PytorchVanillaSampler(data_source=dataset, shuffle=shuffle, seed=sampler_seed)


def initialize_dataset(job_id, dataloader_backend: str, transformations: transforms.Compose, data_dir: str,
                       cache_host=None, cache_port=None):
    if dataloader_backend == "super":
        return SUPERDataset(job_id=job_id, data_dir=data_dir, transform=transformations, cache_host=cache_host,
                            cache_port=cache_port)
    elif dataloader_backend == "pytorch-batch":
        return PytorchBatchDataset(job_id=job_id, data_dir=data_dir, transform=transformations)
    elif dataloader_backend == "pytorch-vanilla":
        return PytorchVanillaDataset(job_id=job_id, data_dir=data_dir, transform=transformations)


def initialize_transformations() -> transforms.Compose:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transformations = transforms.Compose([transforms.ToTensor(), normalize])
    return transformations


if __name__ == "__main__":
    pass
