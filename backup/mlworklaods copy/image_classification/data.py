from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torchvision import transforms
from mlworklaods.dataloaders.baseline.batch_sampler_with_id import BatchSamplerWithID
from mlworklaods.dataloaders.baseline.random_batch_sampler_with_id import RandomBatchSamplerWithID
from mlworklaods.dataloaders.shade.shadedataset_s3 import ShadeDatasetS3
from mlworklaods.dataloaders.shade.shadesampler_s3 import ShadeSamplerS3, ShadeBatchSampler
from mlworklaods.dataloaders.baseline.baseline_mapped_dataset import BaselineMappeedDataset
from mlworklaods.dataloaders.super_dl.super_dataset import SUPERDataset
from typing import Tuple, Optional
from mlworklaods.args import * 
import torch

class BaseDataModule:
    def __init__(self, transform, num_classes: int):
        self.transform = transform
        self.num_classes = num_classes

    def make_dataloaders(self, train_args: ImgClassifierTrainArgs, data_args: DataArgs, dataloader_args, world_size: int):

        if isinstance(dataloader_args, SUPERArgs):
            return self.make_super_dataloaders(train_args, data_args, dataloader_args, world_size)
        elif isinstance(dataloader_args, LRUTorchArgs):
            return self.make_lru_torch_dataloaders(train_args, data_args, dataloader_args, world_size)
        elif isinstance(dataloader_args, SHADEArgs):
            return self.make_shde_dataloaders(train_args, data_args, dataloader_args, world_size)
        else:
            raise Exception(f"Unknown dataloader_kind {train_args.dataloader_kind}")

    def make_lru_torch_dataloaders(self, train_args: BaseTrainArgs, data_args: DataArgs, lru_torch_args: LRUTorchArgs, world_size: int) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
        train_dataloader = self.create_lru_dataloader(train_args, data_args.train_data_dir, lru_torch_args, world_size) if train_args.run_training else None
        val_dataloader = self.create_lru_dataloader(train_args, data_args.val_data_dir, lru_torch_args, world_size) if train_args.run_evaluation else None
        return train_dataloader, val_dataloader

    def create_lru_dataloader(self, train_args: BaseTrainArgs, data_dir: str, lru_torch_args: LRUTorchArgs, world_size: int) -> DataLoader:
        dataset = BaselineMappeedDataset(
            data_dir=data_dir,
            transform=self.transform,
            cache_address=lru_torch_args.cache_address,
            cache_granularity=lru_torch_args.cache_granularity
        )
        base_sampler = RandomSampler(data_source=dataset, generator=torch.Generator().manual_seed(train_args.seed)) if lru_torch_args.shuffle else SequentialSampler(data_source=dataset)
        batch_sampler = BatchSamplerWithID(sampler=base_sampler, batch_size=train_args.batch_size(world_size), drop_last=False)
        return DataLoader(dataset=dataset, sampler=batch_sampler, batch_size=None, num_workers=lru_torch_args.num_pytorch_workers)

    def make_super_dataloaders(self, train_args: BaseTrainArgs, data_args: DataArgs, super_args: SUPERArgs, world_size: int) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
        train_dataloader = self.create_super_dataloader(train_args, data_args.train_data_dir, super_args, world_size) if train_args.run_training else None
        val_dataloader = self.create_super_dataloader(train_args, data_args.val_data_dir, super_args, world_size) if train_args.run_evaluation else None
        return train_dataloader, val_dataloader

    def create_super_dataloader(self, train_args: BaseTrainArgs, data_dir: str, super_args: SUPERArgs, world_size: int) -> DataLoader:
        dataset = SUPERDataset(
            job_id=train_args.job_id,
            data_dir=data_dir,
            batch_size=train_args.batch_size(world_size),
            transform=self.transform,
            world_size=world_size,
            super_address=super_args.super_address,
            cache_address=super_args.cache_address,
            simulate_delay=super_args.simulate_data_delay
        )
        return DataLoader(dataset=dataset, batch_size=None, num_workers=super_args.num_pytorch_workers)
    
    def make_shde_dataloaders(self, train_args: BaseTrainArgs, data_args: DataArgs, shade_args: SHADEArgs, world_size: int) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:

        train_dataloader = self.create_shade_dataloader(train_args, data_args.train_data_dir, shade_args, world_size) if train_args.run_training else None
        val_dataloader = self.create_shade_dataloader(train_args, data_args.val_data_dir, shade_args, world_size) if train_args.run_evaluation else None
        return train_dataloader, val_dataloader

    def create_shade_dataloader(self, train_args: BaseTrainArgs, data_dir: str, shade_args: SHADEArgs, world_size: int) -> DataLoader:
        from torchvision.datasets import ImageFolder
        dataset = ShadeDatasetS3(
            data_dir=data_dir,
            transform=self.transform,
            cache_address=shade_args.cache_address,
            PQ=shade_args.pq,
            ghost_cache=shade_args.ghost_cache,
            key_counter=shade_args.key_counter,
            wss=shade_args.working_set_size
        )
        host_ip, port_num = shade_args.cache_address.split(":")

        base_sampler = ShadeSamplerS3(
            dataset=dataset,
            num_replicas=world_size,
            seed=train_args.seed,
            rank=0,
            batch_size=train_args.batch_size(world_size),
            drop_last=False,
            replacement=True,
            host_ip=host_ip,
            port_num=port_num,
            rep_factor=shade_args.replication_factor,
            init_fac=1,
            ls_init_fac=0.01)
        
        batch_sampler = ShadeBatchSampler(sampler=base_sampler, batch_size=train_args.batch_size(world_size), drop_last=False)

        return DataLoader(dataset=dataset, batch_size=None, shuffle=False, num_workers=shade_args.num_pytorch_workers, sampler=batch_sampler)



class CIFAR10DataModule(BaseDataModule):
    def __init__(self):
        # transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # ])
        
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),  # Randomly crop the image with padding
            transforms.RandomHorizontalFlip(),     # Randomly flip the image horizontally
            transforms.ToTensor(),                 # Convert the image to a PyTorch tensor
            transforms.Normalize((0.5, 0.5, 0.5),  # Normalize the image
                                (0.5, 0.5, 0.5))  # Normalize the image
        ])

        super().__init__(transform, num_classes=10)


class ImageNetDataModule(BaseDataModule):
    def __init__(self):
        transform = transforms.Compose([
        transforms.Resize(256),                    # Resize the image to 256x256 pixels
        transforms.RandomResizedCrop(224),   # Randomly crop a 224x224 patch
        transforms.RandomHorizontalFlip(), # Randomly flip the image horizontally
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        super().__init__(transform, num_classes=1000)
