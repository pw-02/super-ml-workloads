from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torchvision import transforms
from mlworklaods.dataloaders.torch_lru.batch_sampler_with_id import BatchSamplerWithID
from mlworklaods.dataloaders.torch_lru.torch_lru_dataset import TorchLRUDataset
from mlworklaods.dataloaders.super_dl.dataset.super_dataset import SUPERDataset
from typing import Tuple, Optional
from mlworklaods.args import * 

class BaseDataModule:
    def __init__(self, transform, num_classes: int):
        self.transform = transform
        self.num_classes = num_classes

    def make_dataloaders(self, train_args: ImgClassifierTrainArgs, data_args: DataArgs, dataloader_args, world_size: int):

        if isinstance(dataloader_args, SUPERArgs):
            return self.make_super_dataloaders(train_args, data_args, dataloader_args, world_size)
        elif isinstance(dataloader_args, LRUTorchArgs):
            return self.make_lru_torch_dataloaders(train_args, data_args, dataloader_args, world_size)
        else:
            raise Exception(f"Unknown dataloader_kind {train_args.dataloader_kind}")

    def make_lru_torch_dataloaders(self, train_args: BaseTrainArgs, data_args: DataArgs, lru_torch_args: LRUTorchArgs, world_size: int) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
        train_dataloader = self.create_lru_dataloader(train_args, data_args.train_data_dir, lru_torch_args, world_size) if train_args.run_training else None
        val_dataloader = self.create_lru_dataloader(train_args, data_args.val_data_dir, lru_torch_args, world_size) if train_args.run_evaluation else None
        return train_dataloader, val_dataloader

    def create_lru_dataloader(self, train_args: BaseTrainArgs, data_dir: str, lru_torch_args: LRUTorchArgs, world_size: int) -> DataLoader:
        dataset = TorchLRUDataset(
            data_dir=data_dir,
            transform=self.transform,
            cache_address=lru_torch_args.cache_address,
            cache_granularity=lru_torch_args.cache_granularity
        )
        base_sampler = RandomSampler(data_source=dataset) if lru_torch_args.shuffle else SequentialSampler(data_source=dataset)
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


class CIFAR10DataModule(BaseDataModule):
    def __init__(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        super().__init__(transform, num_classes=10)


class ImageNetDataModule(BaseDataModule):
    def __init__(self):
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        super().__init__(transform, num_classes=1000)
