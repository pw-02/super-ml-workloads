from torch.utils.data.sampler import BatchSampler, SequentialSampler, RandomSampler
from typing import Any, Dict, Iterable, Iterator, List, Optional, Union, Sized, Tuple
import hashlib
import torch
import random
from mlworkloads.dataloading.coordl.coorl_dl_batched_dataset import CoorDLBatchedDataset

# Custom BatchSampler that generates unique batch IDs for each batch
class CoorDLBatchSampler(BatchSampler):
    def __init__(self, data_source: Sized, batch_size: int, drop_last: bool, shuffle: bool = False, seed: Optional[int] = None):
        
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

        # Choose the appropriate sampler based on shuffle argument
        if shuffle:
            sampler = RandomSampler(data_source)
        else:
            sampler = SequentialSampler(data_source)
        
        # Initialize the base BatchSampler with the chosen sampler
        super().__init__(sampler, batch_size, drop_last)
        
    def __iter__(self) -> Iterator[Tuple[str, List[int]]]:
             # Generate batches from the underlying sampler
        for batch_indices in super().__iter__():
            # Generate a unique ID for the batch using a hash of the indices
            batch_id = hashlib.md5(str(batch_indices).encode()).hexdigest()
            yield (batch_id, batch_indices)

    def __len__(self) -> int:
        return super().__len__()

# Example usage
if __name__ == "__main__":
    from torch.utils.data import RandomSampler
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms
    
    # Example usage
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


    # Example dataset size
    dataset_size = 100

    # Example usage of BatchSamplerWithID with shuffling
    batch_sampler_with_id = CoorDLBatchSampler(data_source=range(dataset_size), batch_size=10, drop_last=False, shuffle=True, seed=42)
    cooordl_dataset = CoorDLBatchedDataset(s3_data_dir="s3://sdl-cifar10/test/", transform=transform)
    dataloader = DataLoader(cooordl_dataset, sampler=batch_sampler_with_id, num_workers=0, batch_size=None)  # batch_size=None since sampler provides batches

    # Iterate over batches and print batch IDs and indices
    for batch_id, batch_indices in batch_sampler_with_id:
        print(f"Batch ID: {batch_id}, Batch Indices: {batch_indices}")
