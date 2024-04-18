import functools
import super_dl.s3utils as s3utils
from  super_dl.s3utils import S3Url
from typing import List, Tuple, Dict
from torch.utils.data import SequentialSampler, IterableDataset, RandomSampler, DataLoader
import torch
import torch.nn.functional as F
import numpy as np
import sys

class LocalTextIterableDataset(IterableDataset):
    def __init__(self,data_dir:str, block_size:int):
        super().__init__()
        self.epoch = 0
        self.data_file = data_dir
        self.block_size = block_size
    
    def __iter__(self):
        data = np.memmap(self.data_file, dtype=np.uint16, mode="r")
        while True:
            i = torch.randint(len(data) - self.block_size, (1,)).item()
            x = torch.from_numpy((data[i : i + self.block_size]).astype(np.int64))
            y = torch.from_numpy((data[i + 1 : i + 1 + self.block_size]).astype(np.int64))
            yield x, y

    def __len__(self):
        return sys.maxsize

    def set_epoch(self, epoch):
        self.epoch = epoch
if __name__ == "__main__":

    def get_batch_size_mb(batch_tensor):
        import sys
        # Get the size of the tensor in bytes
        size_bytes = sys.getsizeof(batch_tensor.storage()) + sys.getsizeof(batch_tensor)
        # Convert bytes to megabytes
        size_mb = size_bytes / (1024 ** 2)
        # Convert bytes to kb
        size_in_kb = size_bytes / 1024
        return size_mb,size_in_kb

    # # Example usage
    train_data_dir = 'datasets/openwebtext/val.bin'

    dataset = LocalTextIterableDataset(data_dir=train_data_dir,
                                    block_size=2048,
                                    shuffle=True)

    data_loader = DataLoader(dataset, batch_size=12)
    # Get the size of the tensor using pympler
    for input, target in data_loader:
        batch_size_mb,size_in_kb  = get_batch_size_mb(input)
        print(f"Batch size: {batch_size_mb:.2f} MB, {size_in_kb:.2f} KB")
        print(input.shape, target.shape)
