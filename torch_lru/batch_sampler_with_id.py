from torch.utils.data.sampler import Sampler
from torch.utils.data import SequentialSampler, RandomSampler, BatchSampler, DistributedSampler
from typing import Any, Dict, Iterable, Iterator, List, Optional, Union, Sized
import torch
from typing import Iterator, List, Tuple, Union
import hashlib

class BatchSamplerWithID(BatchSampler):
    def __init__(self, sampler: Union[Sampler[int], Iterable[int]], batch_size: int, drop_last: bool):
        super().__init__(sampler, batch_size, drop_last)
    
    def create_batch_id(self, indicies):
    # Convert integers to strings and concatenate them
        id_string = ''.join(str(x) for x in indicies)
        # Hash the concatenated string to generate a unique ID
        unique_id = hashlib.sha1(id_string.encode()).hexdigest() 
        return unique_id

    def __iter__(self) -> Iterator[Tuple[int, List[int]]]:   
        if self.drop_last:
            sampler_iter = iter(self.sampler)
            while True:
                try:
                    # Collect indices to form a batch
                    batch = [next(sampler_iter) for _ in range(self.batch_size)]
                    yield (self.create_batch_id(batch), batch)  # Yield the batch ID and the indices
                except StopIteration:
                    break
        else:
            batch = []
            for idx in self.sampler:
                batch.append(idx)
                if len(batch) == self.batch_size:
                    yield (self.create_batch_id(batch), batch)
                    batch = []
            if len(batch) > 0:  # Handle the final batch, if any
                yield (self.create_batch_id(batch), batch)

    def __len__(self) -> int:
        return super().__len__()  # Use the length calculation from the parent class


class MyDataset:
    def __init__(self, length) -> None:
        self.length = length
        self.dataset_id = 'foo'

    def __getitem__(self, idx: int) -> int:
       
       batch_id, indicies,  = idx
       return batch_id, idx

    def __len__(self) -> int:
        return self.length

if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = MyDataset(length= 100)

    batch_sampler_with_id = BatchSamplerWithID(RandomSampler(dataset), batch_size=10, drop_last=False)

    loader = DataLoader(dataset, batch_size=None, sampler=batch_sampler_with_id, num_workers=0)
    # Iterate through the dataset during training
    for _ in range(0,32):
        for batch_idx, (batch_id, data) in enumerate(loader):
            # Your training code here
            # 'data' contains the input images
            # 'target' contains the corresponding labels
            print(f'{batch_id}_{data}')
            pass


    
    # for batch in batch_sampler:
    #     print(batch)








