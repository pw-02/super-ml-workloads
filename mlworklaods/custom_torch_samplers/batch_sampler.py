from torch.utils.data.sampler import Sampler
from torch.utils.data import SequentialSampler, RandomSampler, BatchSampler, DistributedSampler
from typing import Any, Dict, Iterable, Iterator, List, Optional, Union, Sized
import torch
from typing import Iterator, List, Tuple, Union
import hashlib
import random

class BatchSamplerWithID(BatchSampler):
    def __init__(self, sampler: Union[Sampler[int], Iterable[int]], batch_size: int, drop_last: bool):
        super().__init__(sampler, batch_size, drop_last)

    def create_batch_id(self, indicies):
    # Convert integers to strings and concatenate them
        id_string = ''.join(str(x) for x in indicies)
        # Hash the concatenated string to generate a unique ID
        unique_id = hashlib.sha1(id_string.encode()).hexdigest() 
        return unique_id
    
    def _generate_batches(self):
        all_indices = list(self.sampler)
        batches = [all_indices[i:i+self.batch_size] for i in range(0, len(all_indices), self.batch_size)]
        self.batch_list = [(self.create_batch_id(batch), batch) for batch in batches]
        random.shuffle(self.batch_list)

    def __iter__(self):
        self._generate_batches()
       
        for batch_id, batch in self.batch_list:
            yield batch_id, batch

    def __len__(self):
        return len(self.batch_list)

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

    batch_sampler_with_id = BatchSamplerWithID(RandomSampler(data_source=dataset,generator=torch.Generator().manual_seed(42)), 
                                                     batch_size=10, drop_last=False)

    loader = DataLoader(dataset, batch_size=None, sampler=batch_sampler_with_id, num_workers=0)
    # Iterate through the dataset during training
    for _ in range(0,32):
        print()
        for batch_idx, (batch_id, data) in enumerate(loader):
            # Your training code here
            # 'data' contains the input images
            # 'target' contains the corresponding labels
            print(f'{batch_id}_{data}')
            pass


    
    # for batch in batch_sampler:
    #     print(batch)








