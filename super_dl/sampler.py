from torch.utils.data import SequentialSampler, RandomSampler, BatchSampler, DistributedSampler
from typing import Any, Dict, Iterator, List, Optional, Union, Sized
from dataclasses import dataclass
import numpy as np
from super_client import SuperClient
import torch


class SuperBatchSampler(BatchSampler):
    def __init__(self,
        data_source: Sized,       
        job_id: Any,
        num_replicas: int,
        global_rank: int,
        batch_size: int,
        drop_last: bool,
        shuffle: bool,
        seed = None,
        super_address = None,
        super_prefetch_lookahead = None):
        """The SUPERBatchSampler handles the generation of batch indices.
            Arguments:
                dataset_size: The size of the dataset.
                num_replicas: The number of processes involves in the distributed training.
                global_rank: The global_rank of the given process
                num_workers: The number of workers provided to the DataLoader.
                batch_size: The number of items in a batch.
                drop_last: Whether to drop the last batch of data.
                shuffle: Whether the data should be shuffled.
            """
        self._dataset_id = data_source.dataset_id
        self._job_id = job_id
        self._super_address = super_address
        if super_prefetch_lookahead is None:
            self._super_prefetch_lookahead = 1
        else:
            self._super_prefetch_lookahead = super_prefetch_lookahead
        self._super_client:SuperClient = None
        self._seed = seed if seed is not None else 0
        self._epoch = 0
        
        if  num_replicas <= 1:
            if shuffle:
                if self._seed > 0:
                    g = torch.Generator()
                    g.manual_seed(self._seed)
                super(SuperBatchSampler, self).__init__(RandomSampler(data_source,generator= g if self._seed > 0 else None),batch_size=batch_size, drop_last=drop_last )
            else:
                super(SuperBatchSampler, self).__init__(SequentialSampler(data_source),batch_size=batch_size, drop_last=drop_last)
        else:  
             super(SuperBatchSampler, self).__init__(DistributedSampler(dataset=data_source,  num_replicas=num_replicas, rank=global_rank, shuffle=shuffle, seed=self._seed,  drop_last=drop_last),batch_size=batch_size, drop_last=drop_last)
       

    def __gen_batch_id__(self, batch_indices: List[int]) -> None:
        batch_id = abs(hash(tuple(batch_indices)))
        return batch_id

    def __share_future_batch_accesses__(self, batches: List[List[int]]) -> None:
        """
        Share future batch accesses with the CacheCoordinatorClient.
        """   
        if self._super_client is not None:
            batches_with_ids= []
            for batch in batches:
                batches_with_ids.append(batch,self.__gen_batch_id__(batch) )
            self._super_client.share_batch_access_pattern(job_id=self._job_id, batches=batches_with_ids, dataset_id = self._dataset_id)

    def __iter__(self) ->Iterator[List[int]]:

        if self._super_client is not None:
            del self._super_client
        if self._super_address is not None:
            self.super_client = SuperClient(server_address=self._super_address)

        buffer = []
        batch_iter = super().__iter__()
        batches_exhausted = False


        for _ in range(self._super_prefetch_lookahead * 2):
            try:
                buffer.append(next(batch_iter))
            except StopIteration:
                break   
        self.__share_future_batch_accesses__(buffer)

        while buffer:
            if len(buffer) <= self._super_prefetch_lookahead and not batches_exhausted:
                prefetch_buffer = []
                for _ in range(self._super_prefetch_lookahead):
                    try:
                        prefetch_buffer.append(next(batch_iter))
                    except StopIteration:
                        batches_exhausted = True
                        break
                
                self.__share_future_batch_accesses__(prefetch_buffer)
                buffer.extend(prefetch_buffer)
            next_batch = buffer.pop(0)
            # if self.super_client is not None:
            #     status = self.super_client.get_batch_status(batch_id, self.dataset_id)
            # else:
            #     status = False
            next_batch = (next_batch, self.__gen_batch_id__(next_batch), False)
            yield next_batch
    
    def set_seed(self, epoch: int) -> None:
        self._seed = epoch


class MyDataset:
    def __init__(self, xs) -> None:
        self.length = xs
        self.dataset_id = 'foo'

    def __getitem__(self, idx: int) -> int:
       #indicies, batch_id, status = idx
       return idx

    def __len__(self) -> int:
        return self.length

if __name__ == "__main__":
    dataset_size = 100
    world_size = 1
    rank = 0
    num_workers = 32
    batch_size = 10
    cache = None
    job_id = 1
    dataset_id =2

    dataset = MyDataset(dataset_size)
    batch_sampler = SuperBatchSampler(
        data_source=dataset, 
        job_id=job_id, 
        num_replicas=world_size, 
        global_rank= rank, 
        batch_size= batch_size, 
        drop_last=False, 
        shuffle=True, 
        seed=0, 
        super_address=None, 
        super_prefetch_lookahead=None)

    batches_1 = []
    for batch in batch_sampler:
        print(batch)
        batches_1.append(batch)