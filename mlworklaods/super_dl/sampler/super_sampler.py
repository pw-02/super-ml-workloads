from mlworklaods.super_dl.super_client import SuperClient
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Union

@dataclass
class ChunkedIndex:
    index: int
    chunk_index: int
    chunk_indexes: Optional[List[int]] = None
    is_last_index: bool = False

class SUPERBacthSampler:
    def __init__(
        self,
        job_id:int,
        # dataset_size: int,
        # dataset_chunks:int,
        num_replicas: int, #The number of processes involves in the distributed training.
        global_rank: int, #The global_rank of the given process
        num_workers: int, #The number of workers provided to the DataLoader.
        batch_size: int, #The number of items in a batch.
        drop_last: bool, #Whether to drop the last batch of data
        super_client:SuperClient = None
    ):
        self._super_client = super_client  
        dataset_info = self._super_client.get_dataset_details('')
        self.job_id = job_id
        self._dataset_size = dataset_info.num_files
        self._dataset_chunked_size = dataset_info.num_chunks
        self.chunk_size = dataset_info.chunk_size
        self._num_replicas = num_replicas
        self._global_rank = global_rank
        self._num_workers = num_workers or 1
        self._shuffled_chunk_intervals = None
        self._batch_size = batch_size
        self._drop_last = drop_last
        self.current_index = 0

        assert self.chunk_size % self._batch_size == 0, "Chunk size is not a multiple of batch size."

    
    def __iter__(self) -> Iterator[List[Union[int, ChunkedIndex]]]:
        if self._num_replicas == 1:
            return self.__iter_non_distributed__()
        # return self.__iter_distributed__()
    
    def _get_next_batch(self):
        next_batch = self._super_client.get_next_batch(self.job_id)
        if not next_batch:
            raise ValueError("Empty batch returned by super_client.")
        self.current_index +=1
        return next_batch[0]
    
    def _split_into_batches(self, next_batch):
        # Calculate number of batches
        num_batches = len(next_batch.indicies) // self._batch_size
        
        # Iterate through next_batch and yield batches
        for i in range(0, len(next_batch.indicies), self._batch_size):
            yield next_batch.indicies[i:i + self._batch_size]

    def __iter_non_distributed__(self):
        worker_size = self._dataset_chunked_size // self._num_workers #equals the number of calls to super for 1 epoch
        batches = []

        while self.current_index < worker_size:
            if batches:
                yield batches.pop(0)
            else:
                next_batch = self._get_next_batch()
                next_batch_size = len(next_batch.indicies)
                if next_batch_size == self._batch_size:
                    yield next_batch.batch_id, len(next_batch.indicies)

                elif next_batch_size > self._batch_size:                        
                    for i in range(0, len(next_batch.indicies), self._batch_size):
                        batches.append(next_batch.batch_id,len(next_batch.indicies))
                        next_batch = batches.pop(0).batch_id
                        yield next_batch.batch_id, len(next_batch.indicies)
                else:
                    combined_batches = [(next_batch.batch_id,len(next_batch.indicies))]
                    while next_batch_size != self._batch_size and self.current_index < worker_size:
                        new_batch = self._get_next_batch()
                        next_batch_size += len(new_batch)
                        combined_batches.append((new_batch.batch_id,len(next_batch.indicies)))
                    yield combined_batches
                
    
if __name__ == '__main__':
    import os
    import torchvision

    job_id = os.getpid()
    super_address = '172.17.0.2:50051'
    data_dir = 's3://sdl-cifar10/train/'
    super_client:SuperClient = SuperClient(super_addresss=super_address)     
    super_client.register_job(job_id, data_dir)

    sampler = SUPERBacthSampler(
        job_id=job_id,
        num_replicas=1,
        global_rank=0,
        num_workers=1,
        batch_size=64,
        drop_last=False,
        super_client= super_client
    )


    for batch_idx,(batch_id) in enumerate(sampler):
        print(f'{batch_idx+1}: {batch_id}')
    