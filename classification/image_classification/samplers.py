import torch
from typing import Iterator, Optional, List, TypeVar, Sized, Union, Iterable
from torch.utils.data import Sampler
import sys

#sys.path.append('/workspaces/super-dl/superdl')
# Specify the path to your module
#print(sys.path)

from super_client import SuperClient

T = TypeVar('T')

class MyDataset:
    def __init__(self, xs: List[T], ys: List[T]) -> None:
        self.xs = xs
        self.ys = ys

    def __getitem__(self, idx: int) -> int:
        return idx

    def __len__(self) -> int:
        return len(self.xs)
    
class BaseSampler(Sampler[int]):
    data_source: Sized
    def __init__(self, data_source: Sized, num_samples: Optional[int] = None, shuffle: bool = True, seed: int = 0) -> None:
        self.data_source = data_source
        self._num_samples = num_samples
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(f"num_samples should be a positive integer value, but got num_samples={self.num_samples}")
        
    @property
    def num_samples(self) -> int:
        return self._num_samples if self._num_samples is not None else len(self.data_source)

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.num_samples, generator=g).tolist()
        else:
            indices = list(range(self.num_samples))

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def set_seed(self, seed: int) -> None:
        self.seed = seed

class BaseBatchSampler(BaseSampler):

    def __init__(self, data_source: Sized, batch_size: int, drop_last: bool, num_samples: Optional[int] = None, shuffle: bool = True, seed: int = 0) -> None:

        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(f"batch_size should be a positive integer value, but got batch_size={batch_size}")
        if not isinstance(drop_last, bool):
            raise ValueError(f"drop_last should be a boolean value, but got drop_last={drop_last}")
        
        super(BaseBatchSampler, self).__init__(data_source, num_samples, shuffle, seed)
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[List[int]]:
        if self.drop_last:
            sampler_iter = iter(super().__iter__())
            while True:
                try:
                    batch = [next(sampler_iter) for _ in range(self.batch_size)]
                    batch_id = abs(hash(tuple(batch)))
                    yield batch, batch_id
                except StopIteration:
                    break
        else:
            batch = [0] * self.batch_size
            idx_in_batch = 0
            for idx in super().__iter__():
                batch[idx_in_batch] = idx
                idx_in_batch += 1
                if idx_in_batch == self.batch_size:
                    batch_id = abs(hash(tuple(batch)))
                    yield batch, batch_id
                    idx_in_batch = 0
                    batch = [0] * self.batch_size

            if idx_in_batch > 0:
                batch_id = abs(hash(tuple(batch[:idx_in_batch])))
                yield batch[:idx_in_batch], batch_id

    def __len__(self) -> int:
        if self.drop_last:
            return super().__len__() // self.batch_size
        else:
            return (super().__len__() + self.batch_size - 1) // self.batch_size



class PytorchVanilliaSampler(BaseSampler):
  
  def __init__(self, data_source: Sized, num_samples: Optional[int] = None, shuffle: bool = True, seed: int = 0) -> None:
        super(PytorchVanilliaSampler, self).__init__(data_source, num_samples, shuffle, seed)


class PytorchBatchSampler(BaseBatchSampler):
    def __init__(self, data_source: Sized, batch_size: int, drop_last: bool, num_samples: Optional[int] = None, shuffle: bool = True, seed: int = 0) -> None:
        super(PytorchBatchSampler, self).__init__(data_source, batch_size, drop_last, num_samples, shuffle, seed)

class SUPERSampler(BaseBatchSampler):

    def __init__(self, data_source: Sized, batch_size: int, drop_last: bool, job_id:int, num_samples: Optional[int] = None, 
                 shuffle: bool = True, seed: int = 0,super_client: SuperClient = None, super_prefetch_lookahead = 10) -> None:

        super(SUPERSampler, self).__init__(data_source, batch_size, drop_last, num_samples,shuffle, seed )

        self.super_client = super_client
        self.super_prefetch_lookahead = super_prefetch_lookahead
        self.job_id = job_id
        self.dataset_id = data_source.dataset_id
        
    def share_future_batch_accesses(self, batches: List[List[int]]) -> None:
        """
        Share future batch accesses with the CacheCoordinatorClient.
        """ 
        if batches and self.super_client is not None:
            self.super_client.share_batch_access_pattern(job_id=self.job_id, batches=batches, dataset_id = self.dataset_id)


    def __iter__(self) -> Iterator[List[int]]:
        """
        Iterator to yield batches with prefetching and sharing access patterns.
        """
        batch_buffer = []
        batch_iter = super().__iter__()
        batches_exhausted = False

        for _ in range(self.super_prefetch_lookahead * 2):
            try:
                batch_buffer.append(next(batch_iter))
            except StopIteration:
                break
            
        self.share_future_batch_accesses(batch_buffer)

        while batch_buffer:
            if len(batch_buffer) <= self.super_prefetch_lookahead and not batches_exhausted:
                prefetch_buffer = []
                for _ in range(self.super_prefetch_lookahead):
                    try:
                        prefetch_buffer.append(next(batch_iter))
                    except StopIteration:
                        batches_exhausted = True
                        break
                
                self.share_future_batch_accesses(prefetch_buffer)
                batch_buffer.extend(prefetch_buffer)
            batch = batch_buffer.pop(0)
            yield batch


# def test_sampler(dataset, job_id, use_super = False, num_Epochs=1, batch_size=10):
        
#     cache_coordinator_client = None

#     if use_super:
#         cache_coordinator_client = SuperClient()
#         cache_coordinator_client.register_new_job(job_id,data_dir='mlworkloads/vision/data/cifar-10', source_system='local')

#     super_grpc_batch_sampler = SUPERSampler(data_source=dataset, job_id= job_id, super_client=cache_coordinator_client,
#                                             shuffle=False,
#                                             seed=0,
#                                             batch_size=batch_size,
#                                             drop_last=False)

#     train_loader = torch.utils.data.DataLoader(dataset, num_workers=0, batch_size=None, sampler=super_grpc_batch_sampler)

#     for epoch in range(num_Epochs):
#         train_loader.sampler.set_seed(epoch)
#         print(f'Epoch: {epoch}:')
#         for batch_indices, batch_id in train_loader:
#             print(f'Batch ID: {batch_id}, Batch Indices: {batch_indices}')

# # Usage example
# if __name__ == "__main__":
#     import os
#     xs = list(range(100))
#     ys = list(range(100, 1000))
#     dataset = MyDataset(xs, ys)
#     base_sampler = SuperBaseSampler(dataset, shuffle=False)
#     job_id = os.getpid()
#     test_sampler(dataset=dataset, job_id=job_id, use_super=False,num_Epochs=1, batch_size=1)
