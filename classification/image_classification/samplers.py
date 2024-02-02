import torch
from typing import Iterator, Optional, List, TypeVar, Sized, Union, Iterable
from torch.utils.data import Sampler
import sys
from torch.utils.data  import DataLoader
#sys.path.append('/workspaces/super-dl/superdl')
# Specify the path to your module
#print(sys.path)

from super_client import SuperClient

T = TypeVar('T')

class MyDataset:
    def __init__(self, xs: List[T], ys: List[T]) -> None:
        self.xs = xs
        self.ys = ys
        self.dataset_id = 'foo'

    def __getitem__(self, idx: int) -> int:
       indicies, batch_id, status = idx
       return batch_id, indicies

    def __len__(self) -> int:
        return len(self.xs)


class BaseSampler(Sampler[int]):
    r"""Combines elements sequentially or randomly based on the shuffle parameter.

    Args:
        data_source (Dataset): dataset to sample from
        shuffle (bool): If True, samples elements randomly, else sequentially.
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`.
        seed: seed used in sampling generator.
    """

    data_source: Sized
    shuffle: bool
    replacement: bool

    def __init__(self, data_source: Sized, shuffle: bool = True, replacement: bool = False,
                 num_samples: Optional[int] = None, seed: Optional[int] = None) -> None:
        self.data_source = data_source
        self.shuffle = shuffle
        self.replacement = replacement
        self._num_samples = num_samples
        self.seed = seed

        if not isinstance(self.shuffle, bool):
            raise TypeError(f"shuffle should be a boolean value, but got shuffle={self.shuffle}")

        if not isinstance(self.replacement, bool):
            raise TypeError(f"replacement should be a boolean value, but got replacement={self.replacement}")

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(f"num_samples should be a positive integer value, but got num_samples={self.num_samples}")

        if self.shuffle and self.replacement:
            raise ValueError("Both shuffle and replacement cannot be True simultaneously.")

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            n = len(self.data_source)
            if self.seed is None:
                seed = int(torch.empty((), dtype=torch.int64).random_().item())
            else:
                seed = self.seed

            generator = torch.Generator()
            generator.manual_seed(seed)

            if self.replacement:
                for _ in range(self.num_samples // 32):
                    yield from torch.randint(high=n, size=(32,), dtype=torch.int64, generator=generator).tolist()
                yield from torch.randint(high=n, size=(self.num_samples % 32,), dtype=torch.int64, generator=generator).tolist()
            else:
                for _ in range(self.num_samples // n):
                    yield from torch.randperm(n, generator=generator).tolist()
                yield from torch.randperm(n, generator=generator).tolist()[:self.num_samples % n]

        else:
            return iter(range(len(self.data_source)))

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
        
        super(BaseBatchSampler, self).__init__(data_source=data_source, shuffle=shuffle, num_samples=num_samples, seed=seed)
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

    def __init__(self, job_id:int, data_source: Sized, batch_size: int, drop_last: bool,  num_samples: Optional[int] = None, 
                 shuffle: bool = True, seed: int = 0,  super_prefetch_lookahead = 10, super_address = None) -> None:

        super(SUPERSampler, self).__init__(data_source, batch_size, drop_last, num_samples,shuffle, seed, )
        self.super_client = None
        self.super_address = super_address
       
        # self.super_client = SuperClient(super_address)
        self.super_prefetch_lookahead = super_prefetch_lookahead
        self.job_id = job_id
        self.dataset_id = data_source.dataset_id
        
    def share_future_batch_accesses(self, batches: List[List[int]]) -> None:
        """
        Share future batch accesses with the CacheCoordinatorClient.
        """ 
        # if  self.super_client is None:
        #     self.super_client = SuperClient(self.super_address)
        
        if batches and self.super_client is not None:
            #pass
            self.super_client.share_batch_access_pattern(job_id=self.job_id, batches=batches, dataset_id = self.dataset_id)

    def __iter__(self) -> Iterator[List[int]]:
        """
        Iterator to yield batches with prefetching and sharing access patterns.
        """
        if self.super_client is not None:
            del self.super_client
        if self.super_address is not None:
            self.super_client = SuperClient(server_address=self.super_address)
    
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
            batch_indices, batch_id = batch
            if self.super_client is not None:
                status = self.super_client.get_batch_status(batch_id, self.dataset_id)
            else:
                status = False
            updated_batch = (batch_indices, batch_id, status)

            yield updated_batch


def test_sampler(dataset, job_id, shuffle = False, seed = 0, num_Epochs=1, batch_size=10):
    
        
    super_grpc_batch_sampler = SUPERSampler(job_id= job_id, data_source=dataset,batch_size=batch_size,  drop_last=False,
                                            shuffle=shuffle, seed=seed, super_prefetch_lookahead=10, super_address= None)
                                            

    train_loader = DataLoader(dataset, num_workers=0, batch_size=None, sampler=super_grpc_batch_sampler)

    for epoch in range(num_Epochs):
        train_loader.sampler.set_seed(epoch)
        print(f'Epoch: {epoch}:')
        for batch_id,batch_indices in train_loader:
            print(f'Batch ID: {batch_id}, Batch Indices: {batch_indices}')

# Usage example
if __name__ == "__main__":
    import os
    xs = list(range(100))
    ys = list(range(100, 1000))
    dataset = MyDataset(xs, ys)
    job_id = os.getpid()
    test_sampler(dataset=dataset, job_id=job_id, shuffle=True,num_Epochs=4, batch_size=10, seed=None)
