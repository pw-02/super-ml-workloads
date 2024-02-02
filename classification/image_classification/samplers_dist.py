import torch
from typing import Iterator, Optional, List, TypeVar, Sized, Union, Iterable
from torch.utils.data import Sampler
import sys
from torch.utils.data  import DataLoader
import math
from super_client import SuperClient


class BaseDistributedSampler(Sampler):
    r"""Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class:`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size and that any instance of it always
        returns the same elements in the same order.

    Args:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.

    .. warning::
        In distributed mode, calling the :meth:`set_epoch` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    Example::

        >>> # xdoctest: +SKIP
        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(self, dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False) -> None:
        
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


class BaseDistributedBatchSampler(BaseDistributedSampler):

    def __init__(self, data_source: Sized, 
                 batch_size: int, 
                 drop_last: bool, 
                 num_replicas:int,
                 rank: int,
                 shuffle: bool = True, 
                 seed: int = 0) -> None:

        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(f"batch_size should be a positive integer value, but got batch_size={batch_size}")
        if not isinstance(drop_last, bool):
            raise ValueError(f"drop_last should be a boolean value, but got drop_last={drop_last}")
        
        super(BaseDistributedBatchSampler, self).__init__(dataset=data_source, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed, drop_last=drop_last)
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
        

    
class SUPERDistributedSampler(BaseDistributedBatchSampler):

    def __init__(self, job_id:int, data_source: Sized, batch_size: int, drop_last: bool,  
                 num_replcias:int,
                 rank:int, 
                 shuffle: bool = True, seed: int = 0, 
                super_prefetch_lookahead = 10, super_address = None) -> None:

        super(SUPERDistributedSampler, self).__init__(data_source=data_source, batch_size=batch_size, drop_last=drop_last,num_replicas=num_replcias, rank=rank, shuffle=shuffle, seed=seed)
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
            # if self.super_client is not None:
            #     status = self.super_client.get_batch_status(batch_id, self.dataset_id)
            # else:
            #     status = False
            updated_batch = (batch_indices, batch_id, False)

            yield updated_batch
    

