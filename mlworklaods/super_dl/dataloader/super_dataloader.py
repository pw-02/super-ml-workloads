from copy import deepcopy
from itertools import cycle
from typing import Any, Callable, Dict, List, Optional, Union
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.dataloader import (
    DataLoader,
    _BaseDataLoaderIter,
    _SingleProcessDataLoaderIter,
)

from mlworklaods.super_dl.datasets.super_imgae_iterable import SUPERIterableImageDataset

class SUPERDataLoader(DataLoader):

    __doc__ = DataLoader.__doc__

    def __init__(
        self,
        dataset: SUPERIterableImageDataset,
        *args: Any,
        batch_size: int = 1,
        num_workers: int = 0,
        profile_batches: Union[bool, int] = False,
        profile_dir: Optional[str] = None,
        prefetch_factor: Optional[int] = None,
        **kwargs: Any,
    ) -> None:  
        # pyright: ignore
        if not isinstance(dataset, (SUPERIterableImageDataset)):
            raise RuntimeError(
                "The provided dataset should be an instance of SUPERIterableImageDataset."
                f" Found {dataset}.")
        
        self.current_epoch = 0
        self.batch_size = batch_size
        self.num_workers = num_workers
        self._profile_batches = profile_batches
        self._profile_dir = profile_dir
        self._num_samples_yielded_streaming = 0
        self.rng_state: Optional[Any] = None
        self._worker_idx = cycle(list(range(self.num_workers if self.num_workers > 0 else 1)))
        self._worker_idx_iter: Optional[Any] = None
        self._latest_worker_idx = 0
        self.restore = False

        super().__init__(
            dataset,
            *args,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=(10 if num_workers > 0 else None) if prefetch_factor is None else prefetch_factor,
            **kwargs,
        )  # type: ignore

    def __iter__(self) -> Any:
        if not self.restore:
            self._latest_worker_idx = 0
            self._worker_idx = cycle(list(range(self.num_workers if self.num_workers > 0 else 1)))
            self._worker_idx_iter = iter(self._worker_idx)
            self.current_epoch += 1
            # self._num_samples_yielded_combined = {}
            self._num_samples_yielded_streaming = 0

        self.dataset.set_epoch(self.current_epoch)

        if isinstance(self.dataset, SUPERIterableImageDataset):
            assert self.batch_size
            for batch in super().__iter__():
                self._latest_worker_idx = next(self._worker_idx_iter)  # type: ignore
                self._num_samples_yielded_streaming += self.batch_size
                yield batch

        self.restore = False

    def state_dict(self) -> Dict[str, Any]:
        if isinstance(self.dataset, SUPERIterableImageDataset):
            assert self.batch_size
            return {
                "dataset": self.dataset.state_dict(
                    self._num_samples_yielded_streaming, self.num_workers, self.batch_size
                ),
                "current_epoch": self.current_epoch,
                "num_samples_yielded": self._num_samples_yielded_streaming,
                "latest_worker_idx": self._latest_worker_idx,
            }

        num_samples_yieled = [0 for _ in range(len(list(self._num_samples_yielded_combined.values())[0]))]
        for worker_idx in self._num_samples_yielded_combined:
            for dataset_idx, samples_yieled in enumerate(self._num_samples_yielded_combined[worker_idx]):
                num_samples_yieled[dataset_idx] += samples_yieled

        return {
            "dataset": self.dataset.state_dict(self.num_workers, self.batch_size, num_samples_yieled),
            "current_epoch": self.current_epoch if self.restore else self.current_epoch - 1,
            "latest_worker_idx": self._latest_worker_idx,
            "num_samples_yielded": deepcopy(self._num_samples_yielded_combined),
        }


if __name__ == '__main__':
    import os
    import torchvision

    job_id = os.getpid()
    super_address = '172.17.0.2:50051'
    cache_address = None
    data_dir = 's3://sdl-cifar10/train/'
    batch_size = 1000
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), normalize])
    simulate_delay = 0.2

    dataset =  SUPERIterableImageDataset(data_dir = data_dir, 
                                         transform=transform,
                                           batch_size=batch_size,
                                           job_id=job_id, 
                                           super_address=super_address,
                                           simulate_delay=simulate_delay)
    
    dataloader = SUPERDataLoader(dataset=dataset, batch_size=batch_size, num_workers=0)

    for batch_idx,(images, target, batch_id) in enumerate(dataloader):
        print(f'{batch_idx+1}: {batch_id}')
    