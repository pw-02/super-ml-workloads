from copy import deepcopy
from itertools import cycle
from typing import Any, Callable, Dict, List, Optional, Union
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.dataloader import (DataLoader,)
from super_dl.dataset.super_dataset import SUPERDataset

class SUPERDataLoader(DataLoader):
    __doc__ = DataLoader.__doc__

    def __init__(self,dataset: SUPERDataset, num_workers: int = 0, batch_size = None,  *args: Any,**kwargs: Any,) -> None:  
        
        if not isinstance(dataset, (SUPERDataset)):
            raise RuntimeError(f"The provided dataset should be an instance of SUPERIterableImageDataset. Found {dataset}.")
        
        # self.batch_size = None
        self.shuffle = None
        self.current_epoch = 0
        self.batch_size = batch_size
        self.num_workers = num_workers
        self._worker_idx = cycle(list(range(self.num_workers if self.num_workers > 0 else 1)))
        self._worker_idx_iter: Optional[Any] = None
        self._latest_worker_idx = 0
        self._num_samples_yielded_streaming = 0
        self.restore = False
        super().__init__(dataset,*args, batch_size=batch_size , num_workers=self.num_workers,**kwargs)  # type: ignore

    def __iter__(self) -> Any:
        
        if not self.restore:
            self._latest_worker_idx = 0
            self._worker_idx = cycle(list(range(self.num_workers if self.num_workers > 0 else 1)))
            self._worker_idx_iter = iter(self._worker_idx)
            self.current_epoch += 1
            # self._num_samples_yielded_combined = {}
            self._num_samples_yielded_streaming = 0    
            self.dataset.index = 0
            # del(self.dataset.super_client)
        
        for data, target, batch_id in super().__iter__():
            self._latest_worker_idx = next(self._worker_idx_iter)  # type: ignore
            self._num_samples_yielded_streaming += data.size(0)
            
            yield data, target,

        self.restore = False

if __name__ == '__main__':
    import os
    import torchvision
    from super_dl.super_client import SuperClient
    
    super_address = '172.17.0.2:50051'
    data_dir = 's3://sdl-cifar10/train/'
    job_id = os.getpid()
    super_client:SuperClient = SuperClient(super_addresss=super_address)     
    super_client.register_job(job_id=job_id, data_dir=data_dir)
    dataset_info = super_client.get_dataset_details('')
    dataset_size = dataset_info.num_files
    dataset_chunked_size = dataset_info.num_chunks
    del(super_client)
   
    #dataset settings
    batch_size = 128
    cache_address = None
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), normalize])
    simulate_delay = 0.01

    #dataloader settings 
    epochs = 2
    num_workers = 4

    dataset =  SUPERDataset(
        job_id=job_id, 
        data_dir=data_dir,
        batch_size=batch_size,
        transform=transform,
        super_address=super_address,
        cache_address=super_address,
        simulate_delay=simulate_delay)
    
    dataloader = SUPERDataLoader(dataset=dataset, num_workers=num_workers, batch_size=None, )
    
    for epoch in range (1,epochs+1):
        for batch_idx,(images, target) in enumerate(dataloader):
            print(f'epoch: {dataloader.current_epoch}, batch: {batch_idx+1}, Size: {images.size(0)}')
    