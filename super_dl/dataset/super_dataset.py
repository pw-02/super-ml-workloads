import functools
import torch
import super_dl.s3utils as s3utils
from typing import List, Tuple, Dict
from PIL import Image
from super_dl.super_client import SuperClient
from typing import Any, Dict, List, Optional, Tuple, Union
from torch.utils.data import SequentialSampler, IterableDataset, RandomSampler, DataLoader, get_worker_info
import redis
import zlib
import io
import base64
import functools
import logging
import time
import math

# from lightning.fabric import Fabric

def configure_logger():
    # Set the log levels for specific loggers to WARNING
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("boto3").setLevel(logging.WARNING)

    # Configure the root logger with a custom log format
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger("SUPERWorkload")
    return logger

logger = configure_logger()

class SUPERDataset(IterableDataset):
    def __init__(self,
                 job_id:int,
                 data_dir:str,
                 batch_size:int,
                 transform,   
                 super_address:str,
                 world_size:int,
                 cache_address:str = None,
                 simulate_delay = None ):
        
        super().__init__()

        super_client:SuperClient = SuperClient(super_addresss=super_address)
        dataset_info = super_client.get_dataset_details(data_dir)
        self.dataset_size = dataset_info.num_files
        self.dataset_chunked_size = dataset_info.num_chunks // world_size
        self.samples: Dict[str, List[str]] = s3utils.load_paired_s3_object_keys(data_dir, True, True)
        # del(super_client)
        self.job_id = job_id
        self.data_dir = data_dir
        self.s3_bucket_name = s3utils.S3Url(data_dir).bucket
        self.transform = transform
        self.super_address  = super_address
        self.batch_size = batch_size
        self.simulate_delay = simulate_delay
        self.index = 0
        self.cache_host, self.cache_port = None,None
        if cache_address:
            self.cache_host, self.cache_port = cache_address.split(":")
        # self.cache_address, self.cache_port = cache_address.split(":") if cache_address else None, None
        if self.cache_host:
            self.setup_cache_client()
        # self.cache_client = None
        self.super_client = None

    
    @functools.cached_property
    def _classed_items(self) -> List[Tuple[str, int]]:
        return [(blob, class_index)
                for class_index, blob_class in enumerate(self.samples)
                for blob in self.samples[blob_class]
                ]


    def setup_cache_client(self):
        self.cache_client = redis.StrictRedis(host=self.cache_host, port=self.cache_port)
      
    # def setup_super_client(self):
    #     self.super_client:SuperClient = SuperClient(super_addresss=self.super_address)
    #     # dataset_info = self.super_client.get_dataset_details( self.data_dir)
    #     # self.dataset_size = dataset_info.num_files
    #     # self.dataset_chunked_size = dataset_info.num_chunks

    def __len__(self) -> int:
        return self.dataset_chunked_size

    def __iter__(self) -> "SUPERDataset":
        # if self.super_client is None:
        #     self.setup_super_client() 
        worker_info = get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            return self.__iter_non_distributed__(self.dataset_chunked_size)
        else:
            #split workload
            num_workers = worker_info.num_workers
            per_worker = self.dataset_chunked_size // num_workers
            remaining_work = self.dataset_chunked_size % num_workers
            workloads = [per_worker + 1 if i < remaining_work else per_worker for i in range(num_workers)]
            return self.__iter_non_distributed__(workloads[worker_info.id])

    
    def split_into_batches(self, images, labels, batch_size):
        num_batches = (images.size(0) + batch_size - 1) // batch_size
        return [(images[i * batch_size:(i + 1) * batch_size], labels[i * batch_size:(i + 1) * batch_size]) for i in range(num_batches)]


    def __iter_non_distributed__(self, iter_len):
        # worker_size = self.dataset_chunked_size // self.num_workers #equals the number of calls to super for 1 epoch
        prepared_batches = []
        super_client:SuperClient = SuperClient(super_addresss=self.super_address)
        while self.index < iter_len:
            if prepared_batches:
                yield prepared_batches.pop(0)
            else:
                next_batch = self._get_next_super_batch(super_client)
                next_batch_size = len(next_batch.indicies)
                if next_batch_size == self.batch_size:
                    yield self.__getitem__(next_batch.batch_id, next_batch.indicies, next_batch.is_cached)
                elif next_batch_size > self.batch_size:   
                    imgs, labels, batch_id = self.__getitem__(next_batch.batch_id, next_batch.indicies, next_batch.is_cached)
                    split_batches = self.split_into_batches(imgs, labels, self.batch_size)

                    for timgs, labels in split_batches:
                        prepared_batches.append((timgs, labels,batch_id))
                    
                    yield prepared_batches.pop(0)
                else:
                    yield self.__getitem__(next_batch.batch_id, next_batch.indicies, next_batch.is_cached)

                    # while torch_imgs.size(0) != self.batch_size and self.index < iter_len:
                    #     new_batch = self._get_next_super_batch(super_client)
                    #     imgs, labels, batch_id = self.__getitem__(new_batch.batch_id, new_batch.indicies, new_batch.is_cached)
                    #     torch_imgs = torch.cat((torch_imgs, imgs), dim=0)
                    #     torch_labels = torch.cat((torch_labels, labels), dim=0)

                    # yield torch_imgs, torch_labels, batch_id, cache_hit
        # self.index = 0
    
    def _get_next_super_batch(self,super_client:SuperClient):       
        next_batch = super_client.get_next_batch(self.job_id)
        if not next_batch:
            raise ValueError("Empty batch returned by super_client.")
        self.index +=1
        return next_batch[0]

    def __getitem__(self, batch_id: str, batch_indicies:List[int], is_cached:bool) -> Tuple[Any, Any]:
        if self.simulate_delay:
            time.sleep(self.simulate_delay)
            torch_imgs = torch.empty(len(batch_indicies), 3, 32, 32)
            torch_labels = torch.empty(len(batch_indicies))
            cache_hit = True
        else:
            cache_hit = False
            batch_data = None
            if is_cached and self.cache_client is not None:
                batch_data = self.fetch_from_cache(batch_id)

            if batch_data is not None:
                torch_imgs, torch_labels = self.deserialize_torch_batch(batch_data)
                cache_hit = True
            else:
                imgs, labels = self.fetch_from_s3(batch_indicies)
                imgs, labels = self.apply_transformations(imgs, labels)
                torch_imgs, torch_labels = torch.stack(imgs), torch.tensor(labels)
        if cache_hit:
            cache_hit_count = len(batch_indicies)
        else:
            cache_hit_count = 0

        return torch_imgs, torch_labels, cache_hit_count
     
    
    def fetch_from_cache(self, batch_id, max_attempts = 5):
        response = None
        attempts = 0
         # Try fetching from cache initially
        response = self.cache_get(batch_id)
        if response is not None and not isinstance(response, Exception):
            return response
        else:
            # Retry fetching from cache for a specified number of attempts
            while attempts < max_attempts:
                logger.error(f"Error fetching batch '{batch_id}' from cache: {response}. Retrying (attempt {attempts}).")
                response = self.cache_get(batch_id)
                if response is not None and not isinstance(response, Exception):
                    break
                else:
                    time.sleep(0.01)
                attempts += 1
        return response
    
    def cache_get(self, batch_id, max_attempts = 1):     
        try:
            return self.cache_client.get(batch_id)
        except Exception as e:
             return e
    
    # def fetch_from_cache(self, batch_id, max_attempts = 1):     
    #     cached_data = None
    #     try:
    #         cached_data = self.cache_client.get(batch_id)
    #     except Exception as e:
    #         logger.error(f"Error fetching batch '{batch_id}' from cache: {e}. Retrying.")
    #         cached_data = None
    #     return cached_data
    
    
    def fetch_from_s3(self, indices:List[int]):
        images = []
        labels = []   
        for idx in indices:
            file_path, label = self._classed_items[idx]
            content = s3utils.get_s3_object(self.s3_bucket_name, file_path)
            img = Image.open(io.BytesIO(content))
            if img.mode == "L":
                img = img.convert("RGB")
            images.append(img)
            labels.append(label)
        return images, labels
    

    def apply_transformations(self, images, labels):
        if self.transform is not None:
            for i in range(len(images)):
                images[i] = self.transform(images[i])
        return images, labels

    def deserialize_torch_batch(self, batch_data):
        decoded_data = base64.b64decode(batch_data) # Decode base64
        decoded_data = zlib.decompress(decoded_data)
        buffer = io.BytesIO(decoded_data)
        batch_samples, batch_labels = torch.load(buffer)
        return  batch_samples, batch_labels