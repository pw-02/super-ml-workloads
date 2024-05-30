import os
import glob
from typing import List
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
from typing import List, Tuple, Dict
import mlworklaods.s3utils as s3utils
from mlworklaods.s3utils import S3Url
import glob
import time
import redis
import io
import base64
import zlib
from mlworklaods.dataloaders.super_dl.super_client import SuperClient
from torch.utils.data import SequentialSampler, IterableDataset, RandomSampler, DataLoader, get_worker_info
from transformers import PreTrainedTokenizer
import logging

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

class TorchLRUTextDataset(IterableDataset):

    def __init__(self,
                job_id:int,
                 data_dir: str, 
                 tokenizer: PreTrainedTokenizer, 
                 transform, 
                 block_size: int, 
                 batch_size: int, 
                 super_address:str,
                 world_size:int,
                 cache_address:str = None,
                 simulate_delay = None):

        super().__init__()
        super_client:SuperClient = SuperClient(super_addresss=super_address)
        super_client.register_job(job_id, data_dir)
        dataset_info = super_client.get_dataset_details(data_dir)
        self.dataset_size = dataset_info.num_files
        self.dataset_chunked_size = dataset_info.num_chunks // world_size
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
        if self.cache_host:
            self.setup_cache_client()

        self.super_client = None
        self.tokenizer:PreTrainedTokenizer = tokenizer
        self.block_size = block_size
        self.is_s3: bool = data_dir.startswith("s3://")
        if self.is_s3:
            self.file_list: Dict[str, List[str]] = s3utils.load_unpaired_s3_object_keys(data_dir, False, True)
            self.bucket_name = S3Url(data_dir).bucket
        else:
            self.file_list = glob.glob(f'{self.data_dir}/*.txt')
            

    def setup_cache_client(self):
        self.cache_client = redis.StrictRedis(host=self.cache_host, port=self.cache_port)

    def __len__(self) -> int:
        return self.dataset_chunked_size

    def __iter__(self):
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


    def __iter_non_distributed__(self, num_files):   
        super_client:SuperClient = SuperClient(super_addresss=self.super_address)
        current_file_idx = 0
        current_tokens = None
        current_position = 0

        while current_file_idx < num_files:
            fetch_start_time = time.perf_counter()

            if current_tokens is None:
                current_tokens, transform_duration, cache_hit = self._load_next_file(super_client)
                current_file_idx +=1
                current_position = 0
            required_size = (self.block_size +1) * self.batch_size
            remaining_tokens = current_tokens[current_position:]

            while len(remaining_tokens) < required_size:
                next_tokens, transform_duration, cache_hit = self._load_next_file(super_client)
                current_file_idx +=1
                current_position = 0
                remaining_tokens = torch.cat((remaining_tokens, next_tokens), dim=0)
        
            batch_tokens = remaining_tokens[:required_size]
            current_position += required_size
            current_tokens = remaining_tokens[required_size:]
            fetch_duration = time.perf_counter() - fetch_start_time
            
            if cache_hit:
                cache_hits = 1
            else:
                cache_hits = 0

            yield batch_tokens,fetch_duration, transform_duration, cache_hits

        current_file_idx = 0
    
    def _load_next_file(self, super_client:SuperClient):
        next_file = super_client.get_next_batch(self.job_id)
        
        if not next_file:
            raise ValueError("Empty file returned by super_client.")
        
        file_path, is_cached = next_file.file_path, next_file.is_cached
        data = None
        
        if self.simulate_delay:
            time.sleep(self.simulate_delay)
            tokens = torch.empty(len(self.block_size), 3, 32, 32)
            transform_duration = 0
            cache_hit = True
            return tokens, transform_duration, cache_hit
        else:
            data = None
            if is_cached and self.cache_client is not None:
                data = self.fetch_from_cache(file_path)

            if data is not None and (isinstance(data, bytes) or isinstance(data, str)):
                transform_start_time = time.perf_counter()
                decoded_data = base64.b64decode(data) # Decode base64
                decoded_data = zlib.decompress(decoded_data)
                buffer = io.BytesIO(decoded_data)
                tokens = torch.load(buffer)
                transform_duration = time.perf_counter() - transform_start_time
                return tokens, transform_duration, cache_hit
            else:
                cache_hit = False
                if self.is_s3:
                    text = s3utils.get_s3_object(self.bucket_name, file_path)
                else:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                 # Apply transform if provided
                transform_start_time = time.perf_counter()
                if self.transform:
                    text = self.transform(text)  
                tokens = self.tokenizer(text, truncation=False, padding=False, return_tensors='pt').input_ids.squeeze()
                transform_duration = time.perf_counter() - transform_start_time
                return tokens, transform_duration, cache_hit


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