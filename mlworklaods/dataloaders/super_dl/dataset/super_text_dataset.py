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
import functools

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

class SUPERTextDataset(IterableDataset):

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
        self.samples: Dict[str, List[str]] = s3utils.load_paired_s3_object_keys(data_dir, False, False)
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
        # if self.is_s3:
        #     self.file_list: Dict[str, List[str]] = s3utils.load_unpaired_s3_object_keys(data_dir, False, True)
        #     self.bucket_name = S3Url(data_dir).bucket
        # else:
        #     self.file_list = glob.glob(f'{self.data_dir}/*.txt')

    @functools.cached_property
    def _classed_items(self) -> List[Tuple[str, int]]:
        return [(blob, class_index)
                for class_index, blob_class in enumerate(self.samples)
                for blob in self.samples[blob_class]
                ]        
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
            # fetch_duration = time.perf_counter() - fetch_start_time
            
            if cache_hit:
                cache_hits = 1
            else:
                cache_hits = 0

            data = batch_tokens[:(self.block_size +1) * self.batch_size].reshape(self.batch_size, (self.block_size+1))
            input_ids = data[:, 0 : self.block_size].contiguous().long()
            targets = data[:, 1 : (self.block_size + 1)].contiguous().long()
            fetch_duration = time.perf_counter() - fetch_start_time
            yield input_ids, targets, fetch_duration, transform_duration, cache_hits

            # yield batch_tokens,fetch_duration, transform_duration, cache_hits

        current_file_idx = 0
    
    def _load_next_file(self, super_client:SuperClient):
        next_data = super_client.get_next_batch(self.job_id)

        if not next_data:
            raise ValueError("Empty file returned by super_client.")
        
        next_data = next_data[0]
        data_id = next_data.batch_id
        data_path, label = self._classed_items[next_data.indicies[0]]
        is_cached = next_data.is_cached
        data = None

        if self.simulate_delay:
            time.sleep(self.simulate_delay)
            tokens = torch.empty(len(self.block_size*self.batch_size))
            transform_duration = 0
            cache_hit = True
            return tokens, transform_duration, cache_hit
        else:
            data = None
            if is_cached and self.cache_client is not None:
                data = self.fetch_from_cache(data_id)

            if data is not None and (isinstance(data, bytes) or isinstance(data, str)):
                transform_start_time = time.perf_counter()
                decoded_data = base64.b64decode(data) # Decode base64
                decoded_data = zlib.decompress(decoded_data)
                buffer = io.BytesIO(decoded_data)
                tokens = torch.load(buffer)
                transform_duration = time.perf_counter() - transform_start_time
                cache_hit = True
                return tokens, transform_duration, cache_hit
            else:
                cache_hit = False
                text = s3utils.get_s3_object(self.s3_bucket_name, data_path)
                transform_start_time = time.perf_counter()
                if self.transform:
                    text = self.transform(text)  
                tokens = self.tokenizer(text, truncation=False, padding=False, return_tensors='pt').input_ids.squeeze()
                transform_duration = time.perf_counter() - transform_start_time
                return tokens, transform_duration, cache_hit

    def fetch_from_cache(self, data_id, max_attempts = 5):
        response = None
        attempts = 0
         # Try fetching from cache initially
        response = self.cache_get(data_id)
        if response is not None and not isinstance(response, Exception):
            return response
        else:
            # Retry fetching from cache for a specified number of attempts
            while attempts < max_attempts:
                logger.error(f"Error fetching batch '{data_id}' from cache: {response}. Retrying (attempt {attempts}).")
                response = self.cache_get(data_id)
                if response is not None and not isinstance(response, Exception):
                    break
                else:
                    time.sleep(0.01)
                attempts += 1
        return response
    
    def cache_get(self, data_id, max_attempts = 1):     
        try:
            return self.cache_client.get(data_id)
        except Exception as e:
             return e

# Example Usage
if __name__ == "__main__":
    from transformers import GPT2Tokenizer
    from mlworklaods.llm.data import TextTransformations

    data_dir = 's3://openwebtxt/owt/train/'  # Adjust the path to your .txt files
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    block_size = 512
    batch_size = 4

    transformations = TextTransformations()


    dataset = TorchLRUTextDataset(data_dir, tokenizer, None, block_size, batch_size)
    dataloader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=0)

    for input_ids, targets, fetch, transform in dataloader:
        print(f"Input IDs: {input_ids.shape}")
        print(f"targets: {targets.shape}")

        print(f"fetch: {fetch}")
        print(f"transform: {transform}")

        # break
