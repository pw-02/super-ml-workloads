import os
import glob
from typing import List
import torch
from transformers import PreTrainedTokenizer
from typing import List, Tuple, Dict
import glob
import time
import redis
import io
from torch.utils.data import  IterableDataset, get_worker_info
from transformers import PreTrainedTokenizer
import logging
import functools
from urllib.parse import urlparse
import grpc
import proto.minibatch_service_pb2 as minibatch_service_pb2
import proto.minibatch_service_pb2_grpc as minibatch_service_pb2_grpc
import time
import boto3
import json
import lz4.frame

class S3Url(object):
    def __init__(self, url):
        self._parsed = urlparse(url, allow_fragments=False)

    @property
    def bucket(self):
        return self._parsed.netloc

    @property
    def key(self):
        if self._parsed.query:
            return self._parsed.path.lstrip('/') + '?' + self._parsed.query
        else:
            return self._parsed.path.lstrip('/')

    @property
    def url(self):
        return self._parsed.geturl()

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
                s3_data_dir: str, 
                tokenizer: PreTrainedTokenizer, 
                transform, 
                block_size: int, 
                batch_size: int, 
                grpc_server_address:str,
                world_size:int,
                cache_address:str = None,
                simulate_delay = None):
        super().__init__()
        self.job_id = str(job_id)
        self.grpc_server_address = grpc_server_address
        self.batch_size = batch_size
        self.total_batches = None
        self.s3_bucket = S3Url(s3_data_dir).bucket
        self.s3_prefix = S3Url(s3_data_dir).key
        self.s3_data_dir = s3_data_dir
        self.transform = transform
        self.samples: Dict[str, List[str]] = self._get_sample_list_from_s3(False, False)
        
        if cache_address is not None:
            self.cache_host, self.cache_port = cache_address.split(":")
            self.cache_port = int(self.cache_port)
            self.use_cache = True
        else:
            self.use_cache = False

        self.cache_client = None

        self.stub = self._create_grpc_stub()
        self._register_dataset_with_super()

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
        self.dataset_length = len(self)

    @functools.cached_property
    def _classed_items(self) -> List[Tuple[str, int]]:
        return [(blob, class_index)
            for class_index, blob_class in enumerate(self.samples)
            for blob in self.samples[blob_class]]
    
    def _get_sample_list_from_s3(self, use_index_file=True, images_only=True) -> Dict[str, List[str]]:
        s3_client = boto3.client('s3')

        index_file_key = f"{self.s3_prefix}_index.json"
        paired_samples = {}

        if use_index_file:
            try:
                index_object = s3_client.get_object(Bucket=self.s3_bucket, Key=index_file_key)
                file_content = index_object['Body'].read().decode('utf-8')
                paired_samples = json.loads(file_content)
                return paired_samples
            except Exception as e:
                print(f"Error reading index file '{index_file_key}': {e}")

        paginator = s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=self.s3_bucket, Prefix=self.s3_prefix):
            for blob in page.get('Contents', []):
                blob_path = blob.get('Key')
                
                if blob_path.endswith("/"):
                    continue  # Skip folders
                
                stripped_path = blob_path[len(self.s3_prefix):].lstrip("/")
                if stripped_path == blob_path:
                    continue  # No matching prefix, skip

                if images_only and not blob_path.lower().endswith(('.jpg', '.jpeg', '.png', 'json')):
                    continue  # Skip non-image files
                
                if 'index.json' in blob_path:
                    continue  # Skip index file

                blob_class = stripped_path.split("/")[0]
                if blob_class not in paired_samples:
                    paired_samples[blob_class] = []
                paired_samples[blob_class].append(blob_path)

        if use_index_file and paired_samples:
            s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=index_file_key,
                Body=json.dumps(paired_samples, indent=4).encode('utf-8')
            )

        return paired_samples

    def _create_grpc_stub(self):
        channel = grpc.insecure_channel(self.grpc_server_address)
        stub = minibatch_service_pb2_grpc.MiniBatchServiceStub(channel)
        return stub

    def _register_dataset_with_super(self):
        try:
            response = self.stub.RegisterDataset(minibatch_service_pb2.RegisterDatasetRequest(
                data_dir=self.s3_data_dir))
           
            print(f"{response.message}")
            self.total_batches = response.total_batches
        except grpc.RpcError as e:
            print(f"Failed to register dataset: {e.details()}")
            exit(1)     

    @functools.cached_property
    def _classed_items(self) -> List[Tuple[str, int]]:
        return [(blob, class_index)
                for class_index, blob_class in enumerate(self.samples)
                for blob in self.samples[blob_class]
                ]        
    def setup_cache_client(self):
        self.cache_client = redis.StrictRedis(host=self.cache_host, port=self.cache_port)

    def __len__(self):
        return sum(len(class_items) for class_items in self.samples.values())

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            return self.__iter_non_distributed__(len(self))
        else:
            #split workload
            num_workers = worker_info.num_workers
            per_worker = self.dataset_chunked_size // num_workers
            remaining_work = self.dataset_chunked_size % num_workers
            workloads = [per_worker + 1 if i < remaining_work else per_worker for i in range(num_workers)]
            return self.__iter_non_distributed__(workloads[worker_info.id])


    def __iter_non_distributed__(self, num_files):
        current_file_idx = 0
        current_tokens = None
        current_position = 0
        while current_file_idx < num_files:
            fetch_start_time = time.perf_counter()
            if current_tokens is None:
                current_tokens, data_loading_time, transformation_time, cache_hit, cached_after_fetch = self._load_next_file_tokens()
                current_file_idx +=1
                current_position = 0
            required_size = (self.block_size +1) * self.batch_size
            remaining_tokens = current_tokens[current_position:]

            while len(remaining_tokens) < required_size:
                next_tokens, data_loading_time, transformation_time, cache_hit, cached_after_fetch = self._load_next_file_tokens()
                current_file_idx +=1
                current_position = 0
                remaining_tokens = torch.cat((remaining_tokens, next_tokens), dim=0)
        
            batch_tokens = remaining_tokens[:required_size]
            current_position += required_size
            current_tokens = remaining_tokens[required_size:]

            data = batch_tokens[:(self.block_size +1) * self.batch_size].reshape(self.batch_size, (self.block_size+1))
            input_ids = data[:, 0 : self.block_size].contiguous().long()
            targets = data[:, 1 : (self.block_size + 1)].contiguous().long()
            
            yield input_ids, targets, data_loading_time, transformation_time, cache_hit, cached_after_fetch


        current_file_idx = 0
    
    def _load_next_file_tokens(self):
        response = self.stub.GetNextBatchForJob(minibatch_service_pb2.GetNextBatchForJobRequest(
                        job_id=self.job_id,
                        data_dir=self.s3_data_dir))          
        cache_key = response.batch.batch_id
        file_idx = list(response.batch.indicies)[0]
        is_cached = response.batch.is_cached
        file_content = None

        # Start data loading timer
        start_loading_time = time.perf_counter()
        if is_cached and self.cache_client is not None:
            file_content = self._load_batch_from_cache(cache_key)
        # If data is fetched from cache and it's in the correct format
        if file_content  is not None and (isinstance(file_content , bytes) or isinstance(file_content , str)):
            start_transformation_time   = time.perf_counter()
            tokens = self._bytes_to_torch_batch(file_content)
            transformation_time  =  time.perf_counter() - start_transformation_time 
            cache_hit = True
        else:
            # Fetch data from S3
            file_content = self._load_batch_from_s3(file_idx)

            # Apply transformations if provided
            start_transformation_time = time.perf_counter()
            if self.transform is not None:
                tokens = self.transform(file_content)
            transformation_time =  time.perf_counter() - start_transformation_time
            cache_hit = False

            # Convert to tensors
            tokens = torch.stack(tokens)
             # Cache the data if enabled
            if self.use_cache:
                try:
                    self._initialize_cache_client()
                    tokens_as_bytes = self._torch_batch_to_bytes(tokens)
                    self.cache_client.set(cache_key, tokens_as_bytes)
                    cached_after_fetch = True
                except Exception as e:
                    print(f"Error saving to cache: {e}")

        # Calculate data loading time excluding transformation time
        data_loading_time  = time.perf_counter() - start_loading_time - transformation_time
        return tokens, data_loading_time, transformation_time, cache_hit, cached_after_fetch


    def _torch_batch_to_bytes(self, samples: torch.Tensor) -> str:
        with io.BytesIO() as buffer:
            torch.save(samples, buffer)
            bytes_minibatch = buffer.getvalue()
            bytes_minibatch = lz4.frame.compress(bytes_minibatch)
        return bytes_minibatch
    
    def _bytes_to_torch_batch(self, bytes_minibatch) -> tuple:
        compressed_batch = lz4.frame.decompress(bytes_minibatch)
        with io.BytesIO(compressed_batch) as buffer:
            data_samples, labels = torch.load(buffer)
        return data_samples, labels

    def _load_batch_from_cache(self, batch_id):
        try:
            self._initialize_cache_client()   
            return self.cache_client.get(batch_id)
        except Exception as e:
            print(f"Error fetching from cache: {e}")
            return None
        
    def _load_batch_from_s3(self, idx) -> Tuple[List[torch.Tensor], List[int]]:
        s3_client = boto3.client('s3')
        data_path, _ = self._classed_items[idx]
        obj = s3_client.get_object(Bucket=self.s3_bucket, Key=data_path)
        content = obj['Body'].read().decode('utf-8')
        return content
   
    
# Example Usage
if __name__ == "__main__":
    from transformers import GPT2Tokenizer
    from torch.utils.data import DataLoader

    data_dir = 's3://owtchunks/train/'  # Adjust the path to your .txt files
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    block_size = 512
    batch_size = 4

    dataset = SUPERTextDataset(
        job_id=1,
        s3_data_dir=data_dir,
        tokenizer=tokenizer,
        transform=None,
        block_size=block_size,
        batch_size=batch_size,
        grpc_server_address="localhost:50051",
        world_size=1,
        cache_address=None,
        simulate_delay=None)
        
    dataloader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=0)

    for input_ids, targets, fetch, transform in dataloader:
        print(f"Input IDs: {input_ids.shape}")
        print(f"targets: {targets.shape}")

        print(f"fetch: {fetch}")
        print(f"transform: {transform}")

        # break
