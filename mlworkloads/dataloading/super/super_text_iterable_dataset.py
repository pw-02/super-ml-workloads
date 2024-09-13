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
import numpy as np

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
                grpc_server_address:str,
                world_size:int,
                cache_address:str = None,
                shuffle:bool = False):
        super().__init__()
        self.job_id = str(job_id)
        self.grpc_server_address = grpc_server_address
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
                data_dir=self.s3_data_dir,
                dataset_kind='text'))
           
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
        current_chunk_id = None
        tokenized_samples = [] 
        while current_file_idx < num_files:
            start = time.perf_counter()

            if not tokenized_samples or len(tokenized_samples) == 0:
                current_chunk_id, tokenized_samples, data_loading_time, transformation_time, cache_hit, cached_after_fetch = self._load_next_chunk_samples()
                current_file_idx +=1
                input_ids, targets = tokenized_samples.pop(0)
                yield (input_ids, targets, current_chunk_id), data_loading_time, transformation_time, cache_hit, cached_after_fetch
            else:
                input_ids, targets = tokenized_samples.pop(0)
                yield (input_ids, targets, current_chunk_id), time.perf_counter() - start, 0, True, False
           
        current_file_idx = 0
    
    def _load_next_chunk_samples(self):
        response = self.stub.GetNextBatchForJob(minibatch_service_pb2.GetNextBatchForJobRequest(
                        job_id=self.job_id,
                        data_dir=self.s3_data_dir))          
        
        chunk_id = response.batch.batch_id
        chunk_idx = list(response.batch.indicies)[0]
        is_cached = response.batch.is_cached
        tokenized_samples = None
        cached_after_fetch = False
        # Start data loading timer
        start_loading_time = time.perf_counter()
        if is_cached and self.cache_client is not None:
            tokenized_samples = self._load_batch_from_cache(chunk_id)

        # If data is fetched from cache and it's in the correct format
        if tokenized_samples  is not None and (isinstance(tokenized_samples , bytes) or isinstance(tokenized_samples , str)):
            start_transformation_time   = time.perf_counter()
            tokenized_samples = self._bytes_to_torch_batch(tokenized_samples)
            transformation_time  =  time.perf_counter() - start_transformation_time 
            cache_hit = True
        else:
            # Fetch data from S3
            data_chunk = self._load_chunk_from_s3(chunk_idx)

            start_transformation_time = time.perf_counter()
            tokenized_chunk = self.tokenize_data_chunk(io.BytesIO(data_chunk))
            tokenized_samples = self._gen_samples(tokenized_chunk)
            transformation_time =  time.perf_counter() - start_transformation_time
            cache_hit = False
             # Cache the data if enabled
            if self.use_cache:
                try:
                    self._initialize_cache_client()
                    tokens_as_bytes = self._torch_tenors_to_bytes(tokenized_samples)
                    self.cache_client.set(chunk_id, tokens_as_bytes)
                    cached_after_fetch = True
                except Exception as e:
                    print(f"Error saving to cache: {e}")

        # Calculate data loading time excluding transformation time
        data_loading_time  = time.perf_counter() - start_loading_time - transformation_time
        return chunk_id, tokenized_samples, data_loading_time, transformation_time, cache_hit, cached_after_fetch
    
    
    def _initialize_cache_client(self):
        """Initialize Redis cache client if not already connected."""
        if self.cache_client is None:
            self.cache_client = redis.StrictRedis(host=self.cache_host, port=self.cache_port)


    def _torch_tenors_to_bytes(self, tokenized_data: torch.Tensor) -> str:
        with io.BytesIO() as buffer:
            torch.save(tokenized_data, buffer)
            compressed_byte_data = lz4.frame.compress(buffer.getvalue())
        return compressed_byte_data
    
    def _bytes_to_torch_batch(self, bytes_minibatch) -> tuple:
        compressed_batch = lz4.frame.decompress(bytes_minibatch)
        with io.BytesIO(compressed_batch) as buffer:
            samples = torch.load(buffer)
        return samples

    def _load_batch_from_cache(self, batch_id):
        try:
            self._initialize_cache_client()   
            return self.cache_client.get(batch_id)
        except Exception as e:
            print(f"Error fetching from cache: {e}")
            return None
        
    def tokenize_data_chunk(self, data_chunk:io.BytesIO):
        tokenized_docs = []
        index = 0
        while True:
            offset = (1 + (index - 0) if index >= 0 else index + 1) * 4
            # Read the entire content of the binary file
            data_chunk.seek(offset)
            pair = data_chunk.read(8)
            begin, end = np.frombuffer(pair, np.uint32)
            if begin.item() == len(data_chunk.getvalue()):
                break
            data_chunk.seek(begin)
            raw_item_data = data_chunk.read(end - begin)

            shift_idx = 4
            sizes = np.frombuffer(raw_item_data[:shift_idx], np.uint32)
            data = ""
            for size, data_format in zip(sizes, 'str'):
                # size = size.item()
                data_bytes = raw_item_data[shift_idx : shift_idx + size]
                data += data_bytes.decode('utf-8')
                shift_idx += size
            index += 1
            tokenized_docs.append(self.tokenizer.encode(data, eos=True))
        return tokenized_docs
    

    def _gen_samples(self, tokenized_data):
        # Create a list to hold samples of size self.context_size
        samples = []
        for item in tokenized_data:
            chunk_size = self.block_size + 1  # Define the chunk size
            # Split ids into chunks of size block_size+1
            for i in range(0, item.size(0), chunk_size):
                # Extract a chunk from the ids
                chunk = item[i:i + chunk_size]
                # Pad the last chunk if it is smaller than block_size+1
                if chunk.size(0) < chunk_size:
                    padding_length = chunk_size - chunk.size(0)
                    padding = torch.full((padding_length,), fill_value=0, dtype=torch.long)
                    chunk = torch.cat((chunk, padding))

                input_ids = chunk[0:self.block_size].contiguous().long()
                targets = chunk[1:self.block_size + 1].contiguous().long()
                samples.append((input_ids, targets))

        return samples
    

    def _load_chunk_from_s3(self, idx) -> Tuple[List[torch.Tensor], List[int]]:
        s3_client = boto3.client('s3')
        data_path, _ = self._classed_items[idx]
        response = s3_client.get_object(Bucket=self.s3_bucket, Key=data_path)
        content = response['Body'].read()
        return content
    
    def send_job_update_to_super(self, 
                                 batch_id, 
                                 wait_for_data_time: float, 
                                 is_cache_hit: bool, 
                                 gpu_time: float, 
                                 cached_batch_on_miss: bool):
        try:
            self.stub.JobUpdate(minibatch_service_pb2.JobUpdateRequest(
                job_id=self.job_id,
                data_dir=self.s3_data_dir,
                previous_step_batch_id = batch_id,
                previous_step_wait_for_data_time = wait_for_data_time,
                previous_step_is_cache_hit = is_cache_hit,
                previous_step_gpu_time = gpu_time,
                cached_previous_batch = cached_batch_on_miss
                ))  
        except grpc.RpcError as e:
            print(f"Failed to send job update info to SUPER: {e.details()}")