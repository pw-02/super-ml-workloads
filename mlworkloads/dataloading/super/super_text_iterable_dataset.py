import os
import glob
from typing import List
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
from typing import List, Tuple, Dict
import glob
import time
import redis
import io
import base64
import zlib
from torch.utils.data import SequentialSampler, IterableDataset, RandomSampler, DataLoader, get_worker_info
from transformers import PreTrainedTokenizer
import logging
import functools
import uuid
import proto.minibatch_service_pb2 as minibatch_service_pb2
import proto.minibatch_service_pb2_grpc as minibatch_service_pb2_grpc
import grpc
from urllib.parse import urlparse
import json
import boto3

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
    
    
class SUPERTextDataset(IterableDataset):
    def __init__(self,
                 s3_data_dir: str,
                 sampler: SUPERSampler,
                 tokenizer: PreTrainedTokenizer, 
                 block_size: int, 
                 batch_size: int,
                 transform=None):
        super().__init__()
        self.s3_bucket = S3Url(s3_data_dir).bucket
        self.s3_prefix = S3Url(s3_data_dir).key
        self.transform = transform
        self.sampler = sampler
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.batch_size = batch_size
        self.s3_client = boto3.client('s3')


    def _test_grpc_connection(self):
        try:
            # Example ping to test connection
            self.stub.Ping(minibatch_service_pb2.PingRequest(message='ping'))
            print("Connection to SUPER server confirmed. Registering as a client...")
        except grpc.RpcError as e:
            print(f"Failed to connect to SUPER server: {e.details()}")
            exit(1)

    def _create_grpc_stub(self):
        channel = grpc.insecure_channel(self.grpc_server_address)
        stub = minibatch_service_pb2_grpc.MiniBatchServiceStub(channel)
        return stub
    
    def _register_dataset_with_super(self):
        try:
            response = self.stub.RegisterDataset(minibatch_service_pb2.RegisterDatasetRequest(
                data_dir=self.dataset.s3_data_dir))
           
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

        current_file_idx = 0
        current_tokens = None
        current_position = 0

        while current_file_idx < num_files:
            fetch_start_time = time.perf_counter()
            if current_tokens is None:
                current_tokens, transform_duration, cache_hit = self._load_next_file()
                current_file_idx +=1
                current_position = 0
            required_size = (self.block_size +1) * self.batch_size
            remaining_tokens = current_tokens[current_position:]

            while len(remaining_tokens) < required_size:
                next_tokens, transform_duration, cache_hit = self._load_next_file()
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
    
    def _load_next_file(self):
     
            response = self.stub.GetNextBatchForJob(minibatch_service_pb2.GetNextBatchForJobRequest(
            job_id=self.job_id,
            data_dir=self.dataset.s3_data_dir))
                    
            batch_id = response.batch.batch_id
            batch_indices = list(response.batch.indicies)
            is_cached = response.batch.is_cached
            self.current_batch += 1
        
        
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
                chunk_text = s3utils.get_s3_object(self.s3_bucket_name, data_path)
                transform_start_time = time.perf_counter()
                if self.transform:
                    chunk_text = self.transform(chunk_text)
                
                documents = chunk_text.split('\n')  # Split based on the specified delimiter 
                eos_token_id = tokenizer.eos_token_id
                all_tokens = []
                for doc in documents:
                    encoded_input = tokenizer(doc, truncation=False, padding=False, return_tensors='pt')
                    input_ids_with_eos = torch.cat([encoded_input.input_ids, torch.tensor([[eos_token_id]])], dim=1)            
                    # Append the input_ids_with_eos to the list
                    all_tokens.append(input_ids_with_eos)
                    # tokens = self.tokenizer(text, truncation=False, padding=False, return_tensors='pt').input_ids.squeeze()
                tokens = torch.cat(all_tokens, dim=0)
                transform_duration = time.perf_counter() - transform_start_time     
                return tokens, transform_duration, cache_hit

  
    def _get_sample_list_from_s3(self, use_index_file=False, images_only=False) -> Dict[str, List[str]]:
        s3_client = boto3.client('s3')

        index_file_key = f"{self.s3_prefix}_paired_index.json"
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

                if images_only and not blob_path.lower().endswith(('.jpg', '.jpeg', '.png')):
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


# Example Usage
if __name__ == "__main__":
    from transformers import AutoTokenizer
    data_dir = 's3://openwebtxt/owt/train/'  # Adjust the path to your .txt files
    tokenizer_name = "EleutherAI/pythia-70m"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    block_size = 512
    batch_size = 4

    dataset = SUPERTextDataset(data_dir, tokenizer, None, block_size, batch_size)
    dataloader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=0)

    for input_ids, targets, fetch, transform in dataloader:
        print(f"Input IDs: {input_ids.shape}")
        print(f"targets: {targets.shape}")

        print(f"fetch: {fetch}")
        print(f"transform: {transform}")