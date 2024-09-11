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
from super_sampler import SUPERSampler
import redis
from io import BytesIO
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
    
    def __iter__(self):
        """
        Provides an iterable interface over the dataset using the SUPERSampler.
        Each iteration fetches a batch of data using gRPC calls through the sampler.
        """
        current_tokens = None
        current_position = 0
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0

        while True:
            if current_tokens is None:
                # Load next file and initialize tokens
                current_tokens, transform_duration = self._load_next_file()
                current_position = 0
                # Get next batch indices using SUPERSampler
                
    def _load_next_file(self):
        batch_id, batch_indices, is_cached = next(self.sampler)
        next_minibatch  = None
        cached_after_fetch = False

        # Check cache if caching is enabled
        if is_cached and self.use_cache:
            next_minibatch = self._load_batch_from_cache(batch_id)

        # If data is fetched from cache and it's in the correct format
        if next_minibatch  is not None and (isinstance(next_minibatch , bytes) or isinstance(next_minibatch , str)):
            tokens = self._bytes_to_torch_batch(next_minibatch)
            cache_hit = True
        else:
            # Fetch data from S3
            text_data = self._load_text_from_s3(self._classed_items[batch_indices[0]])

            # Apply transformations if provided
            if self.transform is not None:
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



            cache_hit = False
            
            # Convert to tensors
            data_samples= torch.stack(data_samples)
            labels = torch.tensor(labels)
            
            # Cache the data if enabled
            if self.use_cache:
                try:
                    self._initialize_cache_client()
                    batch_as_bytes = self._torch_batch_to_bytes(data_samples, labels)
                    self.cache_client.set(batch_id, batch_as_bytes)
                    cached_after_fetch = True
                except Exception as e:
                    print(f"Error saving to cache: {e}")
    
    def _load_text_from_s3(self, data_path) -> Tuple[List[torch.Tensor], List[int]]:
        s3_client = boto3.client('s3')
        obj = s3_client.get_object(Bucket=self.s3_bucket, Key=data_path)
        content = obj['Body'].read().decode('utf-8')
        return content


    def _load_batch_from_cache(self, batch_id):
        try:
            self._initialize_cache_client()   
            return self.cache_client.get(batch_id)
        except Exception as e:
            print(f"Error fetching from cache: {e}")
            return None

    def _torch_batch_to_bytes(self, data_samples: torch.Tensor, labels: torch.Tensor) -> str:
        with BytesIO() as buffer:
            torch.save((data_samples, labels), buffer)
            bytes_minibatch = buffer.getvalue()
            bytes_minibatch = lz4.frame.compress(bytes_minibatch)
        return bytes_minibatch
    
    def _bytes_to_torch_batch(self, bytes_minibatch) -> tuple:
        compressed_batch = lz4.frame.decompress(bytes_minibatch)
        with BytesIO(compressed_batch) as buffer:
            data_samples, labels = torch.load(buffer)
        return data_samples, labels







    
    def _load_batch(self, batch_indices):
        """
        Loads and processes a batch of data from S3 given a set of batch indices.
        """
        input_ids_list = []
        targets_list = []
        transform_start_time = time.perf_counter()

        # Fetch and tokenize data from S3 based on indices
        for index in batch_indices:
            text_data = self._fetch_from_s3(index)
            if text_data:
                tokens = self.tokenizer.encode(text_data, add_special_tokens=False)
                # Padding or truncating to the block size
                if len(tokens) < self.block_size + 1:
                    tokens = tokens + [self.tokenizer.pad_token_id] * (self.block_size + 1 - len(tokens))
                else:
                    tokens = tokens[:self.block_size + 1]

                input_ids_list.append(tokens[:-1])  # Input tokens
                targets_list.append(tokens[1:])  # Target tokens

        transform_duration = time.perf_counter() - transform_start_time
        # Convert to tensors
        input_ids = torch.tensor(input_ids_list, dtype=torch.long)
        targets = torch.tensor(targets_list, dtype=torch.long)

        return input_ids, targets, transform_duration



       

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