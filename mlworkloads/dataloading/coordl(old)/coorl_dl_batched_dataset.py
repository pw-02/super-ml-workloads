from concurrent.futures import ThreadPoolExecutor, as_completed
import boto3
import io
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from typing import List, Dict, Tuple
import functools
import time
from urllib.parse import urlparse
import redis
from io import BytesIO
import lz4.frame
import botocore.config
# import zstandard as zstd
import sys
import pickle

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


class CoorDLBatchedDataset(Dataset):
    def __init__(self, s3_data_dir: str, transform=None, cache_address= None, cache_transformations=True):
        self.s3_bucket = S3Url(s3_data_dir).bucket
        self.s3_prefix = S3Url(s3_data_dir).key
        self.s3_data_dir = s3_data_dir
        self.s3_client = None
        self.transform = transform
        self.samples = self._get_sample_list_from_s3()
        self.cache_transformations = cache_transformations
        if cache_address is not None:
            self.cache_host, self.cache_port = cache_address.split(":")
            self.cache_port = int(self.cache_port)
            self.use_cache = True
        else:
            self.use_cache = False

        self.cache_client = None
        # self.compressor = None
        # self.decompressor = None
    
    def check_s3_client(self):
        if self.s3_client is None:
            self.s3_client = boto3.client('s3', config=botocore.config.Config(
                max_pool_connections=100))

    
    @functools.cached_property
    def _classed_items(self) -> List[Tuple[str, int]]:
        return [(blob, class_index)
            for class_index, blob_class in enumerate(self.samples)
            for blob in self.samples[blob_class]]
    
    def _get_sample_list_from_s3(self, use_index_file=True, images_only=True) -> Dict[str, List[str]]:
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
    
    def get_size_of_regular_object_as_bytes(self, obj) -> int:
        return sys.getsizeof(obj)
    
    def get_size_of_tensor_object_as_bytes(self, obj: torch.Tensor) -> int:
        return obj.element_size() * obj.nelement()

    def __len__(self) -> int:
        return sum(len(class_items) for class_items in self.samples.values())
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
        batch_id, batch_indices = idx
        next_minibatch  = None
        cached_after_fetch = False
        # self.set_compesor()
        # Start data loading timer
        start_loading_time = time.perf_counter()

        # Check cache if caching is enabled
        if self.use_cache:
            next_minibatch = self.get_cached_minibatch_with_retries(batch_id, max_retries=5)

        # If data is fetched from cache and it's in the correct format
        if next_minibatch  is not None and (isinstance(next_minibatch , bytes) or isinstance(next_minibatch , str)):
            start_transformation_time   = time.perf_counter()
            
            if self.cache_transformations == False:
                data_samples, labels  = pickle.loads(next_minibatch)
                if self.transform is not None:
                    for i in range(len(data_samples)):
                        data_samples[i] = self.transform(data_samples[i])
            else:
                data_samples, labels = self._bytes_to_torch_batch(next_minibatch)

            transformation_time  =  time.perf_counter() - start_transformation_time 
            cache_hit = True
        else:
            # Fetch data from S3
            data_samples, labels = self._load_batch_from_s3(batch_indices)
            
            if self.use_cache and self.cache_transformations == False:
                #in here we cache the data without transformations
                cache_value = pickle.dumps((data_samples, labels)) 
                print(f'cached object size (no transformations):{self.get_size_of_regular_object_as_bytes(cache_value)}')  
                cached_after_fetch = self.cache_minibatch_with_retries(batch_id, cache_value)    
                pass

            # Apply transformations if provided
            start_transformation_time = time.perf_counter()
            if self.transform is not None:
                for i in range(len(data_samples)):
                    data_samples[i] = self.transform(data_samples[i])        
            transformation_time =  time.perf_counter() - start_transformation_time
            cache_hit = False

            # Convert to tensors
            data_samples= torch.stack(data_samples)
            labels = torch.tensor(labels)
            
            if self.use_cache and self.cache_transformations:
                try:
                    self._initialize_cache_client()
                    batch_as_bytes = self._torch_batch_to_bytes(data_samples, labels)
                    print(f'cached object size (with transformations):{self.get_size_of_regular_object_as_bytes(batch_as_bytes)}')  
                    cached_after_fetch = self.cache_minibatch_with_retries(batch_id, batch_as_bytes)
                except Exception as e:
                    print(f"Error saving to cache: {e}, batch_id: {batch_id}")
        
        # Calculate data loading time excluding transformation time
        data_loading_time  = time.perf_counter() - start_loading_time - transformation_time
        
        return (data_samples,labels,batch_id), data_loading_time, transformation_time, cache_hit, cached_after_fetch
    
    def _initialize_cache_client(self):
        """Initialize Redis cache client if not already connected."""
        if self.cache_client is None:
            self.cache_client = redis.StrictRedis(host=self.cache_host, port=self.cache_port)
            # self.cache_client = redis.StrictRedis(host=self.cache_host, port=self.cache_port,  ssl=True)

    def _torch_batch_to_bytes(self, data_samples: torch.Tensor, labels: torch.Tensor) -> str:
        with BytesIO() as buffer:
            torch.save((data_samples, labels), buffer)
            bytes_minibatch = buffer.getvalue()
            # print(f"Serialized minibatch size: {sys.getsizeof(bytes_minibatch)} bytes")
            bytes_minibatch = lz4.frame.compress(bytes_minibatch,  compression_level=0)
            # bytes_minibatch = zlib.compress(bytes_minibatch,level=0)

            # print(f"Compressed minibatch size: {sys.getsizeof(bytes_minibatch)} bytes)")
            #bytes_minibatch = self.compressor.compress(bytes_minibatch)
        return bytes_minibatch
    
    def _bytes_to_torch_batch(self, bytes_minibatch) -> tuple:
        # time_start = time.perf_counter()
        bytes_minibatch = lz4.frame.decompress(bytes_minibatch)
        # compressed_batch = zlib.decompress(bytes_minibatch)
        # print(f"Decompression time: {time.perf_counter() - time_start}")
        # time_start = time.perf_counter()
        # bytes_minibatch = self.decompressor.decompress(bytes_minibatch)
        with BytesIO(bytes_minibatch) as buffer:
            data_samples, labels = torch.load(buffer)
        # print(f"Deserialization time: {time.perf_counter() - time_start}")
        return data_samples, labels
    
    def cache_minibatch_with_retries(self, batch_id, minibatch, max_retries=4, retry_interval=0.1):
        retries = 0
        while retries < max_retries:
            try:
                # Attempt to cache the minibatch in Redis
                self.cache_client.set(batch_id, minibatch)
                return True # Exit the function on success
            except Exception as e:
                print(f"Error saving to cache: {e}, batch_id: {batch_id}, retrying {retries}...")
            # Increment the retry count
            retries += 1
            # Wait before retrying
            time.sleep(retry_interval)
        return False
 
    
    def get_cached_minibatch_with_retries(self, batch_id, max_retries=4, retry_interval=0.05):
        self._initialize_cache_client()   
        retries = 0
        exception = None
        while retries < max_retries:
            try:
                # Attempt to cache the minibatch in Redis
                data = self.cache_client.get(batch_id)
                if data:
                    return data
            except Exception as e:
                exception = e
            # Increment the retry count
            retries += 1
            # Wait before retrying
            time.sleep(retry_interval)
        # print(f"Error fetching from cache: {exception}, batch_id: {batch_id}")


    def _load_batch_from_cache(self, batch_id):
        try:
            self._initialize_cache_client()   
            return self.cache_client.get(batch_id)
        except Exception as e:
            # print(f"Error fetching from cache: {e}, batch_id: {batch_id}")
            return None
    
    def _load_batch_from_s3(self, batch_indices: List[str]) -> Tuple[List[torch.Tensor], List[int]]:
        data_samples, labels = [], []
        self.check_s3_client()
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.get_data_sample, idx): idx for idx in batch_indices}
            for future in as_completed(futures):
                data_sample, label = future.result()
                data_samples.append(data_sample)
                labels.append(label)
        return data_samples, labels

    
    def get_data_sample(self,idx) -> tuple:  
        data_path, label = self._classed_items[idx]
        obj = self.s3_client.get_object(Bucket=self.s3_bucket, Key=data_path)
        data = Image.open(BytesIO(obj['Body'].read())).convert("RGB")
        return data, label



from torch.utils.data.sampler import BatchSampler, SequentialSampler, RandomSampler
from typing import Any, Dict, Iterable, Iterator, List, Optional, Union, Sized, Tuple
import hashlib
import torch
import random

class CoorDLBatchSampler(BatchSampler):
    def __init__(self, data_source: Sized, batch_size: int, drop_last: bool, shuffle: bool = False, seed: Optional[int] = None):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)

        # Precompute all indices based on the sampler
        self.sampler = RandomSampler(data_source) if shuffle else SequentialSampler(data_source)
        self.batches = self._create_batches()

    def _create_batches(self) -> List[List[int]]:
         # Create batches from the underlying sampler
        batch_list = []
        current_batch = []
        for idx in self.sampler:
            current_batch.append(idx)
            if len(current_batch) == self.batch_size:
                batch_list.append(current_batch)
                current_batch = []

        # Handle drop_last behavior
        if current_batch and not self.drop_last:
            batch_list.append(current_batch)

        return batch_list
    
    def __iter__(self) -> Iterator[Tuple[str, List[int]]]:
        # Shuffle batches if needed
        if self.shuffle:
            random.shuffle(self.batches)

        # Pop batches one by one and yield
        for batch_indices in self.batches:
            batch_id = hashlib.md5(str(batch_indices).encode()).hexdigest()
            yield (batch_id, batch_indices)

        # Regenerate batches after finishing an epoch
        self.batches = self._create_batches()
    
    def __len__(self) -> int:
        return len(self.batches)

# Example usage
if __name__ == "__main__":
    from torch.utils.data import RandomSampler
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms
    
    # Example usage
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


    # Example dataset size
    dataset_size = 1
    cooordl_dataset = CoorDLBatchedDataset(s3_data_dir="s3://imagenet1k-sdl/train/", 
                                           transform=transform,
                                           cache_address='127.0.0.1:6379',
                                           cache_transformations=True)

    # # Example usage of BatchSamplerWithID with shuffling
    batch_sampler_with_id = CoorDLBatchSampler(data_source=cooordl_dataset, batch_size=1, drop_last=False, shuffle=False, seed=42)
    dataloader = DataLoader(cooordl_dataset, sampler=batch_sampler_with_id, num_workers=0, batch_size=None)  # batch_size=None since sampler provides batches
    # batch_sampler_with_id = CoorDLBatchSampler(data_source=range(dataset_size),
    #                                            batch_size=10, drop_last=False,
    #                                              shuffle=True, seed=42)

    # Iterate over batches and print batch IDs and indices
    for batch_idx, (batch, data_load_time, transformation_time, is_cache_hit, cached_on_miss) in enumerate(dataloader):
        print(f"Batch ID: {batch_idx}, Batch Indices: {len(batch)}")

