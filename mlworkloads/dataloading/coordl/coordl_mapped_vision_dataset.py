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


class CoorDLMappedVisionDataset(Dataset):
    def __init__(self, s3_data_dir: str, transform=None, cache_address= None, wss=1.0, max_dataset_size=None):
        self.wss = wss
        self.s3_bucket = S3Url(s3_data_dir).bucket
        self.s3_prefix = S3Url(s3_data_dir).key
        self.s3_data_dir = s3_data_dir
        self.transform = transform
        if cache_address is not None:
            self.cache_host, self.cache_port = cache_address.split(":")
            self.cache_port = int(self.cache_port)
            self.use_cache = True
        else:
            self.use_cache = False
        
        self.cache_client = None
        self.s3_client = None
        self.samples = self._get_sample_list_from_s3(use_index_file=True, images_only=True, max_dataset_size = max_dataset_size)
        self.cache_portion = self.wss * len(self)
        self.cache_portion = int(self.cache_portion // 1)

        #trim samples to match size of cache portion

    @functools.cached_property
    def _classed_items(self) -> List[Tuple[str, int]]:
        return [(blob, class_index)
                for class_index, blob_class in enumerate(self.samples)
                for blob in self.samples[blob_class]]
    
    def get_num_items_in_cache(self):
        if self.cache_client is None:
            self.cache_client = redis.StrictRedis(host=self.cache_host, port=self.cache_port,  ssl=True)
        return self.cache_client.dbsize()
    
    def _get_sample_list_from_s3(self, use_index_file=True, images_only=True, max_dataset_size = None) -> Dict[str, List[str]]:
        s3_client = boto3.client('s3')
        if max_dataset_size:
            index_file_key = f"{self.s3_prefix}_paired_index_{max_dataset_size}GB.json"
        else:
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
        total_size_gb = 0

        for page in paginator.paginate(Bucket=self.s3_bucket, Prefix=self.s3_prefix):
            if max_dataset_size and total_size_gb >= max_dataset_size:
                    break
            for blob in page.get('Contents', []):
                if max_dataset_size and total_size_gb >= max_dataset_size:
                    break
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
                total_size_gb += blob['Size'] / 1024 / 1024 / 1024

        if use_index_file and paired_samples:
            s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=index_file_key,
                Body=json.dumps(paired_samples, indent=4).encode('utf-8')
            )

        return paired_samples
    
    def _initialize_cache_client(self):
        """Initialize Redis cache client if not already connected."""
        if self.cache_client is None:
            self.cache_client = redis.StrictRedis(host=self.cache_host, port=self.cache_port, ssl=True)
    
    def _load_item_from_cache(self, key):
        try:
            self._initialize_cache_client()   
            return self.cache_client.get(key)
        except Exception as e:
            print(f"Error fetching from cache: {e}")
            return None
    
    def __len__(self) -> int:
        return sum(len(class_items) for class_items in self.samples.values())
    
    def fetch_image_from_s3(self, data_path):  
        if self.s3_client is None:
            self.s3_client = boto3.client('s3')
        obj = self.s3_client.get_object(Bucket=self.s3_bucket, Key=data_path)
        img_data = obj['Body'].read()
        image = Image.open(io.BytesIO(img_data)) #.convert('RGB')
        return image
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
        path, target = self._classed_items[index]
        item_data  = None
        cached_after_fetch = False
        start_loading_time = time.perf_counter()

        if self.use_cache:
            item_data = self._load_item_from_cache(path)

        if item_data  is not None and (isinstance(item_data , bytes) or isinstance(item_data , str)):
            start_transformation_time   = time.perf_counter()
            byteImgIO = io.BytesIO(item_data)
            sample = Image.open(byteImgIO)
            sample = sample.convert('RGB')

            if self.transform is not None:
                sample = self.transform(sample)
            transformation_time = time.perf_counter() - start_transformation_time
            cache_hit = True
            fetch_duration  = time.perf_counter() - start_loading_time - transformation_time
            return (sample, target), fetch_duration, transformation_time, cache_hit, cached_after_fetch
            
        sample = self.fetch_image_from_s3(path)
        cache_hit = False

        keys_cnt = self.get_num_items_in_cache()
        
        if self.use_cache and keys_cnt <= self.cache_portion:
            byte_stream = io.BytesIO()
            sample.save(byte_stream, format=sample.format)
            byte_stream.seek(0)
            byte_image = byte_stream.read()
            try:
                self.cache_client.set(path, byte_image)
                cached_after_fetch = True
            except Exception as e:
                pass
        sample = sample.convert('RGB')
 
        
        transform_start_time = time.perf_counter()
        if self.transform is not None:
            sample = self.transform(sample)
        transformation_time = time.perf_counter() - transform_start_time
        
        fetch_duration  = time.perf_counter() - start_loading_time - transformation_time
        
        return (sample, target), fetch_duration, transformation_time, cache_hit, cached_after_fetch


if __name__ == "__main__":
    # Example usage
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = CoorDLMappedVisionDataset(s3_bucket="sdl-cifar10", s3_prefix="train/", transform=transform)
    img, label = dataset[0]
    print(img.shape)