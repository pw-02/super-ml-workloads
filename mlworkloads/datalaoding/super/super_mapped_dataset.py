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
import base64
import zlib
import redis
from io import BytesIO

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


class SUPERMappedDataset(Dataset):
    def __init__(self, s3_data_dir: str, transform=None, cache_address= None):
        self.s3_bucket = S3Url(s3_data_dir).bucket
        self.s3_prefix = S3Url(s3_data_dir).key
        self.s3_data_dir = s3_data_dir
        self.transform = transform
        self.samples = self._get_sample_list_from_s3()
        # self.simulate_mode = simulate_mode
        # self._simlute_time_for_cache_miss = simulate_time_for_cache_miss
        # self._simlute_time_for_cache_hit = simulate_time_for_cache_hit

        if cache_address is not None:
            self.cache_host, self.cache_port = cache_address.split(":")
            self.cache_port = int(self.cache_port)
            self.use_cache = True
        else:
            self.use_cache = False

        self.cache_client = None

    
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


    def __len__(self) -> int:
        return sum(len(class_items) for class_items in self.samples.values())
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
        batch_id, batch_indices, is_cached = idx
        next_minibacth = None
        cached_after_fetch = False
        if is_cached and self.use_cache:
            fetch_start_time = time.perf_counter()
            next_minibacth = self.fetch_from_cache(batch_id)
            fetch_duration = time.perf_counter() - fetch_start_time

        if next_minibacth is not None and (isinstance(next_minibacth, bytes) or isinstance(next_minibacth, str)):
            tranform_start_time = time.perf_counter()
            data_samples, labels = self.decode_cached_minibacth(next_minibacth)
            transform_duration =  time.perf_counter() - tranform_start_time
            cache_hit = True
        else:
            fetch_start_time = time.perf_counter()
            data_samples, labels = self.fetch_batch_from_s3(batch_indices)
            fetch_duration = time.perf_counter() - fetch_start_time

            transform_start_time = time.perf_counter()
            if self.transform is not None:
                for i in range(len(data_samples)):
                    data_samples[i] = self.transform(data_samples[i])        
            transform_duration =  time.perf_counter() - transform_start_time
            cache_hit = False
            data_samples= torch.stack(data_samples)
            labels = torch.tensor(labels)
            if self.use_cache:
                try:
                    if self.cache_client is None:
                        self.cache_client:redis.StrictRedis = redis.StrictRedis(host=self.cache_host, port=self.cache_port)
                    with BytesIO() as buffer:
                        torch.save((data_samples, labels), buffer)
                        compressed_minibatch = zlib.compress(buffer.getvalue())
                        compressed_minibatch = base64.b64encode(compressed_minibatch).decode('utf-8')
                    
                    self.cache_client.set(batch_id, compressed_minibatch)
                    cached_after_fetch = True
                except Exception as e:
                    print(f"Error saving to cache: {e}")

        return (data_samples,labels), fetch_duration, transform_duration, cache_hit, cached_after_fetch


    def decode_cached_minibacth(self, cached_minibacth):
        decoded_data = base64.b64decode(cached_minibacth) # Decode base64
        decoded_data = zlib.decompress(decoded_data)
        buffer = io.BytesIO(decoded_data)
        batch_samples, batch_labels = torch.load(buffer)
        return  batch_samples, batch_labels


    def fetch_from_cache(self, batch_id):
        try:
            if self.cache_client is None:
                self.cache_client:redis.StrictRedis = redis.StrictRedis(host=self.cache_host, port=self.cache_port)
            return self.cache_client.get(batch_id)
        except Exception as e:
            print(f"Error fetching from cache: {e}")
            return None

    def fetch_batch_from_s3(self, batch_indices: List[str]) -> Tuple[List[torch.Tensor], List[int]]:
        s3_client = boto3.client('s3')

        data_samples = []
        labels = []
        for idx in batch_indices:
            data_path, label = self._classed_items[idx]
            obj = s3_client.get_object(Bucket=self.s3_bucket, Key=data_path)
            img_data = obj['Body'].read()
            image = Image.open(io.BytesIO(img_data)).convert('RGB')
            data_samples.append(image)
            labels.append(label)  # Simplified; adjust based on your label extraction
        return data_samples, labels


if __name__ == "__main__":
    # Example usage
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = SUPERMappedDataset(s3_data_dir="s3://sdl-cifar10/test/", transform=transform)
