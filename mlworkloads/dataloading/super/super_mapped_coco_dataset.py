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
import os
from torch.nn.utils.rnn import pad_sequence
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


class SUPERMappedCocoDataset(Dataset):
    def __init__(self, 
                 annotation_file: str,
                 s3_data_dir: str,
                 image_transform=None,
                 text_transform=None, 
                 cache_address= None):
        self.s3_bucket = S3Url(s3_data_dir).bucket
        self.s3_prefix = S3Url(s3_data_dir).key
        self.s3_data_dir = s3_data_dir
        self.annotation_file = S3Url(annotation_file).key
        self.image_transform = image_transform
        self.text_transform = text_transform

        if cache_address is not None:
            self.cache_host, self.cache_port = cache_address.split(":")
            self.cache_port = int(self.cache_port)
            self.use_cache = True
        else:
            self.use_cache = False
        self.cache_client = None
        self.s3_client = None
        self.simulate_mode = None
        self.samples = self._get_sample_list_from_s3()
        self._simlute_time_for_cache_hit = 0
        self._simlute_time_for_cache_miss = 0

    
    @functools.cached_property
    def _classed_items(self) -> List[Tuple[str, int]]:
        return [(blob, class_index)
            for class_index, blob_class in enumerate(self.samples)
            for blob in self.samples[blob_class]]
    
    def _get_sample_list_from_s3(self, use_index_file=True, images_only=True) -> Dict[str, List[str]]:
        s3_client = boto3.client('s3')
        index_object = s3_client.get_object(Bucket=self.s3_bucket, Key=self.annotation_file)
        file_content = index_object['Body'].read().decode('utf-8')
        # samples = json.loads(file_content)
        paired_samples = json.loads(file_content)
        return paired_samples

    def __len__(self) -> int:
        return sum(len(class_items) for class_items in self.samples.values())
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
        batch_id, batch_indices, is_cached = idx
        next_minibatch  = None
        cached_after_fetch = False

        # Start data loading timer
        start_loading_time = time.perf_counter()

        if self.simulate_mode:
            if is_cached:
                time.sleep(self._simlute_time_for_cache_hit)
            else:
                cached_after_fetch = True
                time.sleep(self._simlute_time_for_cache_miss)
            return batch_id, time.perf_counter() - start_loading_time, 0, is_cached, cached_after_fetch

        # Check cache if caching is enabled
        if is_cached and self.use_cache:
            next_minibatch = self._load_batch_from_cache(batch_id)

        # If data is fetched from cache and it's in the correct format
        if next_minibatch  is not None and (isinstance(next_minibatch , bytes) or isinstance(next_minibatch , str)):
            start_transformation_time   = time.perf_counter()
            images, text,text_atts,ids = self._bytes_to_torch_batch(next_minibatch )
            transformation_time  =  time.perf_counter() - start_transformation_time 
            cache_hit = True
        else:
            # Fetch data from S3
            image_list, text_list, image_id_list = self._load_batch_from_s3(batch_indices)
                    
            # Apply transformations if provided
            start_transformation_time = time.perf_counter()
            for i in range(len(image_list)):
                image_list[i] = self.image_transform(image_list[i])      
            
            for i in range(len(text_list)):
                text_list[i] = self.text_transform(text_list[i]) 

            transformation_time =  time.perf_counter() - start_transformation_time
            cache_hit = False

            # Convert to tensors
            images = torch.stack(image_list, dim=0)
            text = pad_sequence(text_list, batch_first=True)
            text_atts = (text != 0).type(torch.long)
            ids =  torch.Tensor(image_id_list).type(torch.long)
        
            # Cache the data if enabled
            if self.use_cache:
                try:
                    self._initialize_cache_client()
                    batch_as_bytes = self._torch_batch_to_bytes(images, text,text_atts,ids)
                    self.cache_client.set(batch_id, batch_as_bytes)
                    cached_after_fetch = True
                except Exception as e:
                    print(f"Error saving to cache: {e}")
        
        # Calculate data loading time excluding transformation time
        data_loading_time  = time.perf_counter() - start_loading_time - transformation_time
        
        return (images, text,text_atts,ids,batch_id), data_loading_time, transformation_time, cache_hit, cached_after_fetch
    
    def _initialize_cache_client(self):
        """Initialize Redis cache client if not already connected."""
        if self.cache_client is None:
            self.cache_client = redis.StrictRedis(host=self.cache_host, port=self.cache_port)

    def _torch_batch_to_bytes(self, images, text,text_atts,ids) -> str:
        with BytesIO() as buffer:
            mini_batch = images, text,text_atts,ids
            torch.save(mini_batch, buffer)
            bytes_minibatch = buffer.getvalue()
            bytes_minibatch = lz4.frame.compress(bytes_minibatch)
        return bytes_minibatch
    
    def _bytes_to_torch_batch(self, bytes_minibatch) -> tuple:
        compressed_batch = lz4.frame.decompress(bytes_minibatch)
        with BytesIO(compressed_batch) as buffer:
            images, text,text_atts,ids  = torch.load(buffer)
        return images, text,text_atts,ids 

    def _load_batch_from_cache(self, batch_id):
        try:
            self._initialize_cache_client()   
            return self.cache_client.get(batch_id)
        except Exception as e:
            print(f"Error fetching from cache: {e}")
            return None

    def _load_batch_from_s3(self, batch_indices: List[str]) -> Tuple[List[torch.Tensor], List[int]]:
        if self.s3_client is None:
            self.s3_client = boto3.client('s3')
        
        data_samples = []
        captions = []
        image_ids = []
        for idx in batch_indices:
            sample, image_id = self._classed_items[idx]
            data_path, caption = sample

            obj = self.s3_client.get_object(Bucket=self.s3_bucket, Key=data_path)
            img_data = obj['Body'].read()
            image = Image.open(io.BytesIO(img_data)).convert('RGB')
            # image = Image.open(io.BytesIO(img_data)).convert('RGB')
            data_samples.append(image)
            captions.append(caption)  # Simplified; adjust based on your label extraction
            image_ids.append(image_id)
        return data_samples, captions, image_ids


if __name__ == "__main__":
    # Example usage
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # dataset = SUPERMappedDataset(s3_data_dir="s3://sdl-cifar10/test/", transform=transform)
