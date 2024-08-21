from PIL import Image
from torch.utils.data import Dataset
from typing import Optional, List, Tuple, Callable, Dict
import redis
import io
import torch
import zlib
import time
import base64
import functools
import json
from pathlib import Path
from super_dl.s3_tasks import S3Helper

def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        # Unpack result if it's a tuple
        if isinstance(result, tuple):
            return (*result, execution_time)
        else:
            return (result, execution_time)
    return wrapper

class BaseSUPERDataset(Dataset):
    def __init__(self, job_id, data_dir: str, use_s3: bool, cache_address=None):
        self.job_id = job_id
        self.data_dir = data_dir
        self.use_s3 = use_s3
        self.cache_client = redis.StrictRedis(host=cache_address, port=cache_address) if cache_address is not None else None
        self.dataset_id = self.generate_id()
        if self.use_s3:
            self.s3_helper = S3Helper()
            self.samples: Dict[str, List[str]] = self.s3_helper.get_s3_samples(data_dir, False, True)
            self.bucket_name = self.s3_helper.get_bucket_name(data_dir)
        else:
            self.samples: Dict[str, List[str]] = self.load_local_sampels(data_dir)
    
    @functools.cached_property
    def _classed_items(self) -> List[Tuple[str, int]]:
        return [(blob, class_index)
            for class_index, blob_class in enumerate(self.samples)
            for blob in self.samples[blob_class]]

    def generate_id(self):
        import hashlib
        sha256_hash = hashlib.sha256(self.data_dir.encode()).digest()
        truncated_hash = sha256_hash[:8]
        dataset_id = base64.urlsafe_b64encode(truncated_hash).decode()
        return dataset_id

    def __len__(self):
        return sum(len(class_items) for class_items in self.samples.values())
    
    def load_local_sampels(self, data_dir) -> Dict[str, List[str]]:
        import os
        data_dir = str(Path(data_dir))
        classed_samples: Dict[str, List[str]] = {}
        index_file = Path(data_dir) / 'index.json'

        if index_file.exists():
            with open(index_file.absolute()) as f:
                classed_samples = json.load(f)
        else:
            for dirpath, dirnames, filenames in os.walk(data_dir):
                for filename in filter(self.is_image_file, filenames):
                    img_class = os.path.basename(dirpath.removesuffix('/'))
                    img_path = os.path.join(dirpath, filename)
                    classed_samples.setdefault(img_class, []).append(img_path)
            json_object = json.dumps(classed_samples, indent=4)
            with open(index_file, "w") as outfile:
                outfile.write(json_object)
        return classed_samples
        
    @timer_decorator
    def fetch_from_cache(self, batch_id, cache_status, max_attempts = 5):
        cached_data = None
        attempts = 0
         # Try fetching from cache initially
        cached_data = self.try_fetch_from_cache(batch_id)
        if cached_data is not None:
            return cached_data

        if cache_status == False:  # If the status is False, not cached or in progress, return None and fetch locally
            return cached_data
        else:
            # Retry fetching from cache for a specified number of attempts
            while attempts < max_attempts:
                cached_data = self.try_fetch_from_cache(batch_id)
                if cached_data is not None:
                    break
                else:
                    time.sleep(0.01)
                attempts += 1
        return cached_data
    
    def try_fetch_from_cache(self, batch_id):
        try:
            return self.cache_client.get(batch_id)
        except:
             return None
    
    def deserialize_torch_batch(self, batch_data):
        decoded_data = base64.b64decode(batch_data) # Decode base64
        try:
            decoded_data = zlib.decompress(decoded_data)
        except:
            pass
      
        buffer = io.BytesIO(decoded_data)
        batch_samples, batch_labels = torch.load(buffer)
        return  batch_samples, batch_labels
    
    def is_image_file(self, filename: str):
        return any(filename.endswith(extension) for extension in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP'])
    