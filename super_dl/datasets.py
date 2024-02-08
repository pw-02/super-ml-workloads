from PIL import Image
from torchvision.datasets import ImageFolder
import boto3
from typing import Optional, List, Tuple, Callable, Dict
from .utils import S3Url, timer_decorator
import redis
import io
import torch
import zlib
import time
import base64
import functools
import json

class SUPERVisionDataset(ImageFolder):
    
    def __init__(self, 
                 job_id, 
                 data_dir: str,
                 transform=None, 
                 target_transform=None,
                 cache_address=None):
        
        self.job_id = job_id
        self.data_dir = data_dir
        self.use_s3 = self.data_dir.startswith("s3://")
        self.transform = transform
        self.dataset_id = self.generate_id()
        self.target_transform = None
        self.img_extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']
        self.s3_client = boto3.client('s3') if self.use_s3 else None
        self.cache_client = redis.StrictRedis(host=cache_address, port=cache_address) if cache_address is not None else None

        if self.use_s3:
            self.samples: Dict[str, List[str]] = self._classify_samples_s3(S3Url(data_dir))
            self.bucket_name = S3Url(data_dir).bucket
        else:
            super(SUPERVisionDataset, self).__init__(data_dir, transform=transform, target_transform=target_transform)
            self.samples = self.imgs
        
    @functools.cached_property
    def _classed_items(self) -> List[Tuple[str, int]]:
        return [
            (blob, class_index)
            for class_index, blob_class in enumerate(self.samples)
            for blob in self.samples[blob_class]
        ]
    
    def generate_id(self):
        import hashlib
        import json
        dataset_info = {"data_dir": self.data_dir, "transformations": self.convert_transforms_to_dict(),}
        serialized_info = json.dumps(dataset_info, sort_keys=True)
        sha256_hash = hashlib.sha256(serialized_info.encode()).digest()
        truncated_hash = sha256_hash[:8]
        # Encode the truncated hash in Base64 for a shorter representation
        dataset_id = base64.urlsafe_b64encode(truncated_hash).decode()
        return dataset_id

    def __getitem__(self, next_batch):
        indices, batch_id,cache_status = next_batch
        cached_data = None

        if self.cache_client is not None:
            cached_data, fetch_time = self.fetch_from_cache(batch_id, cache_status)
            
        if cached_data:
            # Convert JSON batch to torch format
            torch_imgs, torch_labels, transform_time = self.deserialize_torch_batch(cached_data)
            # print('data returned') 
            return torch_imgs, torch_labels, batch_id, True, fetch_time, transform_time
  
        images, labels, fetch_time = self.fetch_data(indices)
        images, labels, transform_time = self.apply_transformations(images, labels)
        return torch.stack(images), torch.tensor(labels), batch_id, False, fetch_time, transform_time

    @timer_decorator
    def fetch_data(self, indices):
        images = []
        labels = []
        for idx in indices:

            file_path, label = self.samples[idx]

            if self.use_s3:
                # Download file into memory
                obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=file_path)
                content = obj['Body'].read()
                img = Image.open(io.BytesIO(content))
            else:
                img = Image.open(file_path)

            if img.mode == "L":
                img = img.convert("RGB")
            images.append(img)
            labels.append(label)
        return images, labels
    
    @timer_decorator
    def apply_transformations(self, images, labels):
        if self.transform is not None:
            for i in range(len(images)):
                images[i] = self.transform(images[i])

        if self.target_transform is not None:
            for i in range(len(labels)):
                labels[i] = self.target_transform(labels[i])
        return images, labels
    
    @timer_decorator
    def fetch_from_cache(self, batch_id, cache_status, max_attempts = 5):
        cached_data = None
        attempts = 0
         # Try fetching from cache initially
        cached_data = self.try_fetch_from_cache(batch_id)
        if cached_data is not None:
            return cached_data
        # If not found in cache, check batch status with super_client
        # cache_status = self.super_client.get_batch_status(batch_id=batch_id, dataset_id=self.dataset_id)
        # cache_status = False
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
    
    def convert_transforms_to_dict(self):
        transform_dict = {}
        for tf in self.transform.transforms:
            transform_name = tf.__class__.__name__
            if transform_name == 'Resize':
                transform_dict[transform_name] = tf.size
            elif transform_name == 'Normalize':
                transform_dict[transform_name] = {'mean': tf.mean, 'std': tf.std}
            else:
                transform_dict[transform_name] = None
        return json.dumps(transform_dict)



    def convert_transforms_to_json_dict(self):
        return json.dumps(self.convert_transforms_to_dict())



