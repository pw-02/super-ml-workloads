from PIL import Image
from torch.utils.data import Dataset
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
from pathlib import Path

class SUPERVisionDataset(Dataset):
    
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
            self.samples: Dict[str, List[str]] = self.load_s3_samples(S3Url(data_dir))
            self.bucket_name = S3Url(data_dir).bucket
        else:
            self.samples: Dict[str, List[str]] = self.load_local_sampels(data_dir)
        
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

    def __len__(self):
        return sum(len(class_items) for class_items in self.samples.values())
    
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

            file_path, label = self._classed_items[idx]

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
    
    def load_s3_samples(self, s3url: S3Url) -> Dict[str, List[str]]:
        s3_client = boto3.client('s3')
        s3_resource = boto3.resource("s3")

        try:
            # Check if 'prefix' folder exists
            response = s3_client.list_objects(Bucket=s3url.bucket, Prefix=s3url.key, Delimiter='/', MaxKeys=1)
            if 'NextMarker' not in response:
                # 'prefix' dir not found. Skipping task
                return None

            # Check if index file in the root of the folder to avoid looping through the entire bucket
            index_object = s3_resource.Object(s3url.bucket, s3url.key + 'index.json')
            try:
                file_content = index_object.get()['Body'].read().decode('utf-8')
                blob_classes = json.loads(file_content)
            except:
                # No index file found, creating it
                blob_classes = self._create_index_file_s3(s3url)

            return blob_classes
        except Exception as e:
            # Handle exceptions, e.g., log them
            print(f"Error in _classify_blobs_s3: {e}")
            return None


    def _create_index_file_s3(self, s3url: S3Url) -> Dict[str, List[str]]:
        import json

        s3_client = boto3.client('s3')
        s3_resource = boto3.resource("s3")

        blob_classes: Dict[str, List[str]] = {}
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=s3url.bucket, Prefix=s3url.key)

        for page in pages:
            for blob in page['Contents']:
                blob_path = blob.get('Key')
                # Check if the object is a folder which we want to ignore
                if blob_path[-1] == "/":
                    continue
                stripped_path = self._remove_prefix(blob_path, s3url.key).lstrip("/")
                # Indicates that it did not match the starting prefix
                if stripped_path == blob_path:
                    continue
                if not self.is_image_file(blob_path):
                    continue
                blob_class = stripped_path.split("/")[0]
                blobs_with_class = blob_classes.get(blob_class, [])
                blobs_with_class.append(blob_path)
                blob_classes[blob_class] = blobs_with_class

        index_object = s3_resource.Object(s3url.bucket, s3url.key + 'index.json')
        index_object.put(Body=(bytes(json.dumps(blob_classes, indent=4).encode('UTF-8'))))

        return blob_classes
    
    def _remove_prefix(self,s: str, prefix: str) -> str:
        if not s.startswith(prefix):
            return s
        return s[len(prefix) :]
    def is_image_file(self, filename: str):
        return any(filename.endswith(extension) for extension in self.img_extensions)
    
    def load_local_sampels(self, data_dir) -> Dict[str, List[str]]:
        import os
        data_dir = str(Path(data_dir))
        img_classes: Dict[str, List[str]] = {}
        index_file = Path(data_dir) / 'index.json'

        if index_file.exists():
            with open(index_file.absolute()) as f:
                img_classes = json.load(f)
        else:
            for dirpath, dirnames, filenames in os.walk(data_dir):
                for filename in filter(self.is_image_file, filenames):
                    img_class = os.path.basename(dirpath.removesuffix('/'))
                    img_path = os.path.join(dirpath, filename)
                    img_classes.setdefault(img_class, []).append(img_path)
            json_object = json.dumps(img_classes, indent=4)
            with open(index_file, "w") as outfile:
                outfile.write(json_object)
        return img_classes


