from typing import Optional, List, Tuple, Callable, Dict
from PIL import Image
from torch.utils.data import Dataset
from .utils import S3Url
import torch
import functools
import base64
import io
import json
import os
from pathlib import Path
import boto3
import zlib
import time
from super_client import SuperClient


# Define constants
REDIS_PORT = 6379

class SUPERDataset(Dataset):
    def __init__(self, 
                 data_dir: str,
                 transform: Optional[Callable],
                 cache_client,
                 source_system,
                 s3_bucket_name = None,
                 super_client = None):
        
        self.dataset_id =  f"{source_system}_{data_dir}"
        self.cache_client = cache_client
        self.transform = transform
        self.data_dir = data_dir
        self.source_system = source_system
        self.img_extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']
        self.s3_bucket_name = s3_bucket_name
        self.super_client:SuperClient = super_client
        self.use_s3 = True if source_system == 's3' else False
        if self.use_s3:
            self.samples: Dict[str, List[str]] =  self._classify_samples_s3(S3Url(data_dir))
        else:
            self.samples: Dict[str, List[str]] =  self._classify_samples_local(data_dir)
        pass

    @functools.cached_property
    def _classed_items(self) -> List[Tuple[str, int]]:
        return [
            (blob, class_index)
            for class_index, blob_class in enumerate(self.samples)
            for blob in self.samples[blob_class]
        ]

    def __len__(self):
        return sum(len(class_items) for class_items in self.samples.values())

    def __getitem__(self, next_batch):
        batch_indices, batch_id = next_batch
        cached_data = None
        end = time.time()
        if self.cache_client is not None:
            cached_data = self.fetch_from_cache(batch_id)
            
        if cached_data:
            print(f"data fetch from cache {time.time() - end}")
            # Convert JSON batch to torch format
            decode_end = time.time()
            torch_imgs, torch_labels = self.deserialize_torch_batch(cached_data)
            print(f"data decode from cache {time.time() - decode_end}")

            return torch_imgs, torch_labels, batch_id, True
        
        print(f'cache miss: {batch_id}')

        if self.use_s3:
            # Cache miss, load from primary storage
            images, labels = self.fetch_batch_data_s3(batch_indices, batch_id)
        else:
            images, labels = self.fetch_batch_data_local(batch_indices, batch_id)
            print(f"data fetch {time.time() - end}")


        return torch.stack(images), torch.tensor(labels), batch_id, False
    

    def fetch_from_cache(self, batch_id, max_attempts = 10):
        cached_data = None
        attempts = 0

        while attempts < max_attempts:
            cached_data = self.try_fetch_from_cache(batch_id)
            if cached_data is not None:
                break  # Exit the loop if data is successfully fetched

            if attempts >= 0:
                    # Additional functionality on the second iteration
                    status = self.super_client.get_batch_status(batch_id, self.dataset_id)
                    if status == False:  # not cached or in progress, return none and fetch locally
                        break        
        attempts += 1
        return cached_data

    def try_fetch_from_cache(self, batch_id):
        try:
            return self.cache_client.get(batch_id)
        except:
             return None
    



    def is_image_file(self, filename:str):
        return any(filename.endswith(extension) for extension in self.img_extensions)

    def deserialize_torch_batch(self, batch_data):
        decoded_data = base64.b64decode(batch_data) # Decode base64
        try:
            decoded_data = zlib.decompress(decoded_data)
        except:
            pass
      
        buffer = io.BytesIO(decoded_data)
        batch_samples, batch_labels = torch.load(buffer)
        return  batch_samples, batch_labels 
    
    def convert_json_batch_to_torch_format(self, batch_data):
        samples = json.loads(batch_data)
        imgs = []
        labels = []

        for img, label in samples:
            img = Image.open(io.BytesIO(base64.b64decode(img)))
            if self.transform is not None:
                img = self.transform(img)

            imgs.append(img)
            labels.append(label)
        return torch.stack(imgs), torch.tensor(labels)
    


    def _classify_samples_local(self, data_dir) -> Dict[str, List[str]]:
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
    
    def _classify_samples_s3(self, s3url: S3Url) -> Dict[str, List[str]]:
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
    
    def fetch_batch_data_local(self, batch_indices, batch_id):
        images = []
        labels = []

        for idx in batch_indices:
            path, label = self._classed_items[idx]
            img = Image.open(path)

            if img.mode == "L":
                img = img.convert("RGB")

            if self.transform is not None:
                img = self.transform(img)

            images.append(img)
            labels.append(label)

        return images, labels
    
    def fetch_batch_data_s3(self, batch_indices, batch_id):
        s3_client = boto3.client('s3')
        images = []
        labels = []

        for idx in batch_indices:
            file_path, label = self._classed_items[idx]
            # Download file into memory
            obj = s3_client.get_object(Bucket=self.s3_bucket_name, Key=file_path)
            content = obj['Body'].read()

            img = Image.open(io.BytesIO(content))

            if img.mode == "L":
                img = img.convert("RGB")

            if self.transform is not None:
                img = self.transform(img)

            images.append(img)
            labels.append(label)

        return images, labels