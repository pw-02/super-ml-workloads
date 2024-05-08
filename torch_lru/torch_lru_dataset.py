import functools
import torch
import super_dl.s3utils as s3utils
from super_dl.s3utils import S3Url
from typing import List, Tuple, Dict
from PIL import Image
import io
from pathlib import Path
import json
import os
import redis
import base64
import zlib

class TorchLRUDataset(torch.utils.data.Dataset):
    def __init__(self,data_dir:str, transform, cache_address:str = None, cache_granularity:str = 'sample'):

        self.is_s3: bool = data_dir.startswith("s3://")

        if self.is_s3:
            self.samples: Dict[str, List[str]] = s3utils.load_paired_s3_object_keys(data_dir, True, True)
            self.bucket_name = S3Url(data_dir).bucket
        else:
            self.samples: Dict[str, List[str]] = self.load_local_sample_idxs(data_dir)
        self.transform = transform

        self.cache_host, self.cache_port = None,None
        if cache_address:
            self.cache_host, self.cache_port = cache_address.split(":")
            self.cache_client = redis.StrictRedis(host=self.cache_host, port=self.cache_port)
            self.use_cache = True
        else:
            self.cache_client = None
            self.use_cache = False
        self.cache_granularity = cache_granularity
    
    @functools.cached_property
    def _classed_items(self) -> List[Tuple[str, int]]:
        return [(blob, class_index)
            for class_index, blob_class in enumerate(self.samples)
            for blob in self.samples[blob_class]]
    
    def __len__(self):
        return sum(len(class_items) for class_items in self.samples.values())
    

    def fetch_from_cache(self, key):
        try:
            return self.cache_client.get(key)
        except:
             return None

    def __getitem__(self, idx):
        batch_id, batch_indices,  = idx

        batch_data = None

        if self.use_cache and self.cache_granularity == 'batch':
            batch_data = self.fetch_from_cache(batch_id)
            
        if batch_data:
            # Convert JSON batch to torch format
            torch_imgs, torch_labels, transform_time = self.deserialize_torch_batch(batch_data)
            # print('data returned') 
            return torch_imgs, torch_labels, True, batch_id
         
        data_samples, labels = self.fetch_batch_data(batch_indices)

        if self.transform is not None:
            for i in range(len(data_samples)):
                data_samples[i] = self.transform(data_samples[i])

        return torch.stack(data_samples), torch.tensor(labels), False, batch_id
    
    def random_true_or_false(self) -> bool:
        import random
        return random.choice([True, False])

    def fetch_batch_data(self,batch_indices):
        data_samples = []
        labels = []
        for idx in batch_indices: 
            data = None
            data_path, label = self._classed_items[idx] 
            if self.use_cache and self.cache_granularity == 'sample':
                byte_data = self.fetch_from_cache(data_path)
                byte_img_io = io.BytesIO(byte_data)
                data = Image.open(byte_img_io).convert('RGB')

            if data is None:  #data not retrieved from cache, so get it from primary storage
                if self.is_s3:
                    data = s3utils.get_s3_object(self.bucket_name, data_path)
                    data = Image.open(io.BytesIO(data))
                else:
                    data = Image.open(data_path) #get_local_sample
                
                if data.mode == "L":
                    data = data.convert("RGB")

                if self.use_cache and self.cache_granularity == 'sample':
                    byte_stream = io.BytesIO()
                    data.save(byte_stream, format=data.format)
                    byte_stream.seek(0)
                    self.cache_client.set(data_path, byte_stream.read())

            data_samples.append(data)
            labels.append(label)

        return data_samples, labels
        

    def is_image_file(self, path: str):
        return any(path.endswith(extension) for extension in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP'])
    

    def load_local_sample_idxs(self, data_dir) -> Dict[str, List[str]]:
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
    
    def deserialize_torch_batch(self, batch_data):
            decoded_data = base64.b64decode(batch_data) # Decode base64
            try:
                decoded_data = zlib.decompress(decoded_data)
            except:
                pass
        
            buffer = io.BytesIO(decoded_data)
            batch_samples, batch_labels = torch.load(buffer)
            return  batch_samples, batch_labels