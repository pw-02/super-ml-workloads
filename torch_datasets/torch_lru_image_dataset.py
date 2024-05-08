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

class TorchLRUImageDataset(torch.utils.data.Dataset):
    def __init__(self,data_dir:str, transform, cache_host = None, cache_port = None):

        self.is_s3: bool = data_dir.startswith("s3://")

        if self.is_s3:
            self.samples: Dict[str, List[str]] = s3utils.load_paired_s3_object_keys(data_dir, True, True)
            self.bucket_name = S3Url(data_dir).bucket
        else:
            self.samples: Dict[str, List[str]] = self.load_local_sample_idxs(data_dir)
        self.transform = transform

        if cache_host is not None:
            self.cache_client = redis.StrictRedis(host=cache_host, port=cache_port)
            self.use_cache = True
        else:
            self.cache_client = None
            self.use_cache = False
    

    @functools.cached_property
    def _classed_items(self) -> List[Tuple[str, int]]:
        return [(blob, class_index)
            for class_index, blob_class in enumerate(self.samples)
            for blob in self.samples[blob_class]]
    
    def __len__(self):
        return sum(len(class_items) for class_items in self.samples.values())
    
    def __getitem__(self, idx):
        sample_data = None
        sample_path, sample_label = self._classed_items[idx]

        if self.use_cache:
            sample_data = self.get_from_cache(idx)

        if sample_data is None:  #data not retrieved from cache, so get it from primary storage
            if self.is_s3:
                sample_data = s3utils.get_s3_object(self.bucket_name, sample_path)
                sample_data = Image.open(io.BytesIO(sample_data))
            else:
                sample_data = self.get_local_sample(sample_data)

        if sample_data.mode == "L":
            sample_data = sample_data.convert("RGB")
        
        if self.transform is not None:
            sample_data = self.transform(sample_data)
       
        return sample_data, sample_label

    def is_image_file(self, path: str):
        return any(path.endswith(extension) for extension in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP'])
    

    def get_local_sample(self, file_path):
        img = Image.open(file_path)
        return img

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
    
    def get_from_cache(self, key):
        try:
            sample_data = self.cache_client.get(key)
            byte_img_io = io.BytesIO(sample_data)
            sample_data = Image.open(byte_img_io).convert('RGB')
            return sample_data
        except:
             return None
