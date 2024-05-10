from typing import TypeVar, Generic, Iterable, Iterator, Sequence, List, Optional, Tuple
from torch.utils.data import SequentialSampler, RandomSampler, Sampler, Dataset
import torch.distributed as dist
import os
from datetime import datetime
import random
import PIL.Image as Image
import numpy as np
import redis
import io
import numpy as np
import redis
import PIL
from rediscluster import RedisCluster
import super_dl.s3utils as s3utils
from super_dl.s3utils import S3Url
import json
from pathlib import Path
import functools
from typing import Dict, List
import torch
import time

T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')

class ShadeDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        transform=None,
        target_transform=None,
        cache_data=False,
        PQ=None,
        ghost_cache=None,
        key_counter=None,
        wss=0.1,
        cache_address = None,
        cache_granularity:str = 'sample'
    ):
        self.transform = transform
        self.target_transform = target_transform
        self.cache_data = cache_data
        self.wss = wss

        self.is_s3: bool = data_dir.startswith("s3://")
        if self.is_s3:
            self.samples: Dict[str, List[str]] = s3utils.load_paired_s3_object_keys(data_dir, True, True)
            self.bucket_name = S3Url(data_dir).bucket
        else:
            self.samples: Dict[str, List[str]] = self.load_local_sample_idxs(data_dir)
        self.cache_portion = int(self.wss * len(self))

        self.cache_host, self.cache_port = None,None
        if cache_address:
            self.cache_host, self.cache_port = cache_address.split(":")

        self.key_id_map = redis.StrictRedis(host=self.cache_host, port=self.cache_port)
        self.cache_granularity = cache_granularity
        # if self.cache_host == '0.0.0.0' or self.cache_host is None:
        #     self.key_id_map = redis.Redis()
        # else:
        #     self.startup_nodes = [{"host": self.cache_host, "port": self.cache_port}]
        #     # self.key_id_map = RedisCluster(startup_nodes=self.startup_nodes)
        #     self.key_id_map = redis.Redis()

        self.PQ = PQ
        self.ghost_cache = ghost_cache
        self.key_counter = key_counter

    @functools.cached_property
    def _classed_items(self) -> List[Tuple[str, int]]:
        return [
            (blob, class_index)
            for class_index, blob_class in enumerate(self.samples)
            for blob in self.samples[blob_class]
        ]

    def random_func(self):
        return 0.6858089651836363

    def set_num_local_samples(self, n):
        self.key_counter = n

    def set_PQ(self, curr_PQ):
        self.PQ = curr_PQ

    def set_ghost_cache(self, curr_ghost_cache):
        self.ghost_cache = curr_ghost_cache

    def get_PQ(self):
        return self.PQ

    def get_ghost_cache(self):
        return self.ghost_cache
    
    def fetch_from_cache(self, key):
        try:
            return self.key_id_map.get(key)
        except:
             return None
        
    def cache_and_evict(self, data_path, index):
        data = None
        cache_hit  = False
        if self.cache_data and self.key_id_map.exists(index):
            # print(f'Hitting {index}')
            byte_data = self.fetch_from_cache(index)
            if byte_data:
                byte_img_io = io.BytesIO(byte_data)
                data = Image.open(byte_img_io)
                cache_hit = True
        if data is None:  #data not retrieved from cache, so get it from primary storage
            # print(f'Miss {index}')
            if self.is_s3:
                data = s3utils.get_s3_object(self.bucket_name, data_path)
                data = Image.open(io.BytesIO(data))
            else:
                data = Image.open(data_path) #get_local_sample
            
            if data.mode == "L":
                data = data.convert("RGB")
            
            keys_cnt = self.key_counter + 50

            if keys_cnt >= self.cache_portion:
                try:
                    peek_item = self.PQ.peekitem()
                    if self.ghost_cache[index] > peek_item[1]:
                        evicted_item = self.PQ.popitem()
                        # print(f"Evicting index: {evicted_item[0]} Weight: {evicted_item[1][0]} Frequency: {evicted_item[1][1]}")
                        
                        if self.key_id_map.exists(evicted_item[0]):
                            self.key_id_map.delete(evicted_item[0])
                        keys_cnt -= 1
                except Exception:
                    # print("Could not evict item or PQ was empty.")
                    pass

            if self.cache_data and keys_cnt < self.cache_portion:
                byte_stream = io.BytesIO()
                data.save(byte_stream, format=data.format)
                byte_stream.seek(0)
                self.key_id_map.set(index, byte_stream.read())
                # print(f"Index: {index}")

        return data,cache_hit


    def fetch_batch_data(self, batch_indices):
        data_samples = []
        labels = []
        cache_hit_count = 0
        
        for idx in batch_indices: 
            data = None
            data_path, label = self._classed_items[idx] 
            # insertion_time = datetime.now().strftime("%H:%M:%S")
            # print(f"train_search_index: {idx} time: {insertion_time}")
            data, cache_hit = self.cache_and_evict(data_path, idx)
            data_samples.append(data)
            labels.append(label)
            if cache_hit:
                cache_hit_count +=1
        return data_samples, labels, cache_hit_count
    
    def __getitem__(self, index: int):
    
        fetch_start_time = time.perf_counter()

        batch_id, batch_indices = index

        batch_data = None
        if self.cache_granularity == 'batch':
            pass
            #batch_data = self.fetch_from_cache(batch_id)

        if batch_data:
            pass 
        
        data_samples, labels, cache_hit_count = self.fetch_batch_data(batch_indices)

        tranform_start_time = time.perf_counter()
        if self.transform is not None:
            for i in range(len(data_samples)):
                data_samples[i] = self.transform(data_samples[i])
        transform_duration =  time.perf_counter() - tranform_start_time
        fetch_duration = time.perf_counter() - fetch_start_time - transform_duration
        
        return torch.stack(data_samples), torch.tensor(labels), cache_hit_count, fetch_duration, transform_duration


    def __len__(self) -> int:
        return sum(len(class_items) for class_items in self.samples.values())

    
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

    def is_image_file(self, path: str) -> bool:
        extensions = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp']
        return any(path.endswith(extension) for extension in extensions)


class ShadeValDataset(Dataset):
    def __init__(self, imagefolders, transform=None, target_transform=None, cache_data=False):
        self.samples = []
        self.classes = []
        self.transform = transform
        self.target_transform = target_transform
        self.cache_data = cache_data

        for imagefolder in imagefolders:
            dataset = imagefolder
            self.loader = dataset.loader
            self.samples.extend(dataset.samples)
            self.classes.extend(dataset.classes)

        self.classes = list(set(self.classes))

    def random_func(self):
        return 0.6858089651836363

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target, index) where target is class_index of the target class.
        """
        path, target = self.samples[index]

        image = Image.open(path)
        sample = image.convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, index

    def __len__(self) -> int:
        return len(self.samples)
