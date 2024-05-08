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
        host_ip='0.0.0.0',
        port_num='6379'
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

        if host_ip == '0.0.0.0' or host_ip is None:
            self.key_id_map = redis.Redis()
        else:
            self.startup_nodes = [{"host": host_ip, "port": port_num}]
            # self.key_id_map = RedisCluster(startup_nodes=self.startup_nodes)
            self.key_id_map = redis.Redis()

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

    def cache_and_evict(self, path, target, index):
        if self.cache_data and self.key_id_map.exists(index):
            try:
                print(f'Hitting {index}')
                byte_image = self.key_id_map.get(index)
                byte_img_io = io.BytesIO(byte_image)
                sample = Image.open(byte_img_io).convert('RGB')
            except PIL.UnidentifiedImageError:
                try:
                    print(f"Could not open image from byteIO at path: {path}")
                    sample = Image.open(path).convert('RGB')
                    print("Successfully opened file from path using open.")
                except Exception:
                    print("Could not open image even from path. The image file is corrupted.")
        else:
            if index in self.ghost_cache:
                print(f'Miss {index}')
                
            if self.is_s3:
                image = s3utils.get_s3_object(self.bucket_name, path)
                image = Image.open(io.BytesIO(image)).convert('RGB')
            else:
                image = Image.open(path).convert('RGB')

            keys_cnt = self.key_counter + 50

            if keys_cnt >= self.cache_portion:
                try:
                    peek_item = self.PQ.peekitem()
                    if self.ghost_cache[index] > peek_item[1]:
                        evicted_item = self.PQ.popitem()
                        print(f"Evicting index: {evicted_item[0]} Weight: {evicted_item[1][0]} Frequency: {evicted_item[1][1]}")
                        
                        if self.key_id_map.exists(evicted_item[0]):
                            self.key_id_map.delete(evicted_item[0])
                        keys_cnt -= 1
                except Exception:
                    print("Could not evict item or PQ was empty.")

            if self.cache_data and keys_cnt < self.cache_portion:
                byte_stream = io.BytesIO()
                image.save(byte_stream, format=image.format)
                byte_stream.seek(0)
                byte_image = byte_stream.read()
                self.key_id_map.set(index, byte_image)
                print(f"Index: {index}")

        return image.convert('RGB')

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self._classed_items[index]
        insertion_time = datetime.now().strftime("%H:%M:%S")
        print(f"train_search_index: {index} time: {insertion_time}")

        sample = self.cache_and_evict(path, target, index)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

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
