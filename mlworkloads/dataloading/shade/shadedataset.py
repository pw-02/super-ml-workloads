
# No 'default_generator' in torch/__init__.pyi
from typing import TypeVar, List, Tuple, Dict
from datetime import datetime
import random
import PIL.Image as Image
import numpy as np
import redis
import io
import numpy as np
import redis
import heapdict
import PIL
from urllib.parse import urlparse
import boto3
import functools
from torch.utils.data import Dataset
import json
import time

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
    
T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')


class ShadeDataset(Dataset):
    def __init__(self, s3_data_dir: str, transform=None, cache_address= None, PQ=None, ghost_cache=None, wss=0.1):
        self.s3_bucket = S3Url(s3_data_dir).bucket
        self.s3_prefix = S3Url(s3_data_dir).key
        self.s3_data_dir = s3_data_dir
        self.transform = transform
        self.samples = self._get_sample_list_from_s3()
        self.cache_host, self.cache_port = cache_address.split(":")
        self.cache_data = True if cache_address is not None else False
        self.wss = wss
        self.cache_portion = self.wss * len(self)
        self.cache_portion = int(self.cache_portion // 1)
        self.PQ = PQ
        self.ghost_cache = ghost_cache
        self.key_counter = 0
        self.key_id_map:redis.StrictRedis = redis.StrictRedis(host=self.cache_host, port=self.cache_port)
        self.s3_client = boto3.client('s3')
    
    @functools.cached_property
    def _classed_items(self) -> List[Tuple[str, int]]:
        return [(blob, class_index)
            for class_index, blob_class in enumerate(self.samples)
            for blob in self.samples[blob_class]]

    def random_func(self):
        return 0.6858089651836363

    def set_num_local_samples(self):
        self.key_counter = self.key_id_map.dbsize()

    def set_PQ(self, curr_PQ):
        self.PQ = curr_PQ

    def set_ghost_cache(self, curr_ghost_cache):
        self.ghost_cache = curr_ghost_cache

    def get_PQ(self):
        return self.PQ

    def get_ghost_cache(self):
        return self.ghost_cache
    
    def cache_and_evict(self, path, target, index):
        
        fetch_start_time = time.perf_counter()
        cache_hit = False
        cached_after_fetch = False

        if self.cache_data and self.key_id_map.exists(index):
            try:
                # print('hitting %d' % (index))
                byte_image = self.key_id_map.get(index)
                byteImgIO = io.BytesIO(byte_image)
                sample = Image.open(byteImgIO)
                sample = sample.convert('RGB')
                cache_hit = True
            except PIL.UnidentifiedImageError:
                try:
                    print("Could not open image in path from byteIO: ", path)
                    # sample = Image.open(path)
                    sample = self.fetch_image_from_s3(path)
                    sample = sample.convert('RGB')
                    print("Successfully opened file from path using open.")
                except:
                    print("Could not open even from path. The image file is corrupted.")
        else:
            if index in self.ghost_cache:
                pass
                # print('miss %d' % (index))
            # image = Image.open(path)
            image = self.fetch_image_from_s3(path)
            keys_cnt = self.key_counter + 50

            if keys_cnt >= self.cache_portion:
                try:
                    peek_item = self.PQ.peekitem()
                    if self.ghost_cache[index] > peek_item[1]:
                        evicted_item = self.PQ.popitem()
                        # print("Evicting index: %d Weight: %.4f Frequency: %d" % (evicted_item[0], evicted_item[1][0], evicted_item[1][1]))

                        if self.key_id_map.exists(evicted_item[0]):
                            self.key_id_map.delete(evicted_item[0])
                        keys_cnt -= 1
                except:
                    # print("Could not evict item or PQ was empty.")
                    pass

            if self.cache_data and keys_cnt < self.cache_portion:
                byte_stream = io.BytesIO()
                image.save(byte_stream, format=image.format)
                byte_stream.seek(0)
                byte_image = byte_stream.read()
                self.key_id_map.set(index, byte_image)
                cached_after_fetch = True
                #print("Index: ", index)
            sample = image.convert('RGB')
        fetch_duration = time.perf_counter() - fetch_start_time

        return sample, fetch_duration, int(cache_hit), int(cached_after_fetch)
    
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target, index) where target is class_index of the target class.
        """

        path, target = self._classed_items[index]
        insertion_time = datetime.now()
        insertion_time = insertion_time.strftime("%H:%M:%S")
        # print("train_search_index: %d time: %s" % (index, insertion_time))

        sample, fetch_duration,  cache_hit, cached_after_fetch = self.cache_and_evict(path, target, index)
        transform_start_time = time.perf_counter()
        if self.transform is not None:
            sample = self.transform(sample)
        transform_duration = time.perf_counter() - transform_start_time

        return (sample, target), fetch_duration, transform_duration, cache_hit, cached_after_fetch
    

    def __len__(self) -> int:
        return sum(len(class_items) for class_items in self.samples.values())
    
    def fetch_image_from_s3(self, data_path):
        
        if self.s3_client is None:
            self.s3_client = boto3.client('s3')
        obj = self.s3_client.get_object(Bucket=self.s3_bucket, Key=data_path)
        img_data = obj['Body'].read()
        image = Image.open(io.BytesIO(img_data)) #.convert('RGB')
        return image
    
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

       


# class ShadeValDataset(Dataset):

#     def __init__(self, s3_data_dir: str, transform=None, cache_address= None,):
#         _datasets = []
#         self.samples = []
#         self.classes = []
#         self.transform = transform
#         self.target_transform = target_transform
#         self.loader = None
#         self.cache_data = cache_data

#         for imagefolder in imagefolders:
#             dataset = imagefolder
#             self.loader = dataset.loader
#             _datasets.append(dataset)
#             self.samples.extend(dataset.samples)
#             self.classes.extend(dataset.classes)
#         self.classes = list(set(self.classes))

#     def random_func(self):
#         return 0.6858089651836363

#     def __getitem__(self, index: int):
#         """
#         Args:
#             index (int): Index
#         Returns:
#             tuple: (sample, target, index) where target is class_index of the target class.
#         """
#         path, target = self.samples[index]

#         image = Image.open(path)
#         sample = image.convert('RGB')

#         if self.transform is not None:
#             sample = self.transform(sample)
#         if self.target_transform is not None:
#             target = self.target_transform(target)

#         return sample, target, index

#     def __len__(self) -> int:
#         return len(self.samples)
