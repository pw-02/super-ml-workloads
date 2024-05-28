import functools
import torch
import mlworklaods.s3utils as s3utils
from mlworklaods.s3utils import S3Url
from typing import List, Tuple, Dict
from PIL import Image
import io
from pathlib import Path
import json
import os
import redis
import base64
import zlib
from mlworklaods.utils import timer_decorator
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import concurrent.futures
from torch.utils.data import Dataset
from  heapdict import heapdict

class ShadeDatasetS3(Dataset):
    def __init__(self, data_dir, transform=None, cache_address:str = None, PQ=None, ghost_cache=None, key_counter= None, wss = 0.1):
      
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
            self.key_id_map = redis.StrictRedis(host=self.cache_host, port=self.cache_port)
            self.use_cache = True
      else:
            self.key_id_map = None
            self.use_cache = False
      
      self.cache_granularity  = 'sample'
      self.wss = wss
      # self.classes = list(set(self.classes))
      self.cache_portion = self.wss * len(self) #TODO
      self.cache_portion = int(self.cache_portion // 1)
      self.PQ:heapdict = PQ
      self.ghost_cache = ghost_cache
      self.key_counter = key_counter
    
    @functools.cached_property
    def _classed_items(self) -> List[Tuple[str, int]]:
        return [(blob, class_index)
            for class_index, blob_class in enumerate(self.samples)
            for blob in self.samples[blob_class]]
    def random_func(self):
      return 0.6858089651836363
    
    def set_num_local_samples(self,n):
      self.key_counter = n
    
    def set_PQ(self,curr_PQ):
      self.PQ = curr_PQ

    def set_ghost_cache(self,curr_ghost_cache):
      self.ghost_cache = curr_ghost_cache

    def get_PQ(self):
      return self.PQ

    def get_ghost_cache(self):
      return self.ghost_cache
    
    def get_key_id_map(self):
      return self.key_id_map
    
    def __len__(self):
        return sum(len(class_items) for class_items in self.samples.values())
    
    def fetch_from_cache(self, key):
        try:
            return self.key_id_map.get(key)
        except:
             return None
    def exits_in_cache(self, key):
        data = self.fetch_from_cache(key)
        if data is None:
            return False
        else:
            return True



    def __getitem__(self, idx):
        batch_id, batch_indices,  = idx
        fetch_start_time = time.perf_counter()

        batch_data = None

        if self.use_cache and self.cache_granularity == 'batch':
            batch_data = self.fetch_from_cache(batch_id)
            
        if batch_data:
            # Convert JSON batch to torch format
            tranform_start_time = time.perf_counter()
            torch_imgs, torch_labels = self.deserialize_torch_batch(batch_data)
            transform_duration =  time.perf_counter() - tranform_start_time
            # print('data returned') 
            cache_hits = len(batch_indices)
            fetch_duration = time.perf_counter() - fetch_start_time - transform_duration
            
            if torch_imgs is not None and torch_labels is not None:
                return (torch_imgs, torch_labels), cache_hits, fetch_duration, transform_duration
        
        data_samples, labels, cache_hits = self.fetch_batch_data(batch_indices)

        tranform_start_time = time.perf_counter()
        if self.transform is not None:
            for i in range(len(data_samples)):
                data_samples[i] = self.transform(data_samples[i])
        transform_duration =  time.perf_counter() - tranform_start_time

        if self.use_cache and self.cache_granularity == 'batch':
            minibatch = torch.stack(data_samples), torch.tensor(labels)
            #Serialize the PyTorch tensor
            buffer = io.BytesIO()
            torch.save(minibatch, buffer)
            minibatch = zlib.compress(buffer.getvalue()) #use_compression:
            minibatch = base64.b64encode(minibatch).decode('utf-8')
            try:
                self.key_id_map.set(batch_id, minibatch)
            except:
                pass
        fetch_duration = time.perf_counter() - fetch_start_time - transform_duration
        return (torch.stack(data_samples), torch.tensor(labels)), cache_hits, fetch_duration, transform_duration
    
    def random_true_or_false(self) -> bool:
        import random
        return random.choice([True, False])
    

    def get_data_sample(self, bucket_name, sample_idx):
        sample_path, sample_label = self._classed_items[sample_idx]
        content = s3utils.get_s3_object(bucket_name, sample_path)

        img = Image.open(io.BytesIO(content))
        img = img.convert("RGB")      
        return img, sample_label

    def fetch_batch_from_s3(self, indices: List[int]):
        images = []
        labels = []
        cache_hits = 0
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.get_data_sample, self.bucket_name, sample_idx): sample_idx for sample_idx in indices}
            for future in concurrent.futures.as_completed(futures):
                sample_idx = futures[future]
                try:
                    image_content, label = future.result()
                    images.append(image_content)
                    labels.append(label)
                except Exception as e:
                    print(f"Error processing sample {sample_idx}: {e}")
                    
        return images, labels, cache_hits
    
    
    def fetch_batch_data(self,batch_indices):
      data_samples = []
      labels = []
      cache_hits = 0
      for idx in batch_indices: 
            data = None
            data_path, label = self._classed_items[idx] 
            
            if self.use_cache and self.cache_granularity == 'sample':
                 byte_data = self.fetch_from_cache(idx)
                 if byte_data:
                     byte_img_io = io.BytesIO(byte_data)
                     data = Image.open(byte_img_io)
                     cache_hits +=1
                
            if data is None:  #data not retrieved from cache, so get it from primary storage
                if self.is_s3:
                    data = s3utils.get_s3_object(self.bucket_name, data_path)
                    data = Image.open(io.BytesIO(data))
                else:
                    data = Image.open(data_path) #get_local_sample
                keys_cnt = self.key_counter + 50
                
                if(keys_cnt >= self.cache_portion):
                        try:
                            peek_item = self.PQ.peekitem()
                            if self.ghost_cache[idx] > peek_item[1]:
                              evicted_item = self.PQ.popitem() 
                            #   print("Evicting index: %d Weight: %.4f Frequency: %d" %(evicted_item[0], evicted_item[1][0], evicted_item[1][1]))
                              if self.exits_in_cache(evicted_item[0]):
                                    # self.key_id_map.delete(evicted_item[0])
                                    self.key_counter-=1
                                    keys_cnt-=1
                        except:
                            # print("Could not evict item or PQ was empty.")
                            pass

                if self.use_cache and self.cache_granularity == 'sample' and keys_cnt < self.cache_portion:
                        byte_stream = io.BytesIO()
                        data.save(byte_stream, format=data.format)
                        byte_stream.seek(0)
                        try:
                            self.key_id_map.set(idx, byte_stream.read())
                            self.key_counter+=1
                        except:
                            return None
                data = data.convert("RGB")
                
            data_samples.append(data)
            labels.append(label)

      return data_samples, labels, cache_hits
        

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