# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from typing import Callable, List, Tuple, Union
from torch import Tensor
import boto3
import io
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from typing import List, Dict, Tuple
import functools
import time
from urllib.parse import urlparse
import redis

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

class S3RedisRetrievalTrainingDataset(Dataset):

    def __init__(self, annotation_file, s3_data_dir: str, image_transform=None, text_transform=None,  cache_address= None):
        
        self.s3_bucket = S3Url(s3_data_dir).bucket
        self.s3_prefix = S3Url(s3_data_dir).key
        self.s3_data_dir = s3_data_dir
        self.annotation_file = S3Url(annotation_file).key
        self.image_transform = image_transform
        self.text_transform = text_transform
        if cache_address is not None:
            self.cache_host, self.cache_port = cache_address.split(":")
            self.cache_port = int(self.cache_port)
            self.use_cache = True
        else:
            self.use_cache = False

        self.cache_client = None
        self.s3_client = None
        self.ann = []
        self.idx = {}
        self.samples = self._get_sample_list_from_s3()
        pass

    def _get_sample_list_from_s3(self):
        s3_client = boto3.client('s3')
        paired_samples = {}
        index_object = s3_client.get_object(Bucket=self.s3_bucket, Key=self.annotation_file)
        file_content = index_object['Body'].read().decode('utf-8')
        # samples = json.loads(file_content)
        self.ann += json.loads(file_content)

        i = 0
        for ann in self.ann:
            image_id = ann["image_id"]
            if image_id not in self.idx.keys():
                self.idx[image_id] = i
                i += 1

        # for idx, sample in enumerate(samples):
        #     image_path = sample['image']
        #     image_path =  self.s3_prefix + os.path.basename(image_path)
        #     caption = sample['caption']
        #     if image_path not in paired_samples:
        #         paired_samples[idx] = (image_path, caption)
        # return paired_samples
    
    def __len__(self) -> int:
         return len(self.ann)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, int]:
        ann= self.ann[index]
        image_path = ann["image"]
        caption = ann["caption"]
        item_data  = None
        cached_after_fetch = False
        start_loading_time = time.perf_counter()
        
        if self.use_cache:
            item_data = self._load_item_from_cache(image_path)

        if item_data  is not None and (isinstance(item_data , bytes) or isinstance(item_data , str)):
            start_transformation_time   = time.perf_counter()
            byteImgIO = io.BytesIO(item_data)
            sample = Image.open(byteImgIO)
            sample = sample.convert('RGB')

            if self.transform is not None:
                sample = self.transform(sample)
            transformation_time = time.perf_counter() - start_transformation_time
            cache_hit = True
            fetch_duration  = time.perf_counter() - start_loading_time - transformation_time
            return (sample, caption), fetch_duration, transformation_time, cache_hit, cached_after_fetch
            # return image, caption, self.idx[ann["image_id"]]

        
        image = self.fetch_image_from_s3(image_path)
        cache_hit = False
        if self.use_cache:
            byte_stream = io.BytesIO()
            image.save(byte_stream, format=image.format)
            byte_stream.seek(0)
            byte_image = byte_stream.read()
            self.cache_client.set(image_path, byte_image)
            cached_after_fetch = True
            image = image.convert('RGB')
        
        transform_start_time = time.perf_counter()
        if self.image_transform is not None:
            image = self.image_transform(image)
        if self.text_transform is not None:
            caption = self.text_transform(caption)
        transformation_time = time.perf_counter() - transform_start_time
        fetch_duration  = time.perf_counter() - start_loading_time - transformation_time

        # return (sample, caption), fetch_duration, transformation_time, cache_hit, cached_after_fetch
        return image, caption, index, fetch_duration, transformation_time, cache_hit, cached_after_fetch
    
    
    def _initialize_cache_client(self):
        """Initialize Redis cache client if not already connected."""
        if self.cache_client is None:
            self.cache_client = redis.StrictRedis(host=self.cache_host, port=self.cache_port,  ssl=True)
    
    def _load_item_from_cache(self, key):
        try:
            self._initialize_cache_client()   
            return self.cache_client.get(key)
        except Exception as e:
            print(f"Error fetching from cache: {e}")
            return None
        
    def fetch_image_from_s3(self, data_path):  
        if self.s3_client is None:
            self.s3_client = boto3.client('s3')
        obj = self.s3_client.get_object(Bucket=self.s3_bucket, Key=data_path)
        img_data = obj['Body'].read()
        image = Image.open(io.BytesIO(img_data)) #.convert('RGB')
        return image
    
if __name__ == "__main__":
    # Example usage
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = S3RedisRetrievalTrainingDataset(
        annotation_file="s3://coco-dataset/data/coco_train.json", 
        s3_data_dir="s3://coco-dataset/", 
        image_transform=transform,
        text_transform=None,
        cache_address=None)
    dataset.__getitem__(0)




# class ImageToTextRetrievalDataset(Dataset):
#     """
#     Create the dataset for Image-to-Text Retrieval task.

#     Args:
#         ann_file (List[str]): The paths to annotation json files.
#         image_root (str): The path to image data directory.
#         image_transform (Callable[[Image.Image], Tensor]): Image data transform.

#     Dataset Outputs:
#         image (Tensor): Transformed image input tensor of shape (C, H, W).
#     """

#     def __init__(
#         self,
#         ann_file: List[str],
#         image_root: str,
#         image_transform: Callable[[Image.Image], Tensor],
#     ) -> None:
#         self.image_root = image_root
#         self.image_transform = image_transform

#         self.ann = []
#         self.images = []  # paths to all images in the dataset
#         self.image_to_text = {}  # map image ids to text ids for evaluation
#         for f in ann_file:
#             self.ann += json.load(open(f, "r"))

#         text_id = 0
#         for image_id, ann in enumerate(self.ann):
#             self.images.append(ann["image"])
#             num_text = len(ann["caption"])
#             self.image_to_text[image_id] = list(range(text_id, text_id + num_text))
#             text_id += num_text

#     def __len__(self) -> int:
#         return len(self.images)

#     def __getitem__(self, index: int) -> Tensor:
#         image_path = os.path.join(self.image_root, self.images[index])
#         image = Image.open(image_path).convert("RGB")
#         image = self.image_transform(image)
#         return image


# class TextToImageRetrievalDataset(Dataset):
#     """
#     Create the dataset for Text-to-Image Retrieval task.

#     Args:
#         ann_file (List[str]): The paths to annotation json files.
#         text_transform (Callable[[Union[List[str], str]], Tensor]): Text data transform.

#     Dataset Outputs:
#         text (Tensor): Transformed text token input ids.
#     """

#     def __init__(
#         self,
#         ann_file: List[str],
#         text_transform: Callable[[Union[List[str], str]], Tensor],
#     ) -> None:
#         self.text_transform = text_transform

#         self.ann = []
#         self.text = []  # all text strings in the dataset
#         self.text_to_image = {}  # map text ids to image ids for evaluation
#         for f in ann_file:
#             self.ann += json.load(open(f, "r"))

#         text_id = 0
#         for image_id, ann in enumerate(self.ann):
#             for caption in ann["caption"]:
#                 self.text.append(caption)
#                 self.text_to_image[text_id] = image_id
#                 text_id += 1

#     def __len__(self) -> int:
#         return len(self.text)

#     def __getitem__(self, index: int) -> Tensor:
#         text = self.text_transform(self.text[index])
#         return text
