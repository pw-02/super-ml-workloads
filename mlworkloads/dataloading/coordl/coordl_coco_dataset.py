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

class CoorDLCocoRetrievalTrainingDataset(Dataset):

    def __init__(self, annotation_file, s3_data_dir: str, image_transform=None, text_transform=None,  cache_address= None, wss=0.1):
        
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
        self.wss = wss
        self.cache_portion = self.wss * len(self)
        self.key_counter = 0
        pass

    @functools.cached_property
    def _classed_items(self) -> List[Tuple[str, int]]:
        return [(blob, class_index)
            for class_index, blob_class in enumerate(self.samples)
            for blob in self.samples[blob_class]]

    def _get_sample_list_from_s3(self) -> Dict[str, List[str]]:
        s3_client = boto3.client('s3')
        index_object = s3_client.get_object(Bucket=self.s3_bucket, Key=self.annotation_file)
        file_content = index_object['Body'].read().decode('utf-8')
        # samples = json.loads(file_content)
        paired_samples = json.loads(file_content)
        return paired_samples
    
    def __len__(self) -> int:
        return sum(len(class_items) for class_items in self.samples.values())

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, int]:

        sample, image_id = self._classed_items[index]
        image_path, caption = sample

        image  = None
        cached_after_fetch = False
        start_loading_time = time.perf_counter()

        if self.use_cache:
            image = self._load_item_from_cache(index)

        if image  is not None and (isinstance(image , bytes) or isinstance(image , str)):
            start_transformation_time   = time.perf_counter()
            byteImgIO = io.BytesIO(image)
            image = Image.open(byteImgIO)
            image = image.convert('RGB')
            if self.image_transform is not None:
                image = self.image_transform(image)
            if self.text_transform is not None:
                caption = self.text_transform(caption)
        
            transformation_time = time.perf_counter() - start_transformation_time
            cache_hit = True
            fetch_duration  = time.perf_counter() - start_loading_time - transformation_time
            return image, caption, image_id, fetch_duration, transformation_time, cache_hit, cached_after_fetch
           
        image = self.fetch_image_from_s3(image_path)
        cache_hit = False
        if self.use_cache and self.key_counter < self.cache_portion:
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
        image = Image.open(io.BytesIO(img_data)).convert('RGB')
        return image
    
if __name__ == "__main__":
    # Example usage
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = CoorDLCocoRetrievalTrainingDataset(
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
