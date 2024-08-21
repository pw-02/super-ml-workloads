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

class S3MappedDataset(Dataset):
    def __init__(self, s3_bucket: str, s3_prefix: str, transform=None):
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.transform = transform
        
        # Initialize S3 client

        # List all files in the S3 bucket under the specified prefix
        self.samples = self.get_sample_list_from_s3()
    
    @functools.cached_property
    def _classed_items(self) -> List[Tuple[str, int]]:
        return [(blob, class_index)
            for class_index, blob_class in enumerate(self.samples)
            for blob in self.samples[blob_class]]

    def get_sample_list_from_s3(self, use_index_file=True, images_only=True) -> Dict[str, List[str]]:
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
    
    def __len__(self) -> int:
        return sum(len(class_items) for class_items in self.samples.values())
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
        batch_id, batch_indices = idx
        # batch_indices = self._classed_items[idx][0]  # Simplified for demonstration
        
        fetch_start_time = time.perf_counter()

        data_samples, labels = self.fetch_batch_from_s3(batch_indices)

        fetch_duration = time.perf_counter() - fetch_start_time

        transform_start_time = time.perf_counter()
        if self.transform is not None:
            for i in range(len(data_samples)):
                data_samples[i] = self.transform(data_samples[i])
        transform_duration =  time.perf_counter() - transform_start_time

        return (torch.stack(data_samples), torch.tensor(labels)), fetch_duration, transform_duration, False

    def fetch_batch_from_s3(self, batch_indices: List[str]) -> Tuple[List[torch.Tensor], List[int]]:
        s3_client = boto3.client('s3')

        data_samples = []
        labels = []
        for idx in batch_indices:
            data_path, label = self._classed_items[idx]
            obj = s3_client.get_object(Bucket=self.s3_bucket, Key=data_path)
            img_data = obj['Body'].read()
            image = Image.open(io.BytesIO(img_data)).convert('RGB')
            data_samples.append(image)
            labels.append(label)  # Simplified; adjust based on your label extraction
        return data_samples, labels


if __name__ == "__main__":
    # Example usage
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = S3MappedDataset(s3_bucket="sdl-cifar10", s3_prefix="train/", transform=transform)
    img, label = dataset[0]
    print(img.shape)