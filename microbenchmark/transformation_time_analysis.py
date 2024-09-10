import time
import logging
import base64
# import zlib
from io import BytesIO
import zstandard
import torch
import boto3
from PIL import Image
import io
from typing import List, Tuple
from torchvision import transforms
import pickle
import sys
import csv
import os
#time to create a torch batch with transforms
from functools import wraps
import snappy
import zlib
import lz4.frame
def timer_decorator(func):
    @wraps(func)  # This preserves the original function's metadata (name, docstring, etc.)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record the start time
        result = func(*args, **kwargs)  # Call the original function
        end_time = time.time()  # Record the end time
        elapsed_time = end_time - start_time  # Calculate the elapsed time
        print(f"Function '{func.__name__}' took {elapsed_time:.4f} seconds to execute.")
        return result, elapsed_time
        # if isinstance(result, tuple):
        #     return (*result, elapsed_time)
        # return result, elapsed_time  # Return the result of the original function
    return wrapper



def get_transforms(workload_name: str) -> transforms.Compose:
    """Get the transformation pipeline based on the workload."""
    if workload_name == 'imagenet':
        return transforms.Compose([
            transforms.Resize(256), 
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif workload_name == 'cifar10':
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),    
            transforms.RandomHorizontalFlip(),        
            transforms.ToTensor(),                    
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ])
    else:
        logging.error(f"Unknown workload: {workload_name}")
        raise ValueError(f"Unknown workload: {workload_name}")

def load_batch_from_s3(bucket_name: str, prefix: str, max_images: int = 32) -> List[Tuple[Image.Image, str]]:
    """Load the first `max_images` image files from an S3 bucket."""
    s3_client = boto3.client('s3')
    images = []
    count = 0
    size = 0
    paginator = s3_client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        for obj in page.get('Contents', []):
            key = obj.get('Key')
            if key.lower().endswith(('.jpg', '.jpeg', '.png')):
                if count >= max_images:
                    return images
                try:
                    response = s3_client.get_object(Bucket=bucket_name, Key=key)
                    img_data = response['Body'].read()
                    size += sys.getsizeof(img_data)
                    image = Image.open(io.BytesIO(img_data)).convert('RGB')
                    images.append((image, key))
                    count += 1
                except Exception as e:
                    logging.error(f"Error loading image {key}: {e}")
    return images


def compute_image_size(img):
    total_size = 0
    # Get image dimensions and number of channels
    width, height = img.size
    mode = img.mode  # e.g., 'RGB', 'L', 'RGBA'
    
    # Determine the number of channels based on the image mode
    if mode == 'RGB':
        channels = 3
    elif mode == 'RGBA':
        channels = 4
    elif mode == 'L':
        channels = 1
    else:
        raise ValueError(f"Unsupported image mode: {mode}")

    # Calculate memory size of the image in bytes
    image_size = width * height * channels

    return image_size / 1024 / 1024  # Convert bytes to megabytes


@timer_decorator
def create_torch_batch_from_raw_data(batch_images: List[Tuple[Image.Image, str]], transform=None) -> Tuple[torch.Tensor, List[str]]:
    labels = []
    images = []
    if transform is None:
        transform = transforms.Compose([transforms.ToTensor()])
    for image, label in batch_images:
        labels.append(label)
        images.append(transform(image))
    return torch.stack(images), labels

@timer_decorator
def create_torch_batch_from_serialized_data(serialized_data, transform=None):

    batch_data = deserialize_batch(serialized_data)[0]
    labels = []
    images = []
    if transform is None:
        transform = transforms.Compose([transforms.ToTensor()])
    for image, label in batch_data:
        labels.append(label)
        images.append(transform(image))
    return torch.stack(images), labels


@timer_decorator
def convert_torch_batch_to_bytes(minibatch: Tuple[torch.Tensor, List[str]],  compression_lib=None, use_encoding=False) -> bytes:
    with BytesIO() as buffer:
        torch.save(minibatch, buffer)
        bytes_minibatch = buffer.getvalue()
    if compression_lib is not None:
        if compression_lib == 'zlib':
            bytes_minibatch = zlib.compress(bytes_minibatch)
        elif compression_lib == 'zstandard':
            bytes_minibatch = zstandard.compress(bytes_minibatch)
        elif compression_lib == 'snappy':
            bytes_minibatch = snappy.compress(bytes_minibatch)
        elif compression_lib == 'lz4':
            bytes_minibatch = lz4.frame.compress(bytes_minibatch)
    if use_encoding:
        bytes_minibatch = base64.b64encode(bytes_minibatch).decode('utf-8')
    return bytes_minibatch

@timer_decorator
def convert_bytes_batch_to_tensor(batch: bytes, compression_lib = None, use_encoding=False) -> Tuple[torch.Tensor, List[str]]:
    if use_encoding:
        batch = base64.b64decode(batch)
    if compression_lib is not None:
        if compression_lib == 'zlib':
            batch = zlib.decompress(batch)
        elif compression_lib == 'zstandard':
            batch = zstandard.decompress(batch)
        elif compression_lib == 'snappy':
            batch = snappy.decompress(batch)
        elif compression_lib == 'lz4':
            batch = lz4.frame.decompress(batch)
    with BytesIO(batch) as buffer:
        data_samples, labels = torch.load(buffer)
    return data_samples, labels

@timer_decorator
def serialize_batch(data) -> bytes:
    return pickle.dumps(data)

@timer_decorator
def deserialize_batch(serialized_data: bytes):
    return pickle.loads(serialized_data)


def benchmark_dataloading(raw_data, transform, compression_lib, num_tests=10):
    results = {}
    
    def average_time(func, *args, **kwargs):
        """Helper function to run a function multiple times and return the average time."""
        total_time = 0
        for _ in range(num_tests):
            result, time = func(*args, **kwargs)
            total_time += time
        return result, total_time / num_tests
    
    # Create Torch batch
    batch, avg_time = average_time(create_torch_batch_from_raw_data, raw_data, transform=transform)
    batch_images, labels = batch
    results['create_torch_batch_from_data(s)'] = avg_time
    results['torch_batch_size(mb)'] = batch_images.element_size() * batch_images.nelement() / 1024 / 1024

    # Serialize data
    serialized_data, avg_time = average_time(serialize_batch, raw_data)
    results['serialize_data_as_bytes(s)'] = avg_time
    results['data_as_bytes_size(mb)'] = sys.getsizeof(serialized_data) / 1024 / 1024

    # Deserialize data
    batch, avg_time = average_time(create_torch_batch_from_serialized_data,serialized_data, transform=transform)
    batch_images, labels = batch
    results['create_torch_batch_from_data_bytes(s)'] = avg_time
    assert (batch_images.element_size() * batch_images.nelement() / 1024 / 1024 == results['torch_batch_size(mb)'])

    # Convert Torch batch to bytes
    torch_batch_as_bytes, avg_time = average_time(convert_torch_batch_to_bytes, (batch_images, labels), compression_lib=compression_lib)
    results['convert_torch_batch_to_bytes(s)'] = avg_time
    results['torch_batch_as_bytes_size(mb)'] = sys.getsizeof(torch_batch_as_bytes) / 1024 / 1024

    # Convert bytes back to Torch batch
    batch, avg_time = average_time(convert_bytes_batch_to_tensor, torch_batch_as_bytes, compression_lib=compression_lib)
    batch_images, labels = batch
    results['convert_bytes_into_torch_batch(s)'] = avg_time
    assert (batch_images.element_size() * batch_images.nelement() / 1024 / 1024 == results['torch_batch_size(mb)'])

    return results

def main():
    logging.basicConfig(level=logging.INFO)
    bucket_name = "imagenet1k-sdl"
    prefix = "train/"
    num_tests = 5
    report_file = os.path.join('microbenchmark', 'transformation_time_analysis.csv')
    
    transform = get_transforms('imagenet') if 'imagenet' in bucket_name else get_transforms('cifar10')
    
    for batch_size in [8,16,32,64]:
        print(f"Running tests for batch size: {batch_size}")
        batch = load_batch_from_s3(bucket_name, prefix, max_images=batch_size)
        for compression_lib in [None, 'zlib', 'zstandard', 'snappy', 'lz4']:
                report_data = {
                    'bucket_name': bucket_name,
                    'batch_size': batch_size,
                    'compression_lib': str(compression_lib)}                
                # report_data['inital_data_size(mb)'] = sum(compute_image_size(img) for img, _ in batch)

                test_results = benchmark_dataloading(batch, transform, compression_lib, num_tests=num_tests)
                report_data.update(test_results)

                file_exists = os.path.isfile(report_file)
                with open(report_file, mode='a', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=report_data.keys())
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow(report_data)



if __name__ == "__main__":
   main()