import torch
import numpy as np
from io import BytesIO
import time
import redis
import sys
from torchvision import transforms, datasets
import lz4.frame
import zstandard as zstd
import brotli
import gzip
import snappy
import lzma
import bz2

use_compression = True
compression_level = -1  # -1 for default level

# Initialize Redis connection
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)
redis_client.flushdb()

# Load CIFAR-10 Dataset (batch size of 128)
batch_size = 128
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
cifar10_dataset = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
data_loader = torch.utils.data.DataLoader(cifar10_dataset, batch_size=batch_size, shuffle=True)

# Get a single batch of images and labels
data_samples, labels = next(iter(data_loader))

# Serialize with torch.save
buffer = BytesIO()
start_time = time.time()
torch.save((data_samples, labels), buffer)  # Use torch.save for comparison
torch_serialized = buffer.getvalue()
torch_serialize_time = time.time() - start_time
print(f"Uncompressed size: {sys.getsizeof(torch_serialized) / (1024 * 1024):.6f} MB")

# Function for compressing data
def compress_data(data, method):
    if method == 'lz4':
        return lz4.frame.compress(data, compression_level=compression_level)
    elif method == 'zstd':
        compressor = zstd.ZstdCompressor(level=-1)
        return compressor.compress(data)
    elif method == 'brotli':
        return brotli.compress(data)
    elif method == 'gzip':
        return gzip.compress(data)
    elif method == 'snappy':
        return snappy.compress(data)
    elif method == 'lzma':
        return lzma.compress(data)
    elif method == 'bz2':
        return bz2.compress(data)
    return data

# Function for decompressing data
def decompress_data(data, method):
    if method == 'lz4':
        return lz4.frame.decompress(data)
    elif method == 'zstd':
        decompressor = zstd.ZstdDecompressor()
        return decompressor.decompress(data)
    elif method == 'brotli':
        return brotli.decompress(data)
    elif method == 'gzip':
        return gzip.decompress(data)
    elif method == 'snappy':
        return snappy.decompress(data)
    elif method == 'lzma':
        return lzma.decompress(data)
    elif method == 'bz2':
        return bz2.decompress(data)
    return data

# List of compression methods to benchmark
methods = ['lz4', 'zstd', 'snappy']

for method in methods:
    # Compress the serialized data
    start_time = time.time()
    compressed_data = compress_data(torch_serialized, method) if use_compression else torch_serialized
    compression_time = time.time() - start_time
    print(f"{method.upper()} Compressed size: {sys.getsizeof(compressed_data) / (1024 * 1024):.6f} MB (Time: {compression_time:.6f} seconds)")

    # Write compressed data to Redis
    start_time = time.time()
    redis_client.set(f'torch_cifar10_batch_{method}', compressed_data)
    redis_write_time = time.time() - start_time
    print(f"Write to Redis ({method}): {redis_write_time:.6f} seconds")

    # Read compressed data from Redis
    start_time = time.time()
    serialized_from_redis = redis_client.get(f'torch_cifar10_batch_{method}')
    redis_read_time = time.time() - start_time
    print(f"Read from Redis ({method}): {redis_read_time:.6f} seconds")

    # Decompress the data
    start_time = time.time()
    decompressed_data = decompress_data(serialized_from_redis, method) if use_compression else serialized_from_redis
    decompression_time = time.time() - start_time
    print(f"Decompression ({method}): {decompression_time:.6f} seconds")

    # Deserialize the data
    start_time = time.time()
    buffer_torch = BytesIO(decompressed_data)
    loaded_data_samples_torch, loaded_labels_torch = torch.load(buffer_torch)
    deserialization_time = time.time() - start_time
    print(f"Deserialization ({method}): {deserialization_time:.6f} seconds\n")
