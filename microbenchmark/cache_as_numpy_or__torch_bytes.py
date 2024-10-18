import torch
import numpy as np
from io import BytesIO
import time
import redis
import sys
from torchvision import transforms, datasets
import lz4.frame
import pickle

use_compression = True

# Initialize Redis connection
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)
redis_client.flushdb
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

# Benchmarking with pickle
# Serialize with pickle
start_time = time.time()
pickle_serialized = pickle.dumps((data_samples, labels))  # Faster than torch.save
pickle_serialize_time = time.time() - start_time
print(f"pickle.dumps() serialization: {pickle_serialize_time:.6f} seconds")

# Compress the serialized data (optional)
if use_compression:
    compressed_data = lz4.frame.compress(pickle_serialized)
else:
    compressed_data = pickle_serialized

# Write compressed data to Redis
start_time = time.time()
redis_client.set('pickle_cifar10_batch', compressed_data)
redis_write_time_pickle = time.time() - start_time
print(f"Write to Redis (pickle): {redis_write_time_pickle:.6f} seconds")

# Read compressed data from Redis
start_time = time.time()
pickle_serialized_from_redis = redis_client.get('pickle_cifar10_batch')
if use_compression:
    decompressed_data = lz4.frame.decompress(pickle_serialized_from_redis)
else:
    decompressed_data = pickle_serialized_from_redis
redis_read_time_pickle = time.time() - start_time
print(f"Read from Redis (pickle): {redis_read_time_pickle:.6f} seconds")

# Deserialize with pickle
start_time = time.time()
loaded_data_samples_pickle, loaded_labels_pickle = pickle.loads(decompressed_data)  # Faster than torch.load
pickle_deserialize_time = time.time() - start_time
print(f"pickle.loads() deserialization: {pickle_deserialize_time:.6f} seconds")

# Benchmarking with torch.save
# Serialize with torch.save
buffer = BytesIO()
start_time = time.time()
torch.save((data_samples, labels), buffer)  # Use torch.save for comparison
torch_serialized = buffer.getvalue()
torch_serialize_time = time.time() - start_time
print(f"torch.save() serialization: {torch_serialize_time:.6f} seconds")

# Compress the serialized data (optional)
if use_compression:
    compressed_data_torch = lz4.frame.compress(torch_serialized)
else:
    compressed_data_torch = torch_serialized

# Write compressed data to Redis
start_time = time.time()
redis_client.set('torch_cifar10_batch', compressed_data_torch)
redis_write_time_torch = time.time() - start_time
print(f"Write to Redis (torch): {redis_write_time_torch:.6f} seconds")

# Read compressed data from Redis
start_time = time.time()
torch_serialized_from_redis = redis_client.get('torch_cifar10_batch')
if use_compression:
    decompressed_data_torch = lz4.frame.decompress(torch_serialized_from_redis)
else:
    decompressed_data_torch = torch_serialized_from_redis
redis_read_time_torch = time.time() - start_time
print(f"Read from Redis (torch): {redis_read_time_torch:.6f} seconds")

# Deserialize with torch.load
buffer_torch = BytesIO(decompressed_data_torch)
start_time = time.time()
loaded_data_samples_torch, loaded_labels_torch = torch.load(buffer_torch)  # Use torch.load for comparison
torch_deserialize_time = time.time() - start_time
print(f"torch.load() deserialization: {torch_deserialize_time:.6f} seconds")

# Memory size comparisons
pickle_memory_size_uncompressed = sys.getsizeof(pickle_serialized)
pickle_memory_size_compressed = sys.getsizeof(compressed_data)
torch_memory_size_uncompressed = sys.getsizeof(torch_serialized)
torch_memory_size_compressed = sys.getsizeof(compressed_data_torch)

print(f"pickle Uncompressed size: {pickle_memory_size_uncompressed / (1024 * 1024):.6f} MB")
print(f"pickle Compressed size: {pickle_memory_size_compressed / (1024 * 1024):.6f} MB")
print(f"torch Uncompressed size: {torch_memory_size_uncompressed / (1024 * 1024):.6f} MB")
print(f"torch Compressed size: {torch_memory_size_compressed / (1024 * 1024):.6f} MB")
