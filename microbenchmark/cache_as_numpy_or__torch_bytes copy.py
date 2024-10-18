import numpy as np
from io import BytesIO
import time
import redis
import sys
from torchvision import transforms, datasets
import lz4.frame
import torch
use_compression = False
# Initialize Redis connection
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)

# Load CIFAR-10 Dataset (batch size of 128)
batch_size = 128
transform = transforms.Compose([
    transforms.Resize(224), 
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
cifar10_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
data_loader = torch.utils.data.DataLoader(cifar10_dataset, batch_size=batch_size, shuffle=True)

# Get a single batch of images and labels
data_samples, labels = next(iter(data_loader))

# Serialize with torch.save
with BytesIO() as buffer:
    torch.save((data_samples, labels), buffer)
    torch_serialized = buffer.getvalue()

# Compress the serialized data
if use_compression:
    compressed_data = lz4.frame.compress(torch_serialized)
else:
    compressed_data = torch_serialized

# Write compressed data to Redis
start_time = time.time()
redis_client.set('torch_cifar10_batch', compressed_data)
torch_write_time = time.time() - start_time
print(f"Compressed torch.save() write to Redis: {torch_write_time:.6f} seconds")

# Read compressed data from Redis
start_time = time.time()
torch_serialized_from_redis = redis_client.get('torch_cifar10_batch')
if use_compression:
    decompressed_data = lz4.frame.decompress(torch_serialized_from_redis)
else:
    decompressed_data = torch_serialized_from_redis
torch_read_time = time.time() - start_time
print(f"torch.load() read from Redis: {torch_read_time:.6f} seconds")
start_time = time.time()

# Deserialize with torch.load
with BytesIO(decompressed_data) as buffer:
    loaded_data_samples, loaded_labels = torch.load(buffer)
deserialize_time = time.time() - start_time
print(f"torch.load() deserialization: {deserialize_time:.6f} seconds")

# Memory size comparisons
torch_memory_size_uncompressed = sys.getsizeof(torch_serialized)
torch_memory_size_compressed = sys.getsizeof(compressed_data)
print(f"Uncompressed size: {torch_memory_size_uncompressed / (1024 * 1024):.6f} MB")
print(f"Compressed size: {torch_memory_size_compressed / (1024 * 1024):.6f} MB")