import redis
from io import BytesIO
import zlib
import pickle
import torch
import matplotlib.pyplot as plt
import boto3
from PIL import Image
from torchvision.transforms import transforms
import numpy as np
import albumentations as A

s3_client = boto3.client('s3')



def test_fetch_from_cache():    
    # Connect to Redis
    cache_host, cache_port = '54.184.21.219:6378'.split(":")
    redis_client = redis.StrictRedis(host=cache_host, port=int(cache_port))

    # Retrieve compressed data from Redis
    compressed_minibatch = redis_client.get('1_1_1_6432794cc0dcc31ae4cac1ac0dcb98c3ea9d12ba')

    with BytesIO(zlib.decompress(compressed_minibatch)) as buffer:
        minibatch_np, labels_np = pickle.load(buffer)
 
    # Convert NumPy arrays to PyTorch tensors
    labels_torch = torch.from_numpy(labels_np).long()
    minibatch_torch = torch.from_numpy(minibatch_np)
        
    for image in tensor_to_pil_images(minibatch_torch):
        image.show()
    pass


def create_albumentations_batch(data, transform):
    bucket_name = data.get('bucket_name')
    batch_samples = data.get('batch_samples')
    sample_data, sample_labels = [], []
    for data_sample in batch_samples:
        sample_path, sample_label = data_sample
        obj = s3_client.get_object(Bucket=bucket_name, Key=sample_path)
        data = Image.open(BytesIO(obj['Body'].read())).convert("RGB")
        data = np.array(data)
        data = transform(image=data)['image']  # Albumentations returns a dictionary
        sample_data.append(data)
        sample_labels.append(sample_label)

    minibatch_np = np.stack(sample_data)
    labels_np = np.array(sample_labels)
    
    # Ensure correct shape by permuting dimensions
    minibatch_np = minibatch_np.transpose(0, 3, 1, 2)  # Convert to [batch_size, channels, height, width]

    with BytesIO() as buffer:
        pickle.dump((minibatch_np, labels_np), buffer)
        compressed_minibatch = zlib.compress(buffer.getvalue())
    
    with BytesIO(zlib.decompress(compressed_minibatch)) as buffer:
        minibatch_np, labels_np = pickle.load(buffer)

    labels_torch = torch.from_numpy(labels_np).long()
    minibatch_torch = torch.from_numpy(minibatch_np)

    return minibatch_torch, labels_torch

def create_torch_batch(data, transform):
    bucket_name = data.get('bucket_name')
    batch_samples = data.get('batch_samples')
    sample_data, sample_labels = [], []
    for data_sample in batch_samples:
        sample_path, sample_label = data_sample
        obj = s3_client.get_object(Bucket=bucket_name, Key=sample_path)
        data = Image.open(BytesIO(obj['Body'].read())).convert("RGB")

        data = transform(data)

        sample_data.append(data)
        sample_labels.append(sample_label)

    return torch.stack(sample_data), torch.tensor(sample_labels).long()

def tensors_are_equal(tensor1, tensor2):
    if tensor1.size() != tensor2.size():
        print("Tensors have different shapes.")
        return False
    return torch.allclose(tensor1, tensor2)


def tensor_to_pil_images(tensor_batch):
    # Ensure tensor is on CPU
    tensor_batch = tensor_batch.cpu()
    
    # Convert tensor to numpy array
    numpy_batch = tensor_batch.numpy()
    
    # Swap the dimensions from [batch_size, channels, height, width] to [batch_size, height, width, channels]
    numpy_batch = np.transpose(numpy_batch, (0, 2, 3, 1))
    
    # Convert each numpy image to a PIL Image
    pil_images = [Image.fromarray((img * 255).astype(np.uint8)) for img in numpy_batch]
    
    return pil_images

if __name__ == '__main__':

    test_fetch_from_cache()
    # Define the data dictionary with detailed formatting
    data = {
        "bucket_name": "sdl-cifar10",
        "batch_id": 1279,
        "batch_samples": [
            ["train/Airplane/attack_aircraft_s_001210.png", 0],
            # ["train/Airplane/attack_aircraft_s_001210.png", 0],
            # ["train/Airplane/attack_aircraft_s_001210.png", 0],
            # ["train/Airplane/attack_aircraft_s_001210.png", 0],
            # ["train/Airplane/attack_aircraft_s_001210.png", 0]
        ]
    }
    
    torch_data, torch_labels = create_torch_batch(data, transforms.Compose([
            transforms.Resize(256),
            # # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]))
    
    other_data, other_labels = create_albumentations_batch(data, A.Compose([
        A.Resize(256, 256),
        # # A.RandomResizedCrop(224, 224),
        # A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))
    
    # Convert tensors to numpy arrays for comparison
    torch_data_np = torch_data.numpy()
    other_data_np = other_data.numpy()
    
    # Check if tensors are equivalent
    print("Checking if data tensors are equivalent...")
    data_equal = tensors_are_equal(torch_data, other_data)
    print(f"Data tensors are equivalent: {data_equal}")

    print("Checking if label tensors are equivalent...")
    labels_equal = tensors_are_equal(torch_labels, other_labels)
    print(f"Label tensors are equivalent: {labels_equal}")


    for image in tensor_to_pil_images(torch_data):
        image.show()

    for image in tensor_to_pil_images(other_data):
        image.show()
    pass