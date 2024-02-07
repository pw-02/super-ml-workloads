import torch
from torch.utils.data import IterableDataset, DataLoader, RandomSampler, SequentialSampler,Sampler
from botocore.exceptions import NoCredentialsError
import boto3
import io

class SUPERS3TextDataset(IterableDataset):
    def __init__(self, s3_bucket, s3_prefix, block_size):
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.block_size = block_size
        self.s3_client = boto3.client('s3')
        self.file_list = self.get_file_list()
        shuffle: bool = False,
        drop_last: bool = False,
        seed: int = 42,
        self.shuffle: bool = shuffle
        self.drop_last = drop_last
        self.seed = seed




class S3TextDataset(IterableDataset):
    def __init__(self, s3_bucket, s3_prefix, block_size):
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.block_size = block_size
        self.s3_client = boto3.client('s3')
        self.file_list = self.get_file_list()
        shuffle: bool = False,
        drop_last: bool = False,
        seed: int = 42,
        self.shuffle: bool = shuffle
        self.drop_last = drop_last
        self.seed = seed
    



    def get_file_list(self):
        response = self.s3_client.list_objects_v2(Bucket=self.s3_bucket, Prefix=self.s3_prefix)
        return [obj['Key'] for obj in response.get('Contents', [])]

    def load_file(self, file_key):
        try:
            response = self.s3_client.get_object(Bucket=self.s3_bucket, Key=file_key)
            return response['Body'].read().decode('utf-8')
        except NoCredentialsError:
            print("Credentials not available")

    def process_data(self, text):
        # Add any data processing logic here
        return text

    def __iter__(self):
        for file_key in self.file_list:
            text = self.load_file(file_key)
            processed_data = self.process_data(text)

            # Yield contiguous blocks of data
            for i in range(0, len(processed_data) - self.block_size + 1, self.block_size):
                yield processed_data[i:i + self.block_size]
    
    # def __len__(self):
    #     # Return the total number of samples in your dataset
    #     return sum(len(processed_data) - self.block_size + 1 for processed_data in map(self.process_data, map(self.load_file, self.file_list)))


# Set your S3 bucket, prefix, and other parameters
s3_bucket = 'openweb2000'
s3_prefix = ''
block_size = 128
batch_size = 32

# Create S3TextDataset and DataLoader with CustomSequentialSampler
s3_dataset = S3TextDataset(s3_bucket, s3_prefix, block_size)
s3_data_loader = DataLoader(s3_dataset, batch_size=batch_size)

# Iterate through the DataLoader during training
for batch in s3_data_loader:
    # Your training logic here
    print(batch)