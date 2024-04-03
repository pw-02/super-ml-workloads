import functools
import mlworklaods.super_dl.s3utils as s3utils
from  mlworklaods.super_dl.s3utils import S3Url
from typing import List, Tuple, Dict
from torch.utils.data import SequentialSampler, IterableDataset, RandomSampler, DataLoader
import torchvision
import torch
import torch.nn.functional as F
import tiktoken

class S3TextIterableDataset(IterableDataset):
    def __init__(self,data_dir:str, tokenizer, block_size:int, shuffle = False):
        super().__init__()
        self.epoch = 0
        self.block_size = block_size
        self.shuffle_urls = shuffle
        # if dataset_kind == 'image':
        self.samples:List[str] = s3utils.load_unpaired_s3_object_keys(data_dir, False, True)
        self.bucket_name = S3Url(data_dir).bucket
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)
    
    def tokenize(self, text):
        ids = self.tokenizer.encode_ordinary(text) # encode_ordinary ignores any special tokens
        #ids.append(self.tokenizer.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
        # print(f"tokens: {len(ids)}")

        # Tokenize text into chunks of block_size
        chunks = []
        start_idx = 0
        while start_idx < len(ids):
            end_idx = min(start_idx + self.block_size, len(ids))
            x = torch.tensor(ids[start_idx:end_idx], dtype=torch.long)
            y = torch.tensor(ids[start_idx+1:end_idx+1], dtype=torch.long)
            if len(x) < self.block_size:
                # print(len(ids) + (self.block_size - len(x)))
                x = F.pad(x, (0, self.block_size - len(x)))
            if len(y) < self.block_size:
                y = F.pad(y, (0, self.block_size - len(y)))
         
            chunks.append((x, y))
            start_idx = end_idx

        return chunks


    def __iter__(self):

        if self.shuffle_urls:
            sampler = RandomSampler(self)
        else:
            sampler = SequentialSampler(self)
        
        for idx in sampler:
            file_path = self.samples[idx]
            sample_input = s3utils.get_s3_object(self.bucket_name, file_path)
            tokenized_chunks = self.tokenize(sample_input)
            for x, y in tokenized_chunks:
                yield x, y

    
    def set_epoch(self, epoch):
        self.epoch = epoch
if __name__ == "__main__":
    def get_batch_size_mb(batch_tensor):
        import sys
        # Get the size of the tensor in bytes
        size_bytes = sys.getsizeof(batch_tensor.storage()) + sys.getsizeof(batch_tensor)
        # Convert bytes to megabytes
        size_mb = size_bytes / (1024 ** 2)
        # Convert bytes to kb
        size_in_kb = size_bytes / 1024
        return size_mb,size_in_kb

    # # Example usage
    train_data_dir = 's3://openwebtxt/owt/train/'
    block_size = 2048

    dataset = S3TextIterableDataset(data_dir=train_data_dir,
                                    tokenizer=tiktoken.get_encoding("gpt2"),
                                    block_size=2048,
                                    shuffle=True)

    data_loader = DataLoader(dataset, batch_size=12)
    # Get the size of the tensor using pympler
    for input, target in data_loader:
        batch_size_mb,size_in_kb  = get_batch_size_mb(input)
        print(f"Batch size: {batch_size_mb:.2f} MB, {size_in_kb:.2f} KB")
        print(input.shape, target.shape)
