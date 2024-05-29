import os
import glob
from typing import List
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
from typing import List, Tuple, Dict
import mlworklaods.s3utils as s3utils
from mlworklaods.s3utils import S3Url
import glob
import time
import redis
import io
import base64
import zlib
class TorchLRUTextDataset(Dataset):

    def __init__(self, data_dir: str, tokenizer: PreTrainedTokenizer, transform, block_size: int, batch_size: int, cache_address:str = None):

        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.batch_size = batch_size
        self.transform = transform
        self.current_file_idx = 0
        self.current_position = 0

        self.is_s3: bool = data_dir.startswith("s3://")
        if self.is_s3:
            self.file_list: Dict[str, List[str]] = s3utils.load_unpaired_s3_object_keys(data_dir, False, True)
            self.bucket_name = S3Url(data_dir).bucket
        else:
            self.file_list = glob.glob(f'{self.data_dir}/*.txt')
        self.current_tokens = None #self._load_next_file()
        self.cache_host, self.cache_port = None,None
        if cache_address:
            self.cache_host, self.cache_port = cache_address.split(":")
            self.cache_client = redis.StrictRedis(host=self.cache_host, port=self.cache_port)
            self.use_cache = True
        else:
            self.cache_client = None
            self.use_cache = False
        self.cache_hit = False

    def fetch_from_cache(self, key):
        try:
            return self.cache_client.get(key)
        except:
             return None
        
    def _load_next_file(self):
        if self.current_file_idx >= len(self.file_list):
            # Reset to start reading files from the beginning
            self.current_file_idx = 0
            # raise StopIteration("No more files to read.")
        file_path = self.file_list[self.current_file_idx]
        
        data = None
        if self.use_cache:
            data = self.fetch_from_cache(file_path)

        if data:
            transform_start_time = time.perf_counter()
            decoded_data = base64.b64decode(data) # Decode base64
            decoded_data = zlib.decompress(decoded_data)
            buffer = io.BytesIO(decoded_data)
            tokens = torch.load(buffer)
            transform_duration = time.perf_counter() - transform_start_time
            self.current_file_idx += 1
            self.current_position = 0
            self.cache_hit = True
            return tokens, transform_duration
        
        self.cache_hit = False
        if self.is_s3:
            text = s3utils.get_s3_object(self.bucket_name, file_path)
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        # Apply transform if provided
        transform_start_time = time.perf_counter()
        if self.transform:
            text = self.transform(text)  
        tokens = self.tokenizer(text, truncation=False, padding=False, return_tensors='pt').input_ids.squeeze()
        transform_duration = time.perf_counter() - transform_start_time

        if self.use_cache:
            buffer = io.BytesIO()
            torch.save(tokens, buffer)
            token_data = zlib.compress(buffer.getvalue()) #use_compression:
            token_data = base64.b64encode(token_data).decode('utf-8')
            try:
                self.cache_client.set(file_path, token_data)
            except:
                pass

        self.current_file_idx += 1
        self.current_position = 0
        return tokens, transform_duration

    def _get_next_batch(self):
        
        fetch_start_time = time.perf_counter()
        transform_duration = 0

        if self.current_tokens is None:
            self.current_tokens, transform_duration = self._load_next_file()
        required_size = (self.block_size +1) * self.batch_size
        remaining_tokens = self.current_tokens[self.current_position:]

        while len(remaining_tokens) < required_size:
            next_tokens, transform_duration = self._load_next_file()
            remaining_tokens = torch.cat((remaining_tokens, next_tokens), dim=0)
        
        batch_tokens = remaining_tokens[:required_size]
        self.current_position += required_size
        self.current_tokens = remaining_tokens[required_size:]
        fetch_duration = time.perf_counter() - fetch_start_time

        if self.cache_hit:
            cache_hits = 1
        else:
            cache_hits = 0

        return batch_tokens,fetch_duration, transform_duration, cache_hits

    def __len__(self):
        return len(self.file_list) * 1000  # Placeholder value

    def __getitem__(self, idx):
        batch_tokens, fetch_duration, transform_duration, cache_hits = self._get_next_batch()
        data = batch_tokens[:(self.block_size +1) * self.batch_size].reshape(self.batch_size, (self.block_size+1))
        
        input_ids = data[:, 0 : self.block_size].contiguous().long()
        targets = data[:, 1 : (self.block_size + 1)].contiguous().long()

        return input_ids, targets, fetch_duration, transform_duration, cache_hits

# Example transform function
def example_transform(text: str) -> str:
    return text.lower()  # Example transformation: convert text to lowercase

# Example Usage
if __name__ == "__main__":
    from transformers import GPT2Tokenizer
    from mlworklaods.llm.data import TextTransformations

    data_dir = 's3://openwebtxt/owt/train/'  # Adjust the path to your .txt files
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    block_size = 512
    batch_size = 4

    transformations = TextTransformations()


    dataset = TorchLRUTextDataset(data_dir, tokenizer, None, block_size, batch_size)
    dataloader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=0)

    for input_ids, targets, fetch, transform in dataloader:
        print(f"Input IDs: {input_ids.shape}")
        print(f"targets: {targets.shape}")

        print(f"fetch: {fetch}")
        print(f"transform: {transform}")

        # break
