import torch
from torch.utils.data import Dataset
from super_dl.s3_tasks import S3Helper
import tiktoken
import functools
from torch.utils.data import Dataset, SequentialSampler, DataLoader,IterableDataset, RandomSampler
import torch.nn.functional as F
from super_dl.datasets.super_base_dataset import BaseSUPERDataset, timer_decorator


class SUPERTextDataset(BaseSUPERDataset, IterableDataset):
    def __init__(self, job_id, data_dir: str, tokenizer, block_size, cache_address=None):
        super().__init__(job_id, data_dir, data_dir.startswith("s3://"), cache_address)
        self.block_size = block_size
        self.tokenizer = tokenizer

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
        random_sampler = RandomSampler(self)
        for idx in random_sampler:
            file_key, label = self._classed_items[idx]
            text = self.s3_helper.load_s3_file_into_memory(self.bucket_name, file_key)
            tokenized_chunks = self.tokenize(text)
            for x, y in tokenized_chunks:
                yield x, y
        
if __name__ == "__main__":

# # Example usage
    train_data_dir = 's3://openwebtxt/owt/train/'
    block_size = 2048
    dataset = SUPERTextDataset(1, train_data_dir, block_size,None)
    sampler = SequentialSampler(dataset)

    data_loader = DataLoader(dataset, batch_size=5)

    for input, target in data_loader:
        print(input.shape, target.shape)

