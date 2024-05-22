import os
import glob
from typing import List
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
from typing import List, Tuple, Dict
import mlworklaods.image_classification.s3utils as s3utils
from mlworklaods.image_classification.s3utils import S3Url
import glob

class TorchLRUTextDataset(Dataset):

    def __init__(self, data_dir: str, tokenizer: PreTrainedTokenizer, transform, block_size: int, batch_size: int):

        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.batch_size = batch_size
        self.transform = transform
        self.current_file_idx = 0
        self.current_position = 0

        self.is_s3: bool = data_dir.startswith("s3://")
        if self.is_s3:
            self.file_list: Dict[str, List[str]] = s3utils.load_unpaired_s3_object_keys(data_dir, False)
            self.bucket_name = S3Url(data_dir).bucket
        else:
            self.file_list = glob.glob(f'{self.data_dir}/*.txt')
        self.current_tokens = None #self._load_next_file()

    def _load_next_file(self):
        if self.current_file_idx >= len(self.file_list):
            # Reset to start reading files from the beginning
            self.current_file_idx = 0
            # raise StopIteration("No more files to read.")

        file_path = self.file_list[self.current_file_idx]
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
                    # Apply transform if provided
        
        if self.transform:
            text = self.transform(text)

        self.current_file_idx += 1
        self.current_position = 0
        tokens = self.tokenizer(text, truncation=False, padding=False, return_tensors='pt').input_ids.squeeze()
        return tokens

    def _get_next_batch(self):
        if self.current_tokens is None:
            self.current_tokens = self._load_next_file()
        required_size = (self.block_size +1) * self.batch_size
        remaining_tokens = self.current_tokens[self.current_position:]

        while len(remaining_tokens) < required_size:
            next_tokens = self._load_next_file()
            remaining_tokens = torch.cat((remaining_tokens, next_tokens), dim=0)
        
        batch_tokens = remaining_tokens[:required_size]
        self.current_position += required_size
        self.current_tokens = remaining_tokens[required_size:]
        
        return batch_tokens

    def __len__(self):
        return len(self.file_list) * 1000  # Placeholder value

    def __getitem__(self, idx):
        batch_tokens = self._get_next_batch()
        data = batch_tokens[:(self.block_size +1) * self.batch_size].reshape(self.batch_size, (self.block_size+1))
        
        input_ids = data[:, 0 : self.block_size].contiguous().long()
        targets = data[:, 1 : (self.block_size + 1)].contiguous().long()

        return input_ids, targets

# Example transform function
def example_transform(text: str) -> str:
    return text.lower()  # Example transformation: convert text to lowercase

# Example Usage
if __name__ == "__main__":
    from transformers import GPT2Tokenizer
    from mlworklaods.llm.data import TextTransformations

    data_dir = 'data/openwebtext/test/train'  # Adjust the path to your .txt files
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    block_size = 512
    batch_size = 4

    transformations = TextTransformations()


    dataset = TorchLRUTextDataset(data_dir, tokenizer, transformations.normalize, block_size, batch_size)
    dataloader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=0)

    for input_ids, targets in dataloader:
        print(f"Input IDs: {input_ids.shape}")
        print(f"targets: {targets.shape}")

        # break
