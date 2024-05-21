import os
import glob
from typing import List
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer

class TorchLRUTextDataset(Dataset):

    def __init__(self, file_list: List[str], tokenizer: PreTrainedTokenizer, block_size: int, batch_size: int):
        self.file_list = file_list
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.batch_size = batch_size
        self.current_file_idx = 0
        self.current_position = 0
        self.current_tokens = self._load_next_file()

    def _load_next_file(self):
        if self.current_file_idx >= len(self.file_list):
            raise StopIteration("No more files to read.")
        file_path = self.file_list[self.current_file_idx]
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        self.current_file_idx += 1
        self.current_position = 0
        tokens = self.tokenizer(text, truncation=False, padding=False, return_tensors='pt').input_ids.squeeze()
        return tokens

    def _get_next_batch(self):
        required_size = self.block_size * self.batch_size + 1
        remaining_tokens = self.current_tokens[self.current_position:]
        
        while len(remaining_tokens) < required_size:
            if self.current_file_idx < len(self.file_list):
                next_tokens = self._load_next_file()
                remaining_tokens = torch.cat((remaining_tokens, next_tokens), dim=0)
            else:
                if len(remaining_tokens) >= required_size:
                    self.current_tokens = remaining_tokens
                    self.current_position = 0
                else:
                    # Pad remaining tokens to the required size
                    pad_size = required_size - len(remaining_tokens)
                    remaining_tokens = torch.cat((remaining_tokens, torch.zeros(pad_size, dtype=remaining_tokens.dtype)), dim=0)
                break
        
        batch_tokens = remaining_tokens[:required_size]
        self.current_position += required_size
        self.current_tokens = remaining_tokens[required_size:]
        
        return batch_tokens

    def __len__(self):
        return len(self.file_list) * 1000  # Placeholder value

    def __getitem__(self, idx):
        batch_tokens = self._get_next_batch()
        input_ids = batch_tokens[:self.block_size * self.batch_size].reshape(self.batch_size, self.block_size)
        labels = batch_tokens[1:self.block_size * self.batch_size + 1].reshape(self.batch_size, self.block_size)
        return input_ids, labels

def collate_fn(batch):
    input_ids = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    return input_ids, labels

# Example Usage
if __name__ == "__main__":
    from transformers import GPT2Tokenizer

    file_list = glob.glob('data/openwebtext/train/*.txt')  # Adjust the path to your .txt files
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    block_size = 6
    batch_size = 1

    dataset = TorchLRUTextDataset(file_list, tokenizer, block_size, batch_size)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn, shuffle=False, num_workers=0)

    for input_ids, labels in dataloader:
        print(f"Input IDs: {input_ids.shape}")
        print(f"Labels: {labels.shape}")
        # break
