# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
import os
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Optional, Union, Any
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from abc import abstractmethod
import torch


class DataModule(LightningDataModule):
    """Base class for all data modules in LitGPT."""

    @abstractmethod
    def connect(
        self, tokenizer: Optional[Any] = None, batch_size: int = 1, max_seq_length: Optional[int] = None
    ) -> None:
        """All settings that can't be determined at the time of instantiation need to be passed through here
        before any dataloaders can be accessed.
        """

    def setup(self, stage: str = "") -> None:
        # Stub is to redefine the default signature, because the concept of 'stage' does not exist in LitGPT
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


@dataclass
class OpenWebText(DataModule):
    """The OpenWebText data module for pretraining."""

    data_path: Union[str, Path] = Path("data/openwebtext")
    """The path to the data directory, containing two folders 'train' and 'val'
    which are the output of the preprocessing step. The path can also be a remote path (e.g., s3://)."""
    val_split_fraction: float = 0.0005
    """The fraction of data that should be put aside for validation."""
    seed: int = 42
    """The seed to use for shuffling the training data."""
    num_workers: int = 8
    """The number of workers to use for the dataloaders."""

    tokenizer: Optional[Any] = field(default=None, repr=False, init=False)
    batch_size: int = field(default=1, repr=False, init=False)
    seq_length: int = field(default=2048, repr=False, init=False)

    def __post_init__(self) -> None:
        # Could be a remote path (s3://) or a local path
        self.data_path_train = str(self.data_path).rstrip("/") + "/train"
        self.data_path_val = str(self.data_path).rstrip("/") + "/val"

    def connect(
        self, tokenizer: Optional[Any] = None, batch_size: int = 1, max_seq_length: Optional[int] = 2048
    ) -> None:
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.seq_length = max_seq_length + 1  # Increase by one because we need the next token as well

    def decode(self, tensor: torch.Tensor) -> str:
        tokens = [tensor.item()] if tensor.ndim == 0 else tensor.tolist()
        return self.processor.decode(tokens)
    
    def prepare_data(self) -> None:
        from datasets import Dataset, load_dataset
        from litdata import optimize

        if str(self.data_path).startswith("s3://"):
            print(f"The OpenWebText data path points to an S3 location: {self.data_path}. Skipping preprocessing.")
            return

        if Path(self.data_path_train).is_dir() and Path(self.data_path_val).is_dir():
            print(f"Found OpenWebText train and val dir: {self.data_path}. Skipping preprocessing.")
            return

        dataset = load_dataset("openwebtext", num_proc=(os.cpu_count() // 2), trust_remote_code=True)

        # Split the data in training and validation
        split_dataset = dataset["train"].train_test_split(
            test_size=self.val_split_fraction, seed=self.seed, shuffle=True
        )
        split_dataset["val"] = split_dataset.pop("test")  # rename the test split to val

        def tokenize(data: Dataset, index: int, eos:bool=True):
            # yield tokenizer(data[index]["text"], truncation=False, padding=True, return_tensors='pt').input_ids.squeeze()
            encoded_input = tokenizer(data[index]["text"], truncation=False, padding=False, return_tensors='pt')
            eos_token_id = tokenizer.eos_token_id
            if eos:
                input_ids_with_eos = torch.cat([encoded_input.input_ids, torch.tensor([[eos_token_id]])], dim=1)
                yield input_ids_with_eos
            else:
                yield encoded_input.input_ids

        optimize(
            fn=partial(tokenize, split_dataset["train"]),
            inputs=list(range(len(split_dataset["train"]))),
            output_dir=self.data_path_train,
            num_workers=(os.cpu_count() - 1),
            chunk_bytes="5MB",
        )
        optimize(
            fn=partial(tokenize, split_dataset["val"]),
            inputs=list(range(len(split_dataset["val"]))),
            output_dir=self.data_path_val,
            num_workers=(os.cpu_count() - 1),
            chunk_bytes="5MB",
        )

    def train_dataloader(self) -> DataLoader:
        from litdata.streaming import StreamingDataLoader, StreamingDataset, TokensLoader

        train_dataset = StreamingDataset(
            input_dir=self.data_path_train,
            item_loader=TokensLoader(block_size=self.seq_length),
            shuffle=True,
            drop_last=True,
        )
        train_dataloader = StreamingDataLoader(
            train_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=self.num_workers, drop_last=True
        )
        return train_dataloader

    def val_dataloader(self) -> DataLoader:
        from litdata.streaming import StreamingDataset, TokensLoader

        val_dataset = StreamingDataset(
            input_dir=self.data_path_val,
            item_loader=TokensLoader(block_size=self.seq_length),
            shuffle=True,
            # Consider setting to False, but we would lose some samples due to truncation when world size > 1
            drop_last=True,
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=self.num_workers, drop_last=True
        )
        return val_dataloader


if __name__ == "__main__":
    from transformers import AutoTokenizer

    data_path = Path('data/openwebtext/litdata_example')
    batch_size = 32
    # tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # Specify the tokenizer name
    tokenizer_name = "EleutherAI/pythia-14m"

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    max_seq_length = 512
    tkn = tokenizer.eos_token_id
    data = OpenWebText(
        data_path=data_path,
        seed=42,
        num_workers=8,
        # batch_size= batch_size,
        # tokenizer = tokenizer,
        # max_seq_length = max_seq_length
    )

    data.connect(tokenizer=tokenizer, batch_size=batch_size, max_seq_length=max_seq_length)
    # text_files.prepare_data()
    data.setup()
    dataloader = data.train_dataloader()
    # dataloader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=0)

    for idx, (batch) in enumerate (dataloader):
        print(f"Input idx: {idx}")
        # print(f"targets: {targets.shape}")

        # print(f"fetch: {fetch}")
        # print(f"transform: {transform}")

        # break