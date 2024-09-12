
import os
from logging import Logger
from time import time
from typing import Any, Dict, List, Optional, Tuple, Union
from litgpt.tokenizer import Tokenizer
import torch
import numpy as np
from torch.utils.data import IterableDataset

from litdata.constants import (
    _INDEX_FILENAME,
)
from litdata.streaming import Cache
from litdata.streaming.downloader import get_downloader_cls  # noqa: F401
from litdata.streaming.item_loader import BaseItemLoader
from litdata.streaming.resolver import Dir, _resolve_dir
from litdata.streaming.sampler import ChunkedIndex
from litdata.streaming.serializers import Serializer
from litdata.streaming.shuffle import FullShuffle, NoShuffle, Shuffle
from litdata.utilities.dataset_utilities import _should_replace_path, _try_create_cache_dir, subsample_streaming_dataset
from litdata.utilities.encryption import Encryption
from litdata.utilities.env import _DistributedEnv, _is_in_dataloader_worker, _WorkerEnv
from litdata.utilities.shuffle import (
    _find_chunks_per_workers_on_which_to_skip_deletion,
    _map_node_worker_rank_to_chunk_indexes_to_not_delete,
)
from litdata.streaming import StreamingDataLoader, TokensLoader, StreamingDataset


logger = Logger(__name__)

class CustomStreamingDataset(StreamingDataset):
    """Customized version of StreamingDataset with overridden methods."""

    def __init__(
        self,
        input_dir: Union[str, "Dir"],
        item_loader: Optional[BaseItemLoader] = None,
        shuffle: bool = False,
        drop_last: Optional[bool] = None,
        seed: int = 42,
        serializers: Optional[Dict[str, Serializer]] = None,
        max_cache_size: Union[int, str] = "100GB",
        subsample: float = 1.0,
        encryption: Optional[Encryption] = None,
        storage_options: Optional[Dict] = {},
        tokenizer: Optional[Tokenizer] = None,
        batch_size: int = 1,
        seq_length: int = 2048

    ) -> None:
        # Call the superclass constructor
        super().__init__(
            input_dir=input_dir,
            item_loader=item_loader,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=seed,
            serializers=serializers,
            max_cache_size=max_cache_size,
            subsample=subsample,
            encryption=encryption,
            storage_options=storage_options,
        )
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.seq_length = seq_length  # Increase by one because we need the next token as well
    
    def _load_more_data(self) -> Any:
        """Method to load more data from the source if the current data is insufficient."""
        # Implement data loading logic here
        try:
            # Fetch the next chunk or file of data
            more_data = super().__next__()  # Or use another method to load more data
            return more_data
        except StopIteration:
            self.data_loader_exhausted = True
            return None
    
    
    def __getitem__(self, index: Union[ChunkedIndex, int]) -> Any:
        if self.cache is None:
            self.worker_env = _WorkerEnv.detect()
            self.cache = self._create_cache(worker_env=self.worker_env)
            self.shuffler = self._create_shuffler(self.cache)
        if isinstance(index, int):
            index = ChunkedIndex(*self.cache._get_chunk_index_from_index(index))
        elif isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            _my_indices = list(range(start, stop, step))
            _my_cache_indices = [ChunkedIndex(*self.cache._get_chunk_index_from_index(idx)) for idx in _my_indices]
            data = [self.cache[chnk_idx] for chnk_idx in _my_cache_indices]
        data = self.cache[index]
        tokenized_data = self.tokenize_data(data)
            # Create input_ids and targets
        if len(tokenized_data) < self.seq_length + 1:
            padding_length = (self.seq_length + 1) - len(tokenized_data)
            padding = torch.full((padding_length,), fill_value=0, dtype=torch.long)  # Use 0 or your padding token index
            tokenized_data = torch.cat((tokenized_data, padding))

        input_ids = tokenized_data[0:self.seq_length].contiguous().long()
        targets = tokenized_data[1:self.seq_length + 1].contiguous().long()
        print(input_ids.size(), targets.size())
        return (input_ids, targets)
    
    def tokenize_data(self, data: Any) -> Any:
        return self.tokenizer.encode(data, eos=True)




if __name__ == "__main__":
    # Define the input directory
    input_dir = "path/to/data"
    
    # Initialize the custom streaming dataset
    custom_dataset = CustomStreamingDataset(input_dir=input_dir, shuffle=True)
    
    # Iterate over the dataset
    for item in custom_dataset:
        # Custom processing or logging
        print(item)