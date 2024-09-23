
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
# from litdata.streaming import Cache
from litdata.streaming.downloader import get_downloader_cls  # noqa: F401
from litdata.streaming.item_loader import BaseItemLoader
from litdata.streaming.resolver import Dir, _resolve_dir
from litdata.streaming.sampler import ChunkedIndex
from litdata.streaming.serializers import Serializer
from litdata.utilities.dataset_utilities import _should_replace_path, _try_create_cache_dir, subsample_streaming_dataset
from litdata.utilities.encryption import Encryption
from litdata.utilities.env import _DistributedEnv, _is_in_dataloader_worker, _WorkerEnv
from litdata.streaming import StreamingDataset
from litdata.streaming import Cache
from litdata.streaming.reader import BinaryReader, PrepareChunksThread, PyTreeLoader
import time



logger = Logger(__name__)

class CustomCache(Cache):
    def __init__(
        self,
        input_dir: Optional[Union[str, Dir]],
        subsampled_files: Optional[List[str]] = None,
        region_of_interest: Optional[List[Tuple[int, int]]] = None,
        compression: Optional[str] = None,
        encryption: Optional[Encryption] = None,
        chunk_size: Optional[int] = None,
        chunk_bytes: Optional[Union[int, str]] = None,
        item_loader: Optional[BaseItemLoader] = None,
        max_cache_size: Union[int, str] = "100GB",
        serializers: Optional[Dict[str, Serializer]] = None,
        writer_chunk_index: Optional[int] = None,
        storage_options: Optional[Dict] = {},
    ) -> None:
        # Call the superclass constructor
        super().__init__(
            input_dir=input_dir,
            subsampled_files=subsampled_files,
            region_of_interest=region_of_interest,
            compression=compression,
            encryption=encryption,
            chunk_size=chunk_size,
            chunk_bytes=chunk_bytes,
            item_loader=item_loader,
            max_cache_size=max_cache_size,
            serializers=serializers,
            writer_chunk_index=writer_chunk_index,
            storage_options=storage_options,
        )

    def __getitem__(self, index: Union[int, ChunkedIndex]) -> Dict[str, Any]:
        """Read an item in the reader."""
        if isinstance(index, int):
            index = ChunkedIndex(*self._get_chunk_index_from_index(index))
        
        if not isinstance(index, ChunkedIndex):
            raise ValueError("The Reader.read(...) method expects a chunked Index.")

        # Load the config containing the index
        if self._reader._config is None and self._reader._try_load_config() is None:
            raise Exception("The reader index isn't defined.")

        if self._reader._config and (self._reader._config._remote_dir or self._reader._config._compressor):
            # Create and start the prepare chunks thread
            if self._reader._prepare_thread is None and self._reader._config:
                self._reader._prepare_thread = PrepareChunksThread(
                    self._reader._config, self._reader._item_loader, self._reader._distributed_env, self._reader._max_cache_size
                )
                self._reader._prepare_thread.start()
                if index.chunk_indexes:
                    self._reader._prepare_thread.download(index.chunk_indexes)

            # If the chunk_index is new, request for it to be downloaded.
            if index.chunk_index != self._reader._last_chunk_index:
                assert self._reader._prepare_thread
                self._reader._prepare_thread.download([index.chunk_index])

            if self._reader._last_chunk_index is None:
                self._reader._last_chunk_index = index.chunk_index

        # Fetch the element
        chunk_filepath, begin, chunk_bytes = self._reader.config[index]
        
        is_cache_hit = os.path.exists(chunk_filepath) and os.stat(chunk_filepath).st_size >= chunk_bytes

        if isinstance(self._reader._item_loader, PyTreeLoader):
            item = self._reader._item_loader.load_item_from_chunk(
                index.index, index.chunk_index, chunk_filepath, begin, chunk_bytes, self._reader._encryption
            )
        else:
            item = self._reader._item_loader.load_item_from_chunk(
                index.index, index.chunk_index, chunk_filepath, begin, chunk_bytes
            )

        # We need to request deletion after the latest element has been loaded.
        # Otherwise, this could trigger segmentation fault error depending on the item loader used.
        if (
            self._reader._config
            and (self._reader._config._remote_dir or self._reader._config._compressor)
            and index.chunk_index != self._reader._last_chunk_index
        ):
            assert self._reader._prepare_thread
            assert self._reader._last_chunk_index is not None

            # inform the chunk has been completely consumed
            self._reader._prepare_thread.delete([self._reader._last_chunk_index])

            # track the new chunk index as the latest one
            self._reader._last_chunk_index = index.chunk_index

        if index.is_last_index and self._reader._prepare_thread:
            # inform the thread it is time to stop
            self._reader._prepare_thread.stop()
            self._reader._prepare_thread = None

        return item, is_cache_hit


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
        max_cache_size: Union[int, str] = "10GB",
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
    
    def _create_cache(self, worker_env: _WorkerEnv) -> CustomCache:
        if _should_replace_path(self.input_dir.path):
            cache_path = _try_create_cache_dir(
                input_dir=self.input_dir.path if self.input_dir.path else self.input_dir.url
            )
            if cache_path is not None:
                self.input_dir.path = cache_path

        cache = CustomCache(
            input_dir=self.input_dir,
            subsampled_files=self.subsampled_files,
            region_of_interest=self.region_of_interest,
            item_loader=self.item_loader,
            chunk_bytes=1,
            serializers=self.serializers,
            max_cache_size=self.max_cache_size,
            encryption=self._encryption,
            storage_options=self.storage_options,
        )
        cache._reader._try_load_config()

        if not cache.filled:
            raise ValueError(
                f"The provided dataset `{self.input_dir}` doesn't contain any {_INDEX_FILENAME} file."
                " HINT: Did you successfully optimize a dataset to the provided `input_dir`?"
            )

        return cache

    
    
    def __getitem__(self, index: Union[ChunkedIndex, int]) -> Any:

        start_loading_time = time.perf_counter()

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
            data,is_cache_hit = [self.cache[chnk_idx] for chnk_idx in _my_cache_indices]
        data, is_cache_hit = self.cache[index]
        data_loading_time = time.perf_counter() - start_loading_time

        start_transformation_time = time.perf_counter()
        tokenized_data = self.tokenize_data(data)
            # Create input_ids and targets
        if len(tokenized_data) < self.seq_length + 1:
            padding_length = (self.seq_length + 1) - len(tokenized_data)
            padding = torch.full((padding_length,), fill_value=0, dtype=torch.long)  # Use 0 or your padding token index
            tokenized_data = torch.cat((tokenized_data, padding))

        input_ids = tokenized_data[0:self.seq_length].contiguous().long()
        targets = tokenized_data[1:self.seq_length + 1].contiguous().long()
        transformation_time =  time.perf_counter() - start_transformation_time

        # print(input_ids.size(), targets.size())
         # Calculate data loading time excluding transformation time     
        return (input_ids, targets), data_loading_time, transformation_time, is_cache_hit
    
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