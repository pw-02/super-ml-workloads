# Copyright The Lightning AI team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from logging import Logger
from time import time
from typing import Any, Dict, List, Optional, Tuple, Union

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

logger = Logger(__name__)


class StreamingDataset(IterableDataset):
    """The streaming dataset can be used once your data have been optimised using the DatasetOptimiser class."""

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
    ) -> None:
        """The streaming dataset can be used once your data have been optimised using the DatasetOptimiser class.

        Arguments:
            input_dir: Path to the folder where the input data is stored.
            item_loader: The logic to load an item from a chunk.
            shuffle: Whether to shuffle the data.
            drop_last: If `True`, drops the last items to ensure that
                all processes/workers return the same amount of data.
                The argument `drop_last` is set to `True` in a distributed setting
                and `False` otherwise.
            seed: Random seed for shuffling.
            serializers: The serializers used to serialize and deserialize the chunks.
            max_cache_size: The maximum cache size used by the StreamingDataset.
            subsample: Float representing fraction of the dataset to be randomly sampled (e.g., 0.1 => 10% of dataset).
            encryption: The encryption object to use for decrypting the data.
            storage_options: Additional connection options for accessing storage services.

        """
        super().__init__()
        if not isinstance(shuffle, bool):
            raise ValueError(f"Shuffle should be a boolean. Found {shuffle}")

        if not isinstance(subsample, float) or subsample <= 0.0 or subsample > 1.0:
            raise ValueError("subsample must be a float with value between 0 and 1.")

        input_dir = _resolve_dir(input_dir)

        self.input_dir = input_dir
        self.subsampled_files: List[str] = []
        self.region_of_interest: List[Tuple[int, int]] = []
        self.subsampled_files, self.region_of_interest = subsample_streaming_dataset(
            self.input_dir, item_loader, subsample, shuffle, seed, storage_options
        )

        self.item_loader = item_loader
        self.shuffle: bool = shuffle
        self.distributed_env = _DistributedEnv.detect()

        if self.distributed_env.world_size > 1:
            if drop_last is False:
                logger.warn(
                    "You're operating within a distributed environment and have disabled the `drop_last` option. "
                    "Please note that this configuration may lead to training interruptions if your system depends "
                    "on distributed collectives."
                )
            else:
                drop_last = True

        self.drop_last = drop_last or False

        self.seed = seed
        self.max_cache_size = max_cache_size

        self.cache: Optional[Cache] = None
        self.worker_env: Optional[_WorkerEnv] = None
        self.worker_chunks: List[int] = []
        self.worker_intervals: List[List[int]] = []
        self.current_indexes: List[int] = []
        self.chunk_index = 0
        self.num_chunks: Optional[int] = None
        self.global_index = 0
        self.index = 0
        self.has_triggered_download = False
        self.min_items_per_replica: Optional[int] = None
        self.current_epoch = 1
        self.random_state = None
        self.shuffler: Optional[Shuffle] = None
        self.serializers = serializers
        self._state_dict: Optional[Dict[str, Any]] = None
        # Has slightly different meaning in the context of the dataset
        # We consider `num_workers = 0` from `torch.utils.DataLoader` still as 1 worker (the main process)
        self.num_workers: int = 1
        self.batch_size: int = 1
        self._encryption = encryption
        self.storage_options = storage_options

    def set_shuffle(self, shuffle: bool) -> None:
        self.shuffle = shuffle

    def set_drop_last(self, drop_last: bool) -> None:
        self.drop_last = drop_last

    def set_epoch(self, current_epoch: int) -> None:
        """Set the current epoch to the dataset on epoch starts.

        When using the StreamingDataLoader, this is done automatically

        """
        # If the state dict has been reloaded, don't override the current epoch
        # The StreamingDataloader would clean this out
        if self._state_dict is None:
            self.current_epoch = current_epoch

    def _create_cache(self, worker_env: _WorkerEnv) -> Cache:
        if _should_replace_path(self.input_dir.path):
            cache_path = _try_create_cache_dir(
                input_dir=self.input_dir.path if self.input_dir.path else self.input_dir.url
            )
            if cache_path is not None:
                self.input_dir.path = cache_path

        cache = Cache(
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

    def _create_shuffler(self, cache: Cache) -> Shuffle:
        seed = self.seed
        drop_last = self.drop_last
        if self._state_dict is not None:
            state: Dict[str, Any] = self._state_dict
            seed = state["seed"]
            drop_last = state["drop_last"]
        return FullShuffle(cache, seed, drop_last) if self.shuffle else NoShuffle(cache, seed, drop_last)

    def __len__(self) -> int:
        return self.get_len(self.num_workers, self.batch_size if self.batch_size else 1)

    def set_batch_size(self, batch_size: int) -> None:
        self.batch_size = batch_size

    def set_num_workers(self, num_workers: int) -> None:
        self.num_workers = num_workers or 1

    def get_len(self, num_workers: int, batch_size: int) -> int:
        self.num_workers = num_workers
        self.batch_size = batch_size
        worker_env = _WorkerEnv.detect()
        if self.shuffler is None:
            cache = self._create_cache(worker_env=worker_env)
            self.shuffler = self._create_shuffler(cache)
        return self.shuffler.get_len(self.distributed_env, num_workers, batch_size, self.current_epoch)

    def __iter__(self) -> "StreamingDataset":
        # When the StreamingDataset is used within map or optimize, let's refetch the distributed env.
        if os.getenv("DATA_OPTIMIZER_GLOBAL_RANK"):
            self.distributed_env = _DistributedEnv.detect()

        self.worker_env = _WorkerEnv.detect()
        self.cache = self._create_cache(worker_env=self.worker_env)
        self.shuffler = self._create_shuffler(self.cache)

        # Handle restart
        if self._state_dict:
            self._validate_state_dict()
            state: Dict[str, Any] = self._state_dict
            self.current_epoch = state["current_epoch"]

        workers_chunks, workers_intervals = self.shuffler.get_chunks_and_intervals_per_workers(
            self.distributed_env, self.worker_env.world_size, self.batch_size, self.current_epoch
        )

        worker_rank = self.distributed_env.global_rank * self.worker_env.world_size + self.worker_env.rank
        self.worker_chunks = workers_chunks[worker_rank]
        self.worker_intervals = workers_intervals[worker_rank]

        # The max number of samples to return from `__next__` (in worker)
        self.stop_length = sum(interval[2] - interval[1] for interval in self.worker_intervals)

        # Handle restart
        if self._state_dict:
            self._resume(workers_chunks, workers_intervals)
        else:
            # Find the chunks shared across all workers of the current node.
            # For each shared chunk, find the rank and worker to use the chunk last and prevent
            # premature deletion for the other workers.
            node_size = self.distributed_env.world_size // self.distributed_env.num_nodes
            first_rank_this_node = (self.distributed_env.global_rank // node_size) * node_size
            num_workers_per_node = node_size * self.num_workers
            worker_start = first_rank_this_node * num_workers_per_node
            worker_end = worker_start + num_workers_per_node
            local_rank = self.distributed_env.global_rank % node_size

            chunks_indexes_skip_deletion = _find_chunks_per_workers_on_which_to_skip_deletion(
                self.num_workers,
                self.batch_size,
                workers_chunks[worker_start:worker_end],
                workers_intervals[worker_start:worker_end],
            )
            worker_node_rank_to_chunk_indexes = _map_node_worker_rank_to_chunk_indexes_to_not_delete(
                chunks_indexes_skip_deletion
            )

            worker_rank_local_node = local_rank * self.num_workers + self.worker_env.rank
            if worker_rank_local_node in worker_node_rank_to_chunk_indexes:
                self.cache._reader.config.skip_chunk_indexes_deletion = worker_node_rank_to_chunk_indexes[
                    worker_rank_local_node
                ]

            self.num_chunks = len(self.worker_chunks)
            self.current_indexes = []
            self.chunk_index = 0
            self.global_index = 0
            self.index = 0

        self.has_triggered_download = False
        self.last_time = time()

        return self

    def _resume(self, workers_chunks: List[List[int]], workers_intervals: List[Any]) -> None:
        assert self._state_dict
        assert self.worker_env
        assert self.shuffler

        state: Dict[str, Any] = self._state_dict

        num_workers = state["num_workers"]
        batch_size = state["batch_size"]

        # TODO: Implement elastic sampling where the number of workers, ranks can change.
        num_samples_yielded = self._state_dict["num_samples_yielded"]

        worker_start = self.distributed_env.global_rank * num_workers
        worker_end = worker_start + num_workers

        # replay sampling from each worker / chunks using the batch size
        indexes = _replay_sampling(num_samples_yielded, batch_size, num_workers)
        chunks_index, indexes = _replay_chunks_sampling(
            workers_intervals={i: workers_intervals[j] for i, j in enumerate(range(worker_start, worker_end))},
            indexes=indexes,
        )

        # select the chunks and intervals associated to this worker
        worker_rank = self.distributed_env.global_rank * self.worker_env.world_size + self.worker_env.rank
        worker_local_rank = self.worker_env.rank

        self.num_chunks = len(workers_intervals[worker_rank])
        self.chunk_index = chunks_index[worker_local_rank]
        self.worker_chunks = workers_chunks[worker_rank]
        self.worker_intervals = workers_intervals[worker_rank]

        # replay the indexes for the current chunks
        interval = self.worker_intervals[self.chunk_index]
        current_indexes = np.arange(interval[1], interval[2])

        # re-shuffle the indexes
        current_indexes = self.shuffler(current_indexes, self.num_chunks, self.current_epoch, self.chunk_index)

        # skip any indexes already consumed
        current_indexes = current_indexes[indexes[worker_local_rank] :]
        self.current_indexes = current_indexes

        self.global_index = indexes[worker_local_rank]

        # bump the chunk_index
        self.chunk_index += 1

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
            return [self.cache[chnk_idx] for chnk_idx in _my_cache_indices]
        return self.cache[index]

    def __next__(self) -> Any:
        # Prevent to create more batch on a given process
        if self.global_index >= self.stop_length:
            self.current_epoch += 1
            raise StopIteration

        # Lazily re-populate the interval to reduce memory usage.
        if len(self.current_indexes) == 0:
            if self.chunk_index == self.num_chunks:
                self.current_epoch += 1
                raise StopIteration

            # reset index
            self.index = 0

            interval = self.worker_intervals[self.chunk_index]
            current_indexes = np.arange(interval[1], interval[2])

            assert self.shuffler is not None
            assert self.num_chunks is not None
            self.current_indexes = self.shuffler(current_indexes, self.num_chunks, self.current_epoch, self.chunk_index)

            self.chunk_index += 1

        # Get the first index
        index = self.current_indexes.pop(0)

        # Call the `__getitem__` method.
        data = self.__getitem__(
            ChunkedIndex(
                index=index,
                chunk_index=self.worker_chunks[self.chunk_index - 1],
                # We provide the chunks indexes only one the first
                chunk_indexes=None if self.has_triggered_download else self.worker_chunks[self.chunk_index - 1 :],
                is_last_index=(self.chunk_index - 1) == len(self.worker_intervals) and len(self.current_indexes) == 1,
            )
        )

        self.has_triggered_download = True
        self.global_index += 1
        self.index += 1

        return data

    def state_dict(self, num_samples_yielded: int, num_workers: int, batch_size: int) -> Dict[str, Any]:
        if _is_in_dataloader_worker():
            raise RuntimeError("The method `state_dict` should only be called in the main process.")

        if self._state_dict is not None:
            self._state_dict["num_samples_yielded"] = num_samples_yielded
            return self._state_dict

        return {
            "num_samples_yielded": num_samples_yielded,
            "num_workers": num_workers,
            "batch_size": batch_size,
            "current_epoch": self.current_epoch,
            "input_dir_path": self.input_dir.path,
            "input_dir_url": self.input_dir.url,
            "item_loader": self.item_loader.state_dict() if self.item_loader else None,
            "drop_last": self.drop_last,
            "seed": self.seed,
            "world_size": self.distributed_env.world_size,
            "shuffle": self.shuffle,
            "subsampled_files": self.subsampled_files,
            "region_of_interest": self.region_of_interest,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if state_dict:
            # the state is restored within the workers
            self._state_dict = state_dict

    def _validate_state_dict(self) -> None:
        assert self._state_dict
        assert self.worker_env
        assert self.cache

        state: Dict[str, Any] = self._state_dict

        if state["shuffle"] != self.shuffle:
            raise ValueError(
                "The provided `shuffle` state doesn't match the current one. "
                f"Found `{self.shuffle}` instead of `{state['shuffle']}`."
            )

        if state["num_workers"] != self.worker_env.world_size:
            raise ValueError(
                "The provided `num_workers` state doesn't match the current one. "
                f"Found `{self.worker_env.world_size}` instead of `{state['num_workers']}`."
            )

        # Note: We need to check whether the path has been resolved to its associated cache.
        # In this case, validate the cache folder is the same.
        if _should_replace_path(state["input_dir_path"]):
            cache_path = _try_create_cache_dir(
                input_dir=state["input_dir_path"] if state["input_dir_path"] else state["input_dir_url"]
            )
            if cache_path != self.input_dir.path:
                raise ValueError(
                    "The provided `input_dir` path state doesn't match the current one. "
                    f"Found `{self.input_dir.path}` instead of `{cache_path}`."
                )
        elif state["input_dir_path"] != self.input_dir.path:
            raise ValueError(
                "The provided `input_dir` path state doesn't match the current one. "
                f"Found `{self.input_dir.path}` instead of `{state['input_dir_path']}`."
            )

        if state["input_dir_url"] != self.input_dir.url:
            raise ValueError(
                "The provided `input_dir` URL state doesn't match the current one. "
                f"Found `{self.input_dir.url}` instead of `{state['input_dir_url']}`."
            )

        if state["seed"] != self.seed:
            raise ValueError(
                "The provided `seed` state doesn't match the current one. "
                f"Found `{self.seed}` instead of `{state['seed']}`."
            )

        if self.item_loader and state["item_loader"] != self.item_loader.state_dict():
            raise ValueError(
                "The provided `item_loader` state doesn't match the current one. "
                f"Found `{self.item_loader.state_dict()}` instead of `{state['item_loader']}`."
            )

        if state["drop_last"] != self.drop_last:
            raise ValueError(
                "The provided `drop_last` state doesn't match the current one. "
                f"Found `{self.drop_last}` instead of `{state['drop_last']}`."
            )

    def reset(self) -> None:
        # undo all the properties associated with original dataset
        default_properties: Dict[str, Any] = {
            "cache": None,
            "worker_env": None,
            "worker_chunks": [],
            "worker_intervals": [],
            "current_indexes": [],
            "chunk_index": 0,
            "num_chunks": None,
            "global_index": 0,
            "index": 0,
            "has_triggered_download": False,
            "min_items_per_replica": None,
            "current_epoch": 1,
            "random_state": None,
            "shuffler": None,
            "_state_dict": None,
            "num_workers": 1,
            "batch_size": 1,
        }

        for prop, value in default_properties.items():
            setattr(self, prop, value)


def is_integer(value: str) -> bool:
    try:
        int(value)
        return True
    except Exception:
        return False


def _replay_sampling(num_samples_yielded: int, batch_size: int, num_workers: int) -> Dict[int, int]:
    """This function replays the sampling from the dataloader."""
    divisible_num_batches_yielded = num_samples_yielded // (num_workers * batch_size)

    indexes = {}
    for worker_idx in range(num_workers):
        indexes[worker_idx] = divisible_num_batches_yielded * batch_size

    num_samples_yielded = num_samples_yielded - (num_workers * divisible_num_batches_yielded * batch_size)

    # take care of the reminder
    worker_idx = 0  # reset the worker_idx
    while True:
        if num_samples_yielded >= batch_size:
            indexes[worker_idx] += batch_size
            worker_idx = (worker_idx + 1) % num_workers
            num_samples_yielded -= batch_size
        else:
            indexes[worker_idx] += num_samples_yielded
            break
    return indexes


def _replay_chunks_sampling(
    workers_intervals: Dict[int, List[Any]], indexes: Dict[int, int]
) -> Tuple[Dict[int, int], Dict[int, int]]:
    chunks_index = {}

    for worker_idx in range(len(workers_intervals)):
        chunks_index[worker_idx] = 0

    for worker_idx, intervals in workers_intervals.items():
        for interval in intervals:
            size = interval[2] - interval[1]
            if indexes[worker_idx] >= size:
                indexes[worker_idx] -= size
                chunks_index[worker_idx] += 1
            else:
                # We've reached the chunk where resuming needs to take place (for this worker)
                break

    return chunks_index, indexes
