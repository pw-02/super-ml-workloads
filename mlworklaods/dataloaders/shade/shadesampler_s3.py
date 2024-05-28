import math
from typing import TypeVar, Optional, Iterator
import torch
from torch.utils.data import Dataset, Sampler
import torch.distributed as dist
import numpy as np 
import random
import redis
from rediscluster import RedisCluster
T_co = TypeVar('T_co', covariant=True)
import hashlib
import random
from torch.utils.data import SequentialSampler, RandomSampler, BatchSampler, DistributedSampler
from typing import Iterator, List, Tuple, Union

import math
from typing import TypeVar, Optional, Iterator
import torch
from torch.utils.data import Dataset, Sampler
import torch.distributed as dist
import numpy as np 
import random
import redis
from rediscluster import RedisCluster
T_co = TypeVar('T_co', covariant=True)

class ShadeSamplerS3(Sampler[T_co]):

      def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                         rank: Optional[int] = None, shuffle: bool = True, PADS: bool = True, batch_size: int = 64,
                         seed: int = 0, drop_last: bool = False, replacement = True, host_ip = None, port_num = None, rep_factor = 1, init_fac = 1, ls_init_fac = 1e-2 ) -> None:
            if num_replicas is None:
                  if not dist.is_available():
                        raise RuntimeError("Requires distributed package to be available")
                  num_replicas = dist.get_world_size()
            if rank is None:                      
                  if not dist.is_available():
                        raise RuntimeError("Requires distributed package to be available")
                  rank = dist.get_rank()
            if rank >= num_replicas or rank < 0:
                  raise ValueError(
                        "Invalid rank {}, rank should be in the interval"
                        " [0, {}]".format(rank, num_replicas - 1))
            if host_ip is None:
                  raise RuntimeError("Requires Redis Host Node IP.")
            # if port_num is None:
            #       raise RuntimeError("Requires Redis Port Number.")

            self.dataset = dataset
            self.num_replicas = num_replicas
            self.rank = rank
            self.epoch = 0
            self.drop_last = drop_last
            self.batch_wts = []

            self.num_samples = len(dataset)
            
            #initialize importance sampling factors
            self.init_fac, self.ls_init_fac = init_fac, ls_init_fac
            #initialize weights of all of the indices
            self.weights = torch.ones(self.num_samples)*init_fac
            #sampling with replacement
            self.replacement = replacement

            #variable for understanding the portion of processed indices.
            self.item_curr_pos = 0
            self.batch_size = batch_size

            #fix the weights of indices in the log scale for rank-based importance.
            for j in range(batch_size):
                  self.batch_wts.append(math.log(j+10))
            self.batch_wts = torch.tensor(self.batch_wts)
            self.ls_param = 0

            #starting the cache.
            if host_ip == '0.0.0.0':
                  self.key_id_map = redis.StrictRedis(host=host_ip, port=port_num)
            else:
                  self.startup_nodes = [{"host": host_ip, "port": port_num}]
                  self.key_id_map = redis.StrictRedis(host=host_ip, port=port_num)

            
            # If the dataset length is evenly divisible by # of replicas, then there
            # is no need to drop any data, since the dataset will be split equally.
            if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore
                  # Split to nearest available length that is evenly divisible.
                  # This is to ensure each rank receives the same amount of data when
                  # using this Sampler.
                  self.num_samples = math.ceil(
                        # `type:ignore` is required because Dataset cannot provide a default __len__
                        # see NOTE in pytorch/torch/utils/data/sampler.py
                        (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore
                  )
            else:
                  self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore
            self.total_size = self.num_samples * self.num_replicas
            self.shuffle = shuffle
            self.PADS = PADS
            self.seed = seed

            #initializing parameter to decide between aggressive and relaxed sampling.
            self.curr_val_score = 100
            self.rep_factor = rep_factor

      def __iter__(self) -> Iterator[T_co]:
            if self.PADS:
                  # deterministically shuffle based on epoch and seed
                  g = torch.Generator()
                  g.manual_seed(self.seed + self.epoch)
                  #self.indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore
                  self.idxes = torch.multinomial(
                        input=self.weights.add(self.ls_param),
                        num_samples=len(self.dataset),
                        replacement=self.replacement,
                        generator= g)
                        
                  self.indices = self.idxes.tolist()
                  if self.epoch == 0:
                        random.shuffle(self.indices)
                        pass
                  if self.epoch > 0 and self.curr_val_score > 0:
                        self.indices = self.pads_sort(self.indices)
            else:
                  self.idxes = torch.multinomial(self.weights.add(self.ls_param), len(self.dataset), self.replacement)
                  self.indices = self.idxes.tolist()
            
            if not self.drop_last:
                  # add extra samples to make it evenly divisible
                  padding_size = self.total_size - len(self.indices)
                  if padding_size <= len(self.indices):
                        self.indices += self.indices[:padding_size]
                  else:
                        self.indices += (self.indices * math.ceil(padding_size / len(self.indices)))[:padding_size]
            else:
                  # remove tail of data to make it evenly divisible.
                  self.indices = self.indices[:self.total_size]
            assert len(self.indices) == self.total_size

            # take repetitive samples based on loss/val_accuracy.
            self.indices = self.score_based_rep()

            #increase hit rates through PADS
            cache_hit_list, cache_miss_list, num_miss_samps = self.prepare_hits(self.rep_factor)
            
            #create the indices list for processing in the next epoch
            self.indices = cache_hit_list + cache_miss_list[:num_miss_samps]

            # print(f'hit_list_len: {len(cache_hit_list)}')
            # print(f'miss_list_len: {len(cache_miss_list[:num_miss_samps])}')
            # print(len(self.indices))

            #sanity check
            self.indices = self.indices[:self.num_samples]

            assert len(self.indices) == self.num_samples
            self.indices_for_process = self.indices
            return iter(self.indices)

      def __len__(self) -> int:
            return self.num_samples

      def set_epoch(self, epoch: int) -> None:
            r"""
            Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
            use a different random ordering for each epoch. Otherwise, the next iteration of this
            sampler will yield the same ordering.

            Args:
                  epoch (int): Epoch number.
            """
            self.epoch = epoch

      def on_epoch_end(self, metrics):
            if not hasattr(self, 'prev_loss'):
                  self.prev_loss = metrics
                  self.ls_param = self.prev_loss * self.ls_init_fac
            else:
                  cur_loss = metrics
                  # assume normal learning curve
                  ls_fac = np.exp((cur_loss - self.prev_loss) / self.prev_loss)
                  self.ls_param = self.ls_param * ls_fac
                  self.prev_loss = cur_loss
            self.item_curr_pos = 0

      def set_importance_per_batch(self, wts):
            end = min(self.item_curr_pos + self.batch_size, self.num_samples)
            # self.idxes is the entire set of indices of the dataset that will be processed in the epoch. For example : [3,2,1,5,4,7,6,9,8,12,11,10]
            # Getting the portion of those indices that was processed in the current batch.
            # if batch size = 3 and second batch is getting processed, then indices [3,4,5] will be processed. self.i: end = 3:6
            # Finding the actual indices in the dataset between 3:6 which is [5,4,7]
            new_wts = torch.argsort(wts).cpu()
            items = len(wts)
            self.updated_idx_list = self.idxes[self.item_curr_pos+new_wts]
            self.weights[self.updated_idx_list] = self.batch_wts[0:items]
            self.item_curr_pos += self.batch_size
      
      def pass_batch_important_scores(self, raw_score):
            self.set_importance_per_batch(raw_score)
      def get_weights(self):
            return self.weights.add(self.ls_param)
      def get_idxes(self):
            return self.idxes
      def get_indices_for_process(self):
            return self.indices_for_process
      def get_sorted_index_list(self):
            return self.updated_idx_list

      def pass_curr_val_change(self,val_change_avg):
            self.curr_val_score = val_change_avg


      def pads_sort(self,l):
            # l -> list to be sorted
            n = len(l)

            # d is a hashmap
            d = {}
            #d = defaultdict(lambda: 0)
            for i in range(n):
                  #d[l[i]] += 1
                  d[l[i]] = 1 + d.get(l[i],0)

            # Sorting the list 'l' where key
            # is the function based on which
            # the array is sorted
            # While sorting we want to give
            # first priority to Frequency
            # Then to value of item
            l.sort(key=lambda x: (-d[x], x))

            return l

      def score_based_rep(self):
            if self.epoch > 0 and self.curr_val_score > 0:
                  cvs = self.curr_val_score
                  ce = self.epoch
                #   print(f'epoch: {ce}, curr_val_score: {cvs}, so doing aggressive sampling.')
                  self.indices = self.indices[:self.num_samples]
                  random.shuffle(self.indices)
            else:
                  self.indices = self.indices[self.rank:self.total_size:self.num_replicas]

            return self.indices
      
      # def fetch_from_cache(self, key):
      #   try:
      #       return self.key_id_map.get(key)
      #   except:
      #        return None
        
      # def exits_in_cache(self, key):
      #   data = self.fetch_from_cache(key)
      #   if data is None:
      #       return False
      #   else:
      #       return True


      def prepare_hits(self,r):
            hit_list = []
            miss_list = []
            for ind in self.indices:
                  if self.key_id_map.exists(ind):
                        hit_list.append(ind)
                  else:
                        miss_list.append(ind)

            # print(f'hit_list_len: {len(hit_list)}')
            # print(f'miss_list_len: {len(miss_list)}')

            # if rep_factor is a multiple of 0.5
            if r % 1 != 0:
                  r = r - 0.5
                  r = int(r)
                  hit_samps = len(hit_list) * r + len(hit_list)//2
                  miss_samps = len(self.indices) - hit_samps

                #   print(f'hit_samps: {hit_samps}')
                #   print(f'miss_samps: {miss_samps}')

                  art_hit_list = hit_list*r + hit_list[:len(hit_list)//2]
                  art_miss_list = miss_list

                  random.shuffle(art_hit_list)
                  random.shuffle(art_miss_list)
            else:
                  r = int(r)
                  hit_samps = len(hit_list) * r
                  miss_samps = len(self.indices) - hit_samps

                #   print(f'hit_samps: {hit_samps}')
                #   print(f'miss_samps: {miss_samps}')

                  art_hit_list = hit_list*r 
                  art_miss_list = miss_list

                  random.shuffle(art_hit_list)
                  random.shuffle(art_miss_list)

            return art_hit_list,art_miss_list,miss_samps

class ShadeBatchSampler(BatchSampler):
        from typing import Any, Dict, Iterable, Iterator, List, Optional, Union, Sized
      
        def __init__(self, sampler: Union[Sampler[int], Iterable[int]], batch_size: int, drop_last: bool):
            super().__init__(sampler, batch_size, drop_last)
            self.epoch = -1
      
        def create_batch_id(self, indicies):
            # Convert integers to strings and concatenate them
            id_string = ''.join(str(x) for x in indicies)        
        # Hash the concatenated string to generate a unique ID
            unique_id = hashlib.sha1(id_string.encode()).hexdigest() 
            return unique_id
        def get_sorted_index_list(self):
            return self.sampler.get_sorted_index_list()
        
        def on_epoch_end(self, metrics):
              return self.sampler.on_epoch_end(metrics)
        
        def pass_batch_important_scores(self,raw_score):
              self.sampler.pass_batch_important_scores(raw_score)

      
        def __iter__(self) -> Iterator[Tuple[int, List[int]]]:
            self.epoch +=1
            self.sampler.set_epoch(self.epoch)
            if self.drop_last:
                  sampler_iter = iter(self.sampler)
                  while True:
                        try:
                              # Collect indices to form a batch
                              batch = [next(sampler_iter) for _ in range(self.batch_size)]
                              yield (self.create_batch_id(batch), batch)  # Yield the batch ID and the indices
                        except StopIteration:
                              break
            else:
                  batch = []
                  for idx in self.sampler:
                        batch.append(idx)
                        if len(batch) == self.batch_size:
                              yield (self.create_batch_id(batch), batch)
                              batch = []
                  if len(batch) > 0:  # Handle the final batch, if any
                         yield (self.create_batch_id(batch), batch)
                         
        def __len__(self) -> int:
              return super().__len__()  # Use the length calculation from the parent class
        

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from mlworklaods.dataloaders.shade.shadedataset_s3 import ShadeDatasetS3
    from heapdict import heapdict
    from torchvision import transforms

    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
    def create_batch_id(indicies):
            id_string = ''.join(str(x) for x in indicies)
            # Hash the concatenated string to generate a unique ID
            unique_id = hashlib.sha1(id_string.encode()).hexdigest() 
            return unique_id

    cache_address = '127.0.0.1:6379'
    ghost_cachehe = heapdict()
    pq = heapdict()
    batch_size =25000 
    dataset = ShadeDatasetS3(
    data_dir='data/cifar-10/train',
    transform=transform,
    cache_address=cache_address,
    PQ=pq,
    ghost_cache=ghost_cachehe,
    key_counter=0,
    wss=0.1
    )
        
    base_sampler = ShadeSamplerS3(
            dataset=dataset,
            num_replicas=1,
            shuffle=False,
            seed=42,
            rank=0,
            batch_size=batch_size,
            drop_last=False,
            replacement=True,
            host_ip='127.0.0.1',
            port_num='6379',
            rep_factor=3,
            init_fac=1,
            ls_init_fac=0.01)
    
    batch_sampler = ShadeBatchSampler(sampler=base_sampler, batch_size=batch_size, drop_last=False)

    loader = DataLoader(dataset=dataset, batch_size=None, num_workers=0, pin_memory=True, sampler=batch_sampler)
    for _ in range(0,32):
              print()
              for batch_idx, (batch_id, img_indices) in enumerate(loader):
                # Your training code here
                # 'data' contains the input images
                # 'target' contains the corresponding labels
                print(f'{_+1} | {batch_idx} | {batch_id} | ')
                pass

