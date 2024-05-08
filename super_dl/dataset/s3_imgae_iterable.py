import functools
import super_dl.s3utils as s3utils
from  super_dl.s3utils import S3Url
from typing import List, Tuple, Dict
from PIL import Image
import io
from torch.utils.data import SequentialSampler, IterableDataset, RandomSampler, DataLoader
import torchvision

class S3IterableDataset(IterableDataset):
    def __init__(self,data_dir:str, transform, shuffle = False):
        self.epoch = 0
        self.shuffle_urls = shuffle
        # if dataset_kind == 'image':
        self.samples: Dict[str, List[str]] = s3utils.load_paired_s3_object_keys(data_dir, True, True)
        self.bucket_name = S3Url(data_dir).bucket
        self.transform = transform
 
    @functools.cached_property
    def _classed_items(self) -> List[Tuple[str, int]]:
        return [(blob, class_index)
            for class_index, blob_class in enumerate(self.samples)
            for blob in self.samples[blob_class]]
    
    def __len__(self):
        return sum(len(class_items) for class_items in self.samples.values())
    
    def __iter__(self):

        if self.shuffle_urls:
            sampler = RandomSampler(self)
        else:
            sampler = SequentialSampler(self)
        
        for idx in sampler:
            file_path, sample_label = self._classed_items[idx]
            sample_input = s3utils.get_s3_object(self.bucket_name, file_path)
            if s3utils.is_image_file(file_path):
                sample_input = Image.open(io.BytesIO(sample_input))
                if sample_input.mode == "L":
                    sample_input = sample_input.convert("RGB")
            
            if self.transform is not None:
                sample_input = self.transform(sample_input)
                
                yield sample_input, sample_label

    
    def set_epoch(self, epoch):
        self.epoch = epoch

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
train_data_dir = 's3://sdl-cifar10/test/'
dataset = S3IterableDataset(data_dir=train_data_dir,
                             transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
                             shuffle=True)

data_loader = DataLoader(dataset, batch_size=5)
# Get the size of the tensor using pympler
for input, target in data_loader:
    batch_size_mb,size_in_kb  = get_batch_size_mb(input)
    print(f"Batch size: {batch_size_mb:.2f} MB, {size_in_kb:.2f} KB")
    print(input.shape, target.shape)
