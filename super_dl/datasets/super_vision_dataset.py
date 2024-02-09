from PIL import Image
from torch.utils.data import Dataset
from typing import Optional, List, Tuple, Callable, Dict
import io
import torch
from super_dl.datasets.super_base_dataset import BaseSUPERDataset, timer_decorator

class SUPERVisionDataset(BaseSUPERDataset):
    def __init__(self, job_id, data_dir: str,  transform=None,target_transform=None, cache_address=None):
        super().__init__(job_id, data_dir, data_dir.startswith("s3://"), cache_address)
        self.transform = transform
        self.target_transform = target_transform
        # self.img_extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']

    def __getitem__(self, next_batch):
        indices, batch_id,cache_status = next_batch
        cached_data = None

        if self.cache_client is not None:
            cached_data, fetch_time = self.fetch_from_cache(batch_id, cache_status)
            
        if cached_data:
            # Convert JSON batch to torch format
            torch_imgs, torch_labels, transform_time = self.deserialize_torch_batch(cached_data)
            # print('data returned') 
            return torch_imgs, torch_labels, batch_id, True, fetch_time, transform_time
  
        images, labels, fetch_time = self.fetch_batch_data(indices)
        images, labels, transform_time = self.apply_transformations(images, labels)
        return torch.stack(images), torch.tensor(labels), batch_id, False, fetch_time, transform_time
    

    @timer_decorator
    def fetch_batch_data(self, indices):
        images = []
        labels = []
        for idx in indices:
            file_path, label = self._classed_items[idx]
            if self.use_s3:
                # Download file into memory
                content = self.s3_helper.load_s3_file_into_memory(self.bucket_name, file_path)
                img = Image.open(io.BytesIO(content))
            else:
                img = Image.open(file_path)
            if img.mode == "L":
                img = img.convert("RGB")
            images.append(img)
            labels.append(label)
        return images, labels
    
    @timer_decorator
    def apply_transformations(self, images, labels):
        if self.transform is not None:
            for i in range(len(images)):
                images[i] = self.transform(images[i])

        if self.target_transform is not None:
            for i in range(len(labels)):
                labels[i] = self.target_transform(labels[i])
        return images, labels