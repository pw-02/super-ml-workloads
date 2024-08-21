import functools
import torch
import s3utils
from typing import List, Tuple, Dict
from PIL import Image
import io

class S3MappedImageDataset(torch.utils.data.Dataset):
    def __init__(self,data_dir:str, transform):
        # if dataset_kind == 'image':
        self.samples: Dict[str, List[str]] = s3utils.load_paired_s3_object_keys(data_dir, True, True)
        self.bucket_name = s3utils.S3Url(data_dir).bucket
        self.transform = transform

    @functools.cached_property
    def _classed_items(self) -> List[Tuple[str, int]]:
        return [(blob, class_index)
            for class_index, blob_class in enumerate(self.samples)
            for blob in self.samples[blob_class]]
    
    def __len__(self):
        return sum(len(class_items) for class_items in self.samples.values())
    
    def __getitem__(self, idx):
        file_path, sample_label = self._classed_items[idx]
        sample_input = s3utils.get_s3_object(self.bucket_name, file_path)

        if s3utils.is_image_file(file_path):
            sample_input = Image.open(io.BytesIO(sample_input))
            if sample_input.mode == "L":
                sample_input = sample_input.convert("RGB")
        
        if self.transform is not None:
            sample_input = self.transform(sample_input)
        return sample_input, sample_label
    