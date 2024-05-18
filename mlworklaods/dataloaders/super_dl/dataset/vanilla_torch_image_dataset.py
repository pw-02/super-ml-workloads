import functools
import torch
import mlworklaods.s3utils as s3utils
from mlworklaods.s3utils import S3Url
from typing import List, Tuple, Dict
from PIL import Image
import io
from pathlib import Path
import json
import os

class VanillaTorchImageDataset(torch.utils.data.Dataset):
    def __init__(self,data_dir:str, transform):
        self.is_s3: bool = data_dir.startswith("s3://")

        if self.is_s3:
            self.samples: Dict[str, List[str]] = s3utils.load_paired_s3_object_keys(data_dir, True, True)
            self.bucket_name = S3Url(data_dir).bucket
        else:
            self.samples: Dict[str, List[str]] = self.load_local_sample_idxs(data_dir)

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
        if self.is_s3:
            sample_content = s3utils.get_s3_object(self.bucket_name, file_path)
        else:
            sample_content = self.get_local_sample(file_path)

        if self.is_image_file(file_path):
            sample_input = Image.open(io.BytesIO(sample_content))
            if sample_input.mode == "L":
                sample_input = sample_input.convert("RGB")
        
        if self.transform is not None:
            sample_input = self.transform(sample_input)
        return sample_input, sample_label
    

    def is_image_file(self, path: str):
        return any(path.endswith(extension) for extension in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP'])
    

    def get_local_sample(self, file_path):
        img = Image.open(file_path)
        return img

    def load_local_sample_idxs(self, data_dir) -> Dict[str, List[str]]:
            data_dir = str(Path(data_dir))
            classed_samples: Dict[str, List[str]] = {}
            index_file = Path(data_dir) / 'index.json'

            if index_file.exists():
                with open(index_file.absolute()) as f:
                    classed_samples = json.load(f)
            else:
                for dirpath, dirnames, filenames in os.walk(data_dir):
                    for filename in filter(self.is_image_file, filenames):
                        img_class = os.path.basename(dirpath.removesuffix('/'))
                        img_path = os.path.join(dirpath, filename)
                        classed_samples.setdefault(img_class, []).append(img_path)
                json_object = json.dumps(classed_samples, indent=4)
                with open(index_file, "w") as outfile:
                    outfile.write(json_object)
            return classed_samples