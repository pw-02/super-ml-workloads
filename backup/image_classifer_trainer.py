import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import s3fs
from PIL import Image
from io import BytesIO
import time
from typing import Union, List
from pytorch_lightning import Callback
import csv
import psutil
import subprocess

class ProfilingData:
    def __init__(self):
        self.data_loading_time = 0
        self.transformation_time = 0

profiling_data = ProfilingData()


class S3CustomDataset(Dataset):
    def __init__(self, s3_bucket, s3_prefix, transform=None):
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.transform = transform
        
        # Initialize S3 filesystem
        self.s3_fs = s3fs.S3FileSystem()
        
        # List all files in the S3 bucket under the specified prefix
        self.image_files = [
            obj.key for obj in self.s3_fs.ls(s3_prefix)
            if obj.key.endswith('.jpg') or obj.key.endswith('.jpeg')
        ]
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Measure data fetching time
        start_time = time.time()
        img_key = self.image_files[idx]
        with self.s3_fs.open(img_key, 'rb') as f:
            image = Image.open(BytesIO(f.read())).convert('RGB')
        profiling_data.data_loading_time += time.time() - start_time
        
        # Measure transformation time
        start_time = time.time()
        if self.transform:
            image = self.transform(image)
        profiling_data.transformation_time += time.time() - start_time
        
        # Assuming labels are encoded in the filename, adjust as needed
        label = int(img_key.split('/')[-1].split('_')[0])
        
        return image, label

class ResNet50(pl.LightningModule):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.model = torchvision.models.resnet50(pretrained=False, num_classes=1000)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = self.criterion(output, target)
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = self.criterion(output, target)
        acc = (output.argmax(dim=1) == target).float().mean()
        return {'val_loss': loss, 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        self.log('val_loss', avg_loss, prog_bar=True)
        self.log('val_acc', avg_acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

class CustomTrainer(pl.Trainer):
    def train_loop(
        self,
        model: pl.LightningModule,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        limit_batches: Union[int, float] = float("inf"),
        *args,
        **kwargs
    ):
        # Call the default on_train_epoch_start hook if needed
        self.on_train_epoch_start()

        total_loss_for_epoch = 0.
        end = time.perf_counter()
        for batch_idx, batch in enumerate(train_loader):
            data_time = time.perf_counter() - end

            # End epoch if stopping training completely or max batches for this epoch reached
            if self.should_stop or batch_idx >= limit_batches:
                break

            # Move data to GPU
            data, target = batch
            data, target = data.cuda(), target.cuda()

            # Forward pass
            compute_start = time.perf_counter()
            output = model(data)
            loss = model.training_step((data, target), batch_idx)
            total_loss_for_epoch += loss.item()

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Time spent on GPU processing
            batch_time = time.perf_counter() - compute_start
            profiling_data.gpu_processing_time += batch_time

            # Print profiling information if needed
            print(f"Batch processing time: {batch_time:.4f}s")

            end = time.perf_counter()

        return total_loss_for_epoch / (batch_idx + 1)

    def validation_loop(
        self,
        model: pl.LightningModule,
        dataloader: DataLoader,
        *args,
        **kwargs
    ):
        model.eval()
        total_val_loss = 0
        total_val_acc = 0
        with torch.no_grad():
            for batch in dataloader:
                data, target = batch
                data, target = data.cuda(), target.cuda()

                # Validation forward pass
                output = model(data)
                loss = model.validation_step((data, target), 0)
                total_val_loss += loss['val_loss'].item()
                total_val_acc += loss['val_acc'].item()

        # Average metrics
        avg_val_loss = total_val_loss / len(dataloader)
        avg_val_acc = total_val_acc / len(dataloader)

        model.log('val_loss', avg_val_loss, prog_bar=True)
        model.log('val_acc', avg_val_acc, prog_bar=True)

def main():
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # S3 bucket and prefix for your data
    s3_bucket = 'your-s3-bucket-name'
    s3_train_prefix = 'path_to_train_data'
    s3_val_prefix = 'path_to_val_data'

    # Load the custom dataset from S3
    train_dataset = S3CustomDataset(s3_bucket=s3_bucket, s3_prefix=s3_train_prefix, transform=transform)
    val_dataset = S3CustomDataset(s3_bucket=s3_bucket, s3_prefix=s3_val_prefix, transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, num_workers=4)

    # Initialize the model
    model = ResNet50()

    # Initialize custom PyTorch Lightning Trainer
    trainer = CustomTrainer(
        max_epochs=10,
        callbacks=[ProfilingCallback()],
        gpus=1  # Set to the number of GPUs you are using
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()
