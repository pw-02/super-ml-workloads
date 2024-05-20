import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18
from torchvision import transforms
import pytorch_lightning as pl
import torchvision
from torchmetrics.functional.classification.accuracy import accuracy
import torch.nn as nn
from mlworklaods.image_classification.args import *
from mlworklaods.dataloaders.torch_lru.batch_sampler_with_id import BatchSamplerWithID
from mlworklaods.dataloaders.torch_lru.torch_lru_dataset import TorchLRUDataset
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler  
from mlworklaods.dataloaders.super_dl.dataset.super_dataset import SUPERDataset
from mlworklaods.image_classification.utils import AverageMeter


class ImageClassifierModel(pl.LightningModule):

    def __init__(self, model_name, learning_rate, num_classes, optimizer='Adam'):
        # super(ImageClassifierModel, self).__init__()
        super().__init__()
        self.model:nn.Module = torchvision.models.get_model(model_name,pretrained=False)

        # Modify the final fully connected layer to match the number of classes in CIFAR-10
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.losses =  AverageMeter("Loss", ":6.2f")
        #losses = AverageMeter("Loss", ":6.2f")
        self.top1 = AverageMeter("Acc1", ":6.2f")
        # top5 = AverageMeter("Acc5", ":6.2f")


    def forward(self, x):
        return self.model.forward(x)

    def training_step(self, batch, batch_idx: int):
        x, y = batch
        logits = self.model(x)

        loss = self.loss_fn(logits, y)
        top1 = accuracy(logits.argmax(-1), y, num_classes=10, task="multiclass", top_k=1)
        
        self.losses.update(loss.item())
        self.top1.update(top1.item())
        # calculating the cross entropy loss on the result
        return {"loss": loss, "top1": top1}
    
    # def on_train_epoch_end(self):
    #     self.log("ptl/val_loss", self.losses.avg)
    #     self.log("ptl/val_accuracy", self.top1.avg)

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        # calculating the cross entropy loss on the result
        self.total_classified += y.shape[0]
        self.correctly_classified += (y_hat.argmax(1) == y).sum().item()
        # Calculating total and correctly classified images to determine the accuracy later
        self.log('val_loss', loss) 
        # logging the loss with "val_" prefix
        return loss
    


    def validation_epoch_end(self, results):
        accuracy = self.correctly_classified / self.total_classified
        self.log('val_accuracy', accuracy)
        # logging accuracy
        self.total_classified = 0
        self.correctly_classified = 0
        return accuracy

    def configure_optimizers(self):
        # Choose an optimizer and set up a learning rate according to hyperparameters
        if self.optimizer == 'Adam':
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer == 'SGD':
            return torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        else:
            raise NotImplementedError
    
    def make_dataloaders(self, train_args, data_args,dataloader_args, world_size):
        if 'super' in train_args.dataloader_kind:
            return self.make_super_dataloaders(train_args, data_args,dataloader_args, world_size)
        elif 'torch_lru' in train_args.dataloader_kind:
           return self.make_lru_torch_datalaoders(train_args, data_args,dataloader_args, world_size)
        else:
            raise Exception(f"Unknown dataloader_kind {train_args.dataloader_kind}")
    
    
    def make_lru_torch_datalaoders(self,train_args: TrainArgs, data_args: DataArgs, lru_torch_args:LRUTorchArgs, world_size:int):
        train_dataloader = None
        val_dataloader = None
        if train_args.run_training:
            train_dataset =  TorchLRUDataset(
                data_dir = data_args.train_data_dir,
                transform=self.transform(),
                cache_address=lru_torch_args.cache_address,
                cache_granularity=lru_torch_args.cache_granularity)
            
            train_base_sampler = RandomSampler(data_source=train_dataset) if lru_torch_args.shuffle else SequentialSampler(data_source=train_dataset)
            train_batch_sampler = BatchSamplerWithID(sampler=train_base_sampler, batch_size=train_args.get_batch_size(world_size), drop_last=False)  
            train_dataloader = DataLoader(dataset=train_dataset, sampler=train_batch_sampler, batch_size=None, num_workers=lru_torch_args.num_pytorch_workers)

        if train_args.run_evaluation:
            val_dataset =  TorchLRUDataset(
                data_dir = data_args.val_data_dir,
                transform=self.transform(),
                cache_address=lru_torch_args.cache_address,
                cache_granularity=lru_torch_args.cache_granularity)
            
            val_base_sampler = RandomSampler(data_source=val_dataset) if lru_torch_args.shuffle else SequentialSampler(data_source=val_dataset)
            val_batch_sampler = BatchSamplerWithID(sampler=val_base_sampler, batch_size=train_args.get_batch_size(world_size), drop_last=False)
            
            val_dataloader = DataLoader(dataset=val_dataset, sampler=val_batch_sampler, batch_size=None, num_workers=lru_torch_args.num_pytorch_workers)
        return train_dataloader, val_dataloader


    # Dataloader creation function
    def make_super_dataloaders(self,train_args: TrainArgs, data_args: DataArgs, super_args:SUPERArgs, world_size:int):
        train_dataloader = None
        val_dataloader = None
        if train_args.run_training:  
            dataset = SUPERDataset(
                job_id=train_args.job_id,
                data_dir=data_args.train_data_dir,
                batch_size=train_args.get_batch_size(world_size),
                transform=self.transform(),
                world_size=world_size,
                super_address=super_args.super_address,
                cache_address=super_args.cache_address,
                simulate_delay=super_args.simulate_data_delay)
            
            train_dataloader = DataLoader(dataset=dataset, batch_size=None, num_workers=super_args.num_pytorch_workers)

        if train_args.run_evaluation:
            dataset = SUPERDataset(
                job_id=train_args.job_id,
                data_dir=data_args.val_data_dir,
                batch_size=train_args.get_batch_size(world_size),
                transform=self.transform(),
                world_size=world_size,
                super_address=super_args.super_address,
                cache_address=super_args.cache_address,
                simulate_delay=super_args.simulate_data_delay)
        
        return train_dataloader, val_dataloader
    
    # Transformation function for data augmentation
    def transform(self):
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225],
            )
            return transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])


# if __name__ == "__main__":

#     seeds = [47, 881, 123456789]
#     lrs = [0.1, 0.01]
#     optimizers = ['SGD', 'Adam']

#     for seed in seeds:
#         for optimizer in optimizers:
#             for lr in lrs:
#                 # choosing one set of hyperparaameters from the ones above
            
#                 model = ImageClassifierModel(seed, lr, optimizer)
#                 # initializing the model we will train with the chosen hyperparameters

#                 aim_logger = AimLogger(
#                     experiment='resnet18_classification',
#                     train_metric_prefix='train_',
#                     val_metric_prefix='val_',
#                 )
#                 aim_logger.log_hyperparams({
#                     'lr': lr,
#                     'optimizer': optimizer,
#                     'seed': seed
#                 })
#                 # initializing the aim logger and logging the hyperparameters

#                 trainer = pl.Trainer(
#                     logger=aim_logger,
#                     gpus=1,
#                     max_epochs=5,
#                     progress_bar_refresh_rate=1,
#                     log_every_n_steps=10,
#                     check_val_every_n_epoch=1)
#                 # making the pytorch-lightning trainer

#                 trainer.fit(model)
#                 # training the model