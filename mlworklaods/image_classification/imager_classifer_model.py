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
from mlworklaods.utils import AverageMeter


class ImageClassifierModel(pl.LightningModule):

    def __init__(self, model_name, learning_rate, num_classes, optimizer='Adam'):
        # super(ImageClassifierModel, self).__init__()
        super().__init__()
        self.model:nn.Module = torchvision.models.get_model(model_name,weights=None)

        # Modify the final fully connected layer to match the number of classes in CIFAR-10
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.losses =  AverageMeter("Loss", ":6.2f")
        self.top1 = AverageMeter("Acc1", ":6.2f")
        self.num_classes = num_classes


    def forward(self, x):
        return self.model.forward(x)

    def training_step(self, batch, batch_idx: int):
        x, y = batch
        logits = self.model(x)

        loss = self.loss_fn(logits, y)
        top1 = accuracy(logits.argmax(-1), y, num_classes=self.num_classes, task="multiclass", top_k=1)
        
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