from omegaconf import DictConfig
import hydra
import os
from mlworklaods.image_classification.image_classifer_trainer import ImageClassificationTrainer
import torch
from mlworklaods.image_classification.imager_classifer_model import ImageClassifierModel
from lightning.fabric.loggers import CSVLogger
# from ray import tune
# from ray.air import session
from lightning.pytorch.core.saving import save_hparams_to_yaml
from mlworklaods.image_classification.data import CIFAR10DataModule, ImageNetDataModule
from mlworklaods.llm.data import OpenWebTextDataModule
from mlworklaods.utils import get_default_supported_precision
from mlworklaods.args import *
from datetime import datetime
from mlworklaods.llm.pretrainer import LLMPretrainer

@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(hydra_config: DictConfig):
    
    train_args, data_args, dataloader_args = prepare_args(hydra_config, f"{hydra_config.dataloader.name}_{hydra_config.exp_id}/{hydra_config.job_id}" )
    logger = CSVLogger(root_dir=train_args.log_dir, name="", flush_logs_every_n_steps=train_args.log_interval)
    
    if isinstance(train_args, LLMTrainArgs):
        if 'openwebtext' in data_args.dataset_name:
            data_module = OpenWebTextDataModule()

        trainer = LLMPretrainer(
            job_id=hydra_config.job_id,
            train= train_args,
            accelerator=train_args.accelerator,
            precision=get_default_supported_precision(True),
            devices=train_args.devices, 
            loggers=[logger],
            model_name=train_args.model_name,
            seed=train_args.seed,
            max_iters=train_args.max_steps
        )

        model, optmizer = trainer.initialize_model_and_optimizer()  
        train_loader, val_loader = data_module.make_dataloaders(train_args, data_args, dataloader_args, model.max_seq_length, trainer.fabric.world_size)
        os.makedirs(logger.log_dir, exist_ok=True)
        save_hparams_to_yaml(os.path.join(logger.log_dir, "hparms.yaml"), hydra_config)
        trainer.fit(model, optmizer, train_loader, val_loader)
       
    elif isinstance(train_args, ImgClassifierTrainArgs):
        if 'cifar10' in data_args.dataset_name:
            data_module = CIFAR10DataModule()
        elif 'imagenet' in data_args.dataset_name:
            data_module = ImageNetDataModule()

        model = ImageClassifierModel(train_args.model_name, train_args.learning_rate, num_classes=data_module.num_classes)
        
        trainer = ImageClassificationTrainer(
            job_id=hydra_config.job_id,
            accelerator=train_args.accelerator,
            precision=get_default_supported_precision(True),
            devices=train_args.devices, 
            callbacks=None,
            max_steps=train_args.max_steps,
            max_epochs=train_args.max_epochs,
            limit_train_batches=train_args.limit_train_batches, 
            limit_val_batches=train_args.limit_val_batches, 
            loggers=[logger],
            use_distributed_sampler=True,
            dataloader_args =dataloader_args
            )

        train_loader, val_loader = data_module.make_dataloaders(train_args, data_args, dataloader_args, trainer.fabric.world_size)

        avg_loss, avg_acc = trainer.fit(model, train_loader, val_loader, train_args.seed)
        os.makedirs(logger.log_dir, exist_ok=True)
        save_hparams_to_yaml(os.path.join(logger.log_dir, "hparms.yaml"), hydra_config)
        print(f"Training completed with loss: {avg_loss}, accuracy: {avg_acc}")
        
if __name__ == "__main__":
    main()
