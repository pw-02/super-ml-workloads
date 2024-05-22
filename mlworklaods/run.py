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

def train_model(hydra_config):
    # if config:
    #     train_args, data_args, dataloader_args = prepare_args(hydra_config, f"hpo_{session.get_experiment_name()}/{session.get_trial_name()}")
    #else:
        # train_args, data_args, dataloader_args = prepare_args(hydra_config, datetime.now().strftime("train_single_model_%Y-%m-%d_%H-%M-%S"))
    #    train_args, data_args, dataloader_args = prepare_args(hydra_config, f"{hydra_config.exp_id}/{hydra_config.job_id}" )
    
    train_args, data_args, dataloader_args = prepare_args(hydra_config, f"{hydra_config.exp_id}" )

    logger = CSVLogger(root_dir=train_args.log_dir, name="", flush_logs_every_n_steps=train_args.log_interval)
    
    if isinstance(train_args, LLMTrainArgs):
        if 'openwebtext' in data_args.dataset_name:
            data_module = OpenWebTextDataModule()

        trainer = LLMPretrainer(
            train= train_args,
            accelerator=train_args.accelerator,
            precision=get_default_supported_precision(True),
            devices=train_args.devices, 
            loggers=[logger],
            model_name=train_args.model_name,
            seed=train_args.seed
        )

        model, optmizer = trainer.initialize_model_and_optimizer()  
        train_loader, val_loader = data_module.make_dataloaders(train_args, data_args, dataloader_args, model.max_seq_length)
        os.makedirs(logger.log_dir, exist_ok=True)
        save_hparams_to_yaml(os.path.join(logger.log_dir, "hparms.yaml"), hydra_config)
        avg_loss, avg_acc = trainer.fit(model, optmizer, train_loader, val_loader)
       
    elif isinstance(train_args, ImgClassifierTrainArgs):
        if 'cifar10' in data_args.dataset_name:
            data_module = CIFAR10DataModule()
        elif 'imagenet' in data_args.dataset_name:
            data_module = ImageNetDataModule()

        model = ImageClassifierModel(train_args.model_name, train_args.learning_rate, num_classes=data_module.num_classes)
        
        trainer = ImageClassificationTrainer(
            accelerator=train_args.accelerator,
            precision=get_default_supported_precision(True),
            devices=train_args.devices, 
            callbacks=None,
            max_epochs=train_args.max_epochs,
            limit_train_batches=train_args.limit_train_batches, 
            limit_val_batches=train_args.limit_val_batches, 
            loggers=[logger],
            use_distributed_sampler=True
        )

        train_loader, val_loader = data_module.make_dataloaders(train_args, data_args, dataloader_args, trainer.fabric.world_size)

        avg_loss, avg_acc = trainer.fit(model, train_loader, val_loader, train_args.seed)
        os.makedirs(logger.log_dir, exist_ok=True)
        save_hparams_to_yaml(os.path.join(logger.log_dir, "hparms.yaml"), hydra_config)
        print(f"Training completed with loss: {avg_loss}, accuracy: {avg_acc}")
        


@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(hydra_config: DictConfig):
    
    train_model(hydra_config)

    # if hydra_config.num_jobs > 1:
    #     os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"
    #     os.environ["RAY_DEDUP_LOGS"] = "0"

    #     search_space = {
    #         "lr": tune.loguniform(1e-5, 1e-2),
    #         # "training.batch_size": tune.choice([16, 32, 64, 128]),
    #     }

    #     result = tune.run(
    #         tune.with_parameters(train_model, hydra_config=hydra_config),
    #         resources_per_trial={"gpu": 0.25},
    #         metric="loss",
    #         mode="min",
    #         max_concurrent_trials=hydra_config.num_jobs,
    #         config=search_space,
    #         num_samples=hydra_config.num_jobs,
    #         checkpoint_freq=0, 
    #         verbose=0,
    #     )
    #     best_trial = result.get_best_trial("loss", "min", "last")
    #     print(f"Best trial config: {best_trial.config}")
    #     print(f"Best trial final loss: {best_trial.last_result['loss']}")
    #     print(f"Best trial final accuracy: {best_trial.last_result['accuracy']}")
    # else:

    #     train_model(config=None, hydra_config= hydra_config)

if __name__ == "__main__":
    main()
