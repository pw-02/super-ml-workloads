from omegaconf import DictConfig
import hydra
from mlworklaods.args import *
import os
from mlworklaods.image_classification.image_classifer_trainer import MyCustomTrainer
import torch
from mlworklaods.image_classification.imager_classifer_model import ImageClassifierModel
from mlworklaods.image_classification.empty_model import EmptyModel
from mlworklaods.image_classification.dataload_only_trainer import DataloadOnlyTrainer
from lightning.fabric.loggers import CSVLogger
from ray import tune
from ray.air import session
from lightning.pytorch.core.saving import save_hparams_to_yaml

# Helper function to prepare arguments for a job
def prepare_args(config: DictConfig,expid):
    log_dir = f"{config.log_dir}/{config.dataset.name}/{config.training.model_name}/{expid}"

    train_args = TrainArgs(
        job_id=os.getpid(),
        model_name=config.training.model_name,
        max_steps=config.training.max_steps,
        max_epochs=config.training.max_epochs,
        limit_train_batches=config.training.limit_train_batches,
        limit_val_batches=config.training.limit_val_batches,
        batch_size=config.training.batch_size,
        grad_accum_steps=config.training.grad_accum_steps,
        run_training=config.run_training,
        run_evaluation=config.run_evaluation,
        devices=config.num_devices_per_job,
        accelerator=config.accelerator,
        seed=config.seed,
        log_dir=log_dir,
        log_freq=config.log_freq,
        dataloader_kind=config.dataloader.kind,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay)
    
    data_args = DataArgs(
        train_data_dir=config.dataset.train_dir,
        val_data_dir=config.dataset.val_dir,
        num_classes=config.dataset.num_classes
    )

    if 'super' in train_args.dataloader_kind:
        super_args = SUPERArgs(
            num_pytorch_workers=config.training.num_pytorch_workers,
            super_address=config.dataloader.super_address,
            cache_address=config.dataloader.cache_address,
            simulate_data_delay=config.dataloader.simulate_data_delay)
        if super_args.simulate_data_delay is not None:
            train_args.dataload_only = True
        return train_args, data_args, super_args

    elif 'shade' in train_args.dataloader_kind:
        shade_args = SHADEArgs(
            num_pytorch_workers=config.training.num_pytorch_workers,
            cache_address=config.dataloader.cache_address,
            working_set_size=config.dataloader.working_set_size,
            replication_factor=config.dataloader.replication_factor)
        return train_args, data_args, shade_args

    elif 'torch_lru' in train_args.dataloader_kind:
        torchlru_args = LRUTorchArgs(
            num_pytorch_workers=config.training.num_pytorch_workers,
            cache_address=config.dataloader.cache_address,
            cache_granularity=config.dataloader.cache_granularity,
            shuffle=config.dataloader.shuffle)

    return train_args, data_args, torchlru_args

def get_default_supported_precision(training: bool) -> str:
    from lightning.fabric.accelerators import MPSAccelerator
    if MPSAccelerator.is_available() or (torch.cuda.is_available() and not torch.cuda.is_bf16_supported()):
        return "16-mixed" if training else "16-true"
    return "bf16-mixed" if training else "bf16-true"

def train_model(config, hydra_config):
    train_args, data_args, dataloader_args = prepare_args(hydra_config, f"hpo_{session.get_experiment_name()}")
    train_args.learning_rate = config["lr"]
    hydra_config.training.learning_rate =train_args.learning_rate 
    
    
    model = ImageClassifierModel(train_args.model_name, train_args.learning_rate, num_classes = data_args.num_classes)
    
    logger = CSVLogger(root_dir=train_args.log_dir, name="", flush_logs_every_n_steps=train_args.log_freq)

    trainer = DataloadOnlyTrainer(
        accelerator=train_args.accelerator,
        precision=get_default_supported_precision(True),
        devices=train_args.devices, 
        limit_train_batches=train_args.limit_train_batches, 
        limit_val_batches=train_args.limit_val_batches, 
        max_epochs=train_args.max_epochs,
        max_steps=train_args.max_steps,
        loggers=[logger],
        grad_accum_steps=train_args.grad_accum_steps,
    )

    train_loader, val_loader = model.make_dataloaders(train_args, data_args, dataloader_args, trainer.fabric.world_size)
    avg_loss, avg_acc = trainer.fit(model, train_loader, val_loader, train_args.seed)
    hparams_file = os.path.join(logger.log_dir, "hparms.yaml")
    save_hparams_to_yaml(hparams_file, hydra_config)
    session.report({"loss": avg_loss, "accuracy": avg_acc})

@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(hydra_config: DictConfig):

    if hydra_config.num_jobs > 1:
        os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"
        os.environ["RAY_DEDUP_LOGS"] = "0"

        search_space = {
            "lr": tune.loguniform(1e-5, 1e-2),
            # "training.batch_size": tune.choice([16, 32, 64, 128]),
        }

        result = tune.run(
            tune.with_parameters(train_model, hydra_config=hydra_config),
            resources_per_trial={"gpu": 0.1},
            metric="loss",
            mode="min",
            max_concurrent_trials=hydra_config.num_jobs,
            config=search_space,
            num_samples=hydra_config.num_jobs,
            checkpoint_freq=0, 
            verbose=1,
        )
        best_trial = result.get_best_trial("loss", "min", "last")
        print(f"Best trial config: {best_trial.config}")
        print(f"Best trial final loss: {best_trial.last_result['loss']}")
        print(f"Best trial final accuracy: {best_trial.last_result['accuracy']}")
    else:
        from datetime import datetime
        train_args, data_args, dataloader_args = prepare_args(hydra_config, datetime.now().strftime("train_single_model_%Y-%m-%d_%H-%M-%S"))
        
        model = ImageClassifierModel(train_args.model_name, train_args.learning_rate, num_classes=data_args.num_classes)
        
        logger = CSVLogger(root_dir=train_args.log_dir, name="", flush_logs_every_n_steps=train_args.log_freq)

        trainer = MyCustomTrainer(
            accelerator=train_args.accelerator,
            precision=get_default_supported_precision(True),
            devices=train_args.devices, 
            limit_train_batches=train_args.limit_train_batches, 
            limit_val_batches=train_args.limit_val_batches, 
            max_epochs=train_args.max_epochs,
            max_steps=train_args.max_steps,
            loggers=[logger],
            grad_accum_steps=train_args.grad_accum_steps,
        )

        train_loader, val_loader = model.make_dataloaders(train_args, data_args, dataloader_args, trainer.fabric.world_size)
        avg_loss, avg_acc = trainer.fit(model, train_loader, val_loader, train_args.seed)
        hparams_file = os.path.join(logger.log_dir, "hparms.yaml")
        save_hparams_to_yaml(hparams_file, hydra_config)
        print(f"Training completed with loss: {avg_loss}, accuracy: {avg_acc}")


if __name__ == "__main__":
    main()
