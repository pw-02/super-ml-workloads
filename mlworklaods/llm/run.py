from omegaconf import DictConfig
import hydra
from lightning.pytorch.core.saving import save_hparams_to_yaml
from datetime import datetime
from args import *
from mlworklaods.llm.pretrain import setup
from mlworklaods.llm.data.openwebtext import OpenWebText
# Helper function to prepare arguments for a job
def prepare_args(config: DictConfig,expid):
    log_dir = f"{config.log_dir}/{config.dataset.name}/{config.training.model_name}/{expid}"

    train_args = TrainArgs(
        model_name=config.training.model_name,
        dataloader_kind=config.dataloader.kind,
        save_interval=config.training.save_interval,
        max_tokens=config.training.max_tokens,
        log_interval=config.log_freq,
        global_batch_size=config.training.global_batch_size,
        micro_batch_size=config.training.micro_batch_size,
        lr_warmup_steps=config.training.lr_warmup_steps,
        lr_warmup_fraction=config.training.lr_warmup_fraction,
        epochs=config.training.epochs,
        max_steps=config.training.max_steps,
        max_seq_length=config.training.max_seq_length,
        tie_embeddings=config.training.tie_embeddings,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        beta1=config.training.beta1,
        beta2=config.training.beta2,
        max_norm=config.training.max_norm,
        min_lr=config.training.min_lr,
        devices=config.num_devices_per_job,
        seed=config.seed)
    
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


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(hydra_config: DictConfig):

    train_args, data_args, dataloader_args = prepare_args(hydra_config, datetime.now().strftime("train_single_model_%Y-%m-%d_%H-%M-%S"))

    data = OpenWebText('data/openwebtext')

    setup(
        model_name=train_args.model_name,
        model_config=None,
        precision=None,
        resume=False,
        data=None,
        train=train_args,
        # eval=None,
        devices=train_args.devices,
        logger_name='csv',
        seed=train_args.seed)

if __name__ == "__main__":
    main()
