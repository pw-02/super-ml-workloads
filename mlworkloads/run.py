from omegaconf import DictConfig, OmegaConf
import hydra
from lightning.fabric.loggers import CSVLogger
from image_classifer import train_image_classifer
import os
from lightning.pytorch.core.saving import save_hparams_to_yaml
from shade_train_image_classifier import train_image_classifer as shade_train_image_classifer
from datetime import datetime

@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(config: DictConfig):

    print(OmegaConf.to_yaml(config, resolve=True))

    log_dir = f"{config.log_dir}/{config.workload.name}/{config.dataloader.name}/{config.exp_id}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/{config.job_id}".lower()
    log_dir = os.path.normpath(log_dir)  # Normalize path for Windows
    
    train_logger = CSVLogger(root_dir=log_dir, name="train", prefix='', flush_logs_every_n_steps=config.log_interval)
    val_logger = CSVLogger(root_dir=log_dir, name="val", prefix='', flush_logs_every_n_steps=config.log_interval)

    if config.workload.name == 'cifar10_resnet18' or config.workload.name  == 'imagenet_resnet50':
        if config.dataloader.name == 'shade':
            shade_train_image_classifer(config, train_logger,val_logger )
        else:
            train_image_classifer(config, train_logger,val_logger )

    elif config.workload.name  == 'Pythia-14M/OpenWebText':
        pass
    else:
        raise ValueError(f"Invalid workload: {config.workload}")
    save_hparams_to_yaml(os.path.join(log_dir, "hparms.yaml"), config)


if __name__ == "__main__":
    main()
