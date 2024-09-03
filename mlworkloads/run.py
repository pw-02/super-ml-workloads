from omegaconf import DictConfig, OmegaConf
import hydra
from lightning.fabric.loggers import CSVLogger
from image_classifer import train_image_classifer
import os
from lightning.pytorch.core.saving import save_hparams_to_yaml

@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(config: DictConfig):

    print(OmegaConf.to_yaml(config, resolve=True))

    log_dir = f"{config.log_dir}/{config.workload.replace('/','_')}/{config.exp_id}/{config.job_id}".lower()
    log_dir = os.path.normpath(log_dir)  # Normalize path for Windows
    
    save_hparams_to_yaml(os.path.join(log_dir, "hparms.yaml"), config)

    train_logger = CSVLogger(root_dir=log_dir, name="train", prefix='', flush_logs_every_n_steps=config.log_interval)
    val_logger = CSVLogger(root_dir=log_dir, name="val", prefix='', flush_logs_every_n_steps=config.log_interval)

    if config.workload == 'cifar10_resnet18' or config.workload.name == 'imagenet_resnet50':
        train_image_classifer(config, train_logger,val_logger )

    elif config.workload == 'Pythia-14M/OpenWebText':
        pass
    else:
        raise ValueError(f"Invalid workload: {config.workload}")
    

if __name__ == "__main__":
    main()
