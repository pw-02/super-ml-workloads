from omegaconf import DictConfig, OmegaConf
import hydra
from lightning.fabric.loggers import CSVLogger
from datetime import datetime
from mlworklaods.imagenet import run_imagenet

@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(config: DictConfig):
    # print(OmegaConf.to_yaml(config, resolve=True))

    logger = CSVLogger(root_dir=config.log_dir, name="", flush_logs_every_n_steps=config.log_interval)
    log_dir = logger.log_dir
    print(f"Log directory: {log_dir}")

    if config.workload.name == 'ResNet50/ImageNet':
        run_imagenet(config, logger)
    elif config.workload.name  == 'ResNet18/Cifar-10':
        pass
    elif config.workload.name == 'Pythia-14M/OpenWebText':
        pass
    else:
        raise ValueError(f"Invalid workload: {config.workload}")
