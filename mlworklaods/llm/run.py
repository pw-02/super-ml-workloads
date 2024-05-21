from omegaconf import DictConfig
import hydra
from lightning.pytorch.core.saving import save_hparams_to_yaml
from datetime import datetime
from args import *
from mlworklaods.llm.pretrain import setup
from mlworklaods.llm.data import *
# from mlworklaods.llm.data.openwebtext import OpenWebText
from mlworklaods.llm.data.text_files import TextFiles
from pathlib import Path
# Helper function to prepare arguments for a job



@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(hydra_config: DictConfig):

    train_args, data_args, dataloader_args = prepare_args(hydra_config, datetime.now().strftime("train_single_model_%Y-%m-%d_%H-%M-%S"))
    data = TextFiles(train_data_path=Path('data/openwebtxt/owt/train'), 
                     val_data_path=None)

    setup(
        model_name=train_args.model_name,
        model_config=None,
        precision=None,
        resume=False,
        data=data,
        train=train_args,
        # eval=None,
        devices=train_args.devices,
        logger_name='csv',
        seed=train_args.seed,)

if __name__ == "__main__":
    main()
