from jsonargparse import CLI, ArgumentParser, ActionConfigFile
from launch_job import launch_job
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from functools import partial
import hydra

def initialize_parser(hpo_config_file: str) -> ArgumentParser:
    parser = ArgumentParser(prog="app", description="Description for my app", default_config_files=[hpo_config_file])
    parser.add_argument("--base_config_file", type=str, default=None, required=True)
    parser.add_argument("--search_space_file", type=str, default=None, required=True)
    parser.add_argument("--num_trials", default=1, type=int, required=True)
    parser.add_argument("--gpus_per_trial", default=1, type=int, required=True)
    parser.add_argument("--cpus_per_trial", default=1, type=int, required=True)
    parser.add_argument("--max_concurrent_trials", default=1, type=int, required=True)
    parser.add_argument("--config", action=ActionConfigFile)  

    return parser

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def launch_hpo(config_file: str) -> None:
    parser: ArgumentParser = initialize_parser(config_file)
    hparams = parser.parse_args(["--config", config_file])
    config = tune.Experiment.from_json(hparams.search_space_file)
    # config = {
    #         "l1": tune.choice([2**i for i in range(9)]),
    #         "l2": tune.choice([2**i for i in range(9)]),
    #         "lr": tune.loguniform(1e-4, 1e-1),
    #         "batch_size": tune.choice([256]),
    #         }
    
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=1,
        grace_period=1,
        reduction_factor=2,
    )
    result = tune.run(
        partial(launch_job, job_config_file=hparams.base_config_file),
        resources_per_trial={"cpu": hparams.cpus_per_trial, "gpu":hparams.gpus_per_trial},
        config=config,
        max_concurrent_trials=hparams.max_concurrent_trials,
        num_samples=hparams.num_trials,
        scheduler=scheduler,
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")
    



if __name__ == "__main__":
    defaults = {
       "config_file": 'configs/hpo-example-cfg.yaml',}
    
    from jsonargparse import CLI
    CLI(launch_hpo, set_defaults=defaults)
