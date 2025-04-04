from pathlib import Path

import configue
import click

from colpali_engine.trainer.colmodel_training import ColModelTraining, ColModelTrainingConfig
from colpali_engine.utils.gpu_stats import print_gpu_utilization


@click.command()
@click.argument("config_file", type=click.Path(exists=True))
def main(config_file: Path) -> None:
    print_gpu_utilization()
    print("Loading config")
    config = configue.load(config_file, sub_path="config")
    print("Creating Setup")
    if isinstance(config, ColModelTrainingConfig):
        app = ColModelTraining(config)
    else:
        raise ValueError("Config must be of type ColModelTrainingConfig")

    if config.run_train:
        print("Training model")
        app.train()
        app.save(config_file=config_file)
    print("Done!")


if __name__ == "__main__":
    main()
