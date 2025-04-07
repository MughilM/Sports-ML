import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=['.idea', '.git'],
    pythonpath=True,
    project_root_env_var=True
)

from typing import Optional
import logging
import os

import torch
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import Callback, RichProgressBar

from spock import SpockBuilder

from src.spock_configs import *
from src.utils import instantiate_callbacks


def main():
    description = 'Spock basic usage'
    config = SpockBuilder(
        PathsConfig,
        CallbackConfig,
        ModelCheckpointConfig,
        RichProgressBarConfig,
        PlotConfusionMatrixConfig,
        desc='A basic ML configuration'
    ).generate()

    print('Instantiating callbacks...')
    callbacks = instantiate_callbacks(config)

    print(callbacks)

    print(config.PathsConfig)

if __name__ == "__main__":
    main()