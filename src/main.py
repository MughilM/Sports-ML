# File: src/main.py
# Author: Mughil Pari
# Creation Date: 2025-04-01
#
# This contains the main workflow for training any models that we need.
# We use the Spock configuration framework, as opposed to Hydra.
# The benefit is that we get distinct type checking and hinting for
# all the classes. It is just to try something different

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
from src.utils import instantiate_callbacks, instantiate_net, instantiate_module
from src.datasets import *
from src.datamodules import *


def main():
    description = 'Spock basic usage'
    config = SpockBuilder(
        PathsConfig,
        ModelCheckpointConfig,
        RichProgressBarConfig,
        PlotConfusionMatrixConfig,
        KaggleDataConfig,
        CancerDataConfig,
        SimpleConvConfig,
        ResNetConfig,
        ViTConfig,
        OptimizerConfig,
        AdamConfig,
        SGDConfig,
        RunConfig,
        TrainerConfig,
        desc='A basic ML configuration'
    ).generate()

    print('Instantiating callbacks...')
    callbacks = instantiate_callbacks(config)

    print(callbacks)

    print(config.PathsConfig)

    print('Instantiating datamodule...')
    datamodule = Optional[LightningDataModule]
    dm_config = config.RunConfig.datamodule
    if dm_config == 'cancer_data':
        dm_config = config.CancerDataConfig
        datamodule = CancerDataModule(
            comp_name=dm_config.comp_name,
            data_dir=dm_config.data_dir,
            downsample_n=dm_config.downsample_n,
            train_frac=dm_config.train_frac,
            validation_split=dm_config.validation_split,
            batch_size=dm_config.batch_size,
            num_workers=dm_config.num_workers,
            pin_memory=dm_config.pin_memory,
            image_size=dm_config.image_size,
        )
    datamodule.prepare_data()
    datamodule.setup()

    print('Instantiating model...')
    net = instantiate_net(config)
    model = instantiate_module(config, net)
    print(model)

    print('Instantiating trainer...')
    print(config.TrainerConfig)
    trainer = Trainer(
        min_epochs=config.TrainerConfig.min_epochs,
        max_epochs=config.TrainerConfig.max_epochs,
        devices=config.TrainerConfig.devices,
        accelerator=config.TrainerConfig.accelerator,
        check_val_every_n_epoch=config.TrainerConfig.check_val_every_n_epoch,
        log_every_n_steps=config.TrainerConfig.log_every_n_steps,
        deterministic=config.TrainerConfig.deterministic,
        callbacks=callbacks
    )

    print('Starting training...')
    trainer.fit(model, datamodule=datamodule)
    print('Training complete!')


if __name__ == "__main__":
    main()