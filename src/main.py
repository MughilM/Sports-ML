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
from src.utils import instantiate_callbacks, instantiate_net, instantiate_module, instantiate_datamodule
from src.datasets import *
from src.datamodules import *
import colorlog

handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
	'%(log_color)s%(asctime)s - %(name)s(%(funcName)s) - %(levelname)s || %(message)s',
    datefmt=None,
	reset=True,
    log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'red,bg_white',
    },
    secondary_log_colors={},
    style='%'
))

logger = colorlog.getLogger('main')
logger.addHandler(handler)

logger.setLevel(logging.DEBUG)





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

    logger.info('Instantiating callbacks...')
    callbacks = instantiate_callbacks(config)

    logger.debug(callbacks)

    logger.debug(config.PathsConfig)

    logger.info('Instantiating datamodule...')
    datamodule = instantiate_datamodule(config)

    logger.info('Instantiating model...')
    net = instantiate_net(config)
    model = instantiate_module(config, net)
    logger.debug(model)

    logger.info('Instantiating trainer...')
    logger.debug(config.TrainerConfig)
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


    logger.info('Starting training...')
    trainer.fit(model, datamodule=datamodule)
    logger.info('Training complete!')


if __name__ == "__main__":
    main()