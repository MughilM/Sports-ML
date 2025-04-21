# File: src/utils.py
# Author: Mughil Pari
# Creation Date: 2025-04-06
#
# Contains various utility functions, mostly revolving around argument parsing.
# Spock allows for exact configs, but class instantiation still needs to be done manually.
from src.nets import *
from src.spock_configs import RunConfig, ModelCheckpointConfig
from spock.backend.wrappers import Spockspace
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar, Callback
from src.custom_callbacks import *
from typing import List


def instantiate_callbacks(config: Spockspace):
    callback_list = config.RunConfig.callbacks
    # Only a few callbacks are allowed, but this logic is taken care of by Spock
    # due to using the CallbackChoice Enum.
    callbacks: List[Callback] = []  # The list that will actually contain the objects
    for config_name in callback_list:
        match config_name:
            case 'model_checkpoint':
                cfg = config.ModelCheckpointConfig
                callbacks.append(ModelCheckpoint(
                    dirpath=cfg.dirpath,
                    filename=cfg.filename,
                    monitor=cfg.monitor,
                    verbose=cfg.verbose,
                    save_last=cfg.save_last,
                    save_top_k=cfg.save_top_k,
                    mode=cfg.mode,
                    auto_insert_metric_name=cfg.auto_insert_metric_name,
                    save_weights_only=cfg.save_weights_only,
                    every_n_train_steps=cfg.every_n_train_steps,
                    train_time_interval=cfg.train_time_interval,
                    every_n_epochs=cfg.every_n_epochs
                ))
            case 'rich_progress_bar':
                cfg = config.RichProgressBarConfig
                callbacks.append(RichProgressBar(
                    refresh_rate=cfg.refresh_rate,
                    leave=cfg.leave
                ))
            case 'plot_confusion_matrix':
                cfg = config.PlotConfusionMatrixConfig
                callbacks.append(PlotMulticlassConfusionMatrix(
                    labels=cfg.labels,
                    matrix_attr=cfg.matrix_attr,
                    val_acc_attr=cfg.val_acc_attr
                ))
    return callbacks

def instantiate_model(config: Spockspace):
    net_choice = config.RunConfig.net
    match net_choice:
        case 'simple_conv':
            cfg = config.SimpleConvConfig
            return SimpleConvNet(name=cfg.name, output_size=cfg.output_size)
        case 'resnet50':
            cfg = config.ResNetConfig
            return ResNet(name=cfg.name, output_size=cfg.output_size, finetune=cfg.finetune)
        case 'vit_b_16':
            cfg = config.ViTConfig
            return ViT(name=cfg.name, output_size=cfg.output_size, pretrained=cfg.pretrained)
    return None