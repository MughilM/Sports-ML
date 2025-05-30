# File: src/utils.py
# Author: Mughil Pari
# Creation Date: 2025-04-06
#
# Contains various utility functions, mostly revolving around argument parsing.
# Spock allows for exact configs, but class instantiation still needs to be done manually.
from src.nets import *
from spock.backend.wrappers import Spockspace
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar, Callback
from src.custom_callbacks import *
from src.modules import CancerImageClassifier
from src.datamodules import CancerDataModule
from typing import List, Optional


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

def instantiate_datamodule(config: Spockspace) -> Optional[pl.LightningDataModule]:
    datamodule: Optional[pl.LightningDataModule] = None
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
    return datamodule

def instantiate_net(config: Spockspace) -> Optional[nn.Module]:
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

def instantiate_module(config: Spockspace, net_obj: nn.Module) -> Optional[pl.LightningModule]:
    module_choice = config.RunConfig.module
    match module_choice:
        case 'cancer_image_classifier':
            return CancerImageClassifier(net=net_obj, config=config)
    return None