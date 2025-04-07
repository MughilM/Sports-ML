# File: src/spock_configs.py
# Author: Mughil Pari
# Creation Date: 2025-03-27
#
# This file contains all the Config class definitions to be used with Spock.
# Class inheritance is used throughout.

import os
from enum import Enum
from spock import spock, directory, file
from typing import List, Optional

@spock
class ClassOne:
    one: int
    two: str


@spock
class ClassTwo:
    one: int
    two: str


class ClassChoice(Enum):
    class_one = ClassOne
    class_two = ClassTwo

@spock
class TypeConfig:
    param: List[ClassChoice]

@spock
class ModelCheckpointConfig:
    dirpath: directory = os.path.join("${spock.var:PathsConfig.log_dir", "checkpoints")
    filename: str = "epoch_{epoch:03d}"
    monitor: str = "val/acc"  # name of the logged metric which determines when model is improving
    verbose: bool = False  # verbosity mode
    save_last: bool = True  # additionally always save an exact copy of the last checkpoint to a file last.ckpt
    save_top_k: int = 1  # save k best models (determined by above metric)
    mode: str = "max"  # "max" means higher metric value is better, can be also "min"
    auto_insert_metric_name: bool = True  # when True, the checkpoints filenames will contain the metric name
    save_weights_only: bool = False  # if True, then only the model’s weights will be saved
    every_n_train_steps: Optional[int] = None  # number of training steps between checkpoints
    train_time_interval: Optional[int] = None  # checkpoints are monitored at the specified time interval
    every_n_epochs: Optional[int] = None  # number of epochs between checkpoints

@spock
class RichProgressBarConfig:
    refresh_rate: Optional[int] = 1
    leave: Optional[bool] = False

@spock
class PlotConfusionMatrixConfig:
    labels: List[str] = ['0', '1']
    matrix_attr: str = 'matrix'
    val_acc_attr: str = 'val_acc'

class CallbackChoice(Enum):
    model_checkpoint = 'model_checkpoint'
    rich_progress_bar = 'rich_progress_bar'
    plot_confusion_matrix = 'plot_confusion_matrix'

@spock
class CallbackConfig:
    callbacks: List[CallbackChoice]


@spock
class PathsConfig:
    """
    Path configurations, such as the root directory,
    data directory, and logging directories.

    Attributes:
        root_dir: The root directory of the whole project
        data_dir: The directory containing the data files
        log_dir: The directory for logging
    """
    root_dir: directory = "${spock.env:PROJECT_ROOT}"
    data_dir: directory = os.path.join("${spock.var:PathsConfig.root_dir}", "data")
    log_dir: directory = os.path.join("${spock.var:PathsConfig.root_dir}", "logs")