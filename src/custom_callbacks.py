# File: src/custom_callbacks.py
# Author: Mughil Pari
# Creation Date: 2025-04-06
#
# Contains custom-defined callbacks. Each Callback class needs to be sub-classed
# from the PyTorch Lightning Callback object.
# Also make sure this is available to be selected from CallBackConfig
# (Add it to the enum)
# (Add switch case in instantiate_callbacks)

import sys
import io
from PIL import Image
from typing import List
import logging

import numpy as np
import plotly.express as px

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from torchmetrics import ConfusionMatrix, Accuracy

log = logging.getLogger('main.custom_callbacks')


class PlotMulticlassConfusionMatrix(Callback):
    """
    This callback plots a simple confusion matrix, and logs it to Weights and Biases as well.
    This is designed to be used for multiclass classification, where each label is mutually exclusive.
    To plot a multilabel matrix, where each label is NOT exclusive, please use PlotMultilabelConfusionMatrix.
    The plot only happens at the end of validation.
    """
    def __init__(self, labels: List, matrix_attr: str = 'matrix', val_acc_attr: str = 'val_acc'):
        self.labels = list(labels)
        self.matrix_attr = matrix_attr
        self.val_acc_attr = val_acc_attr

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # Get the objects for the confusion matrix and validation accuracy, and exit the
        # program if the specific ones aren't available.
        matrix: ConfusionMatrix = getattr(pl_module, self.matrix_attr)
        acc_metric: Accuracy = getattr(pl_module, self.val_acc_attr)

        if matrix is None:
            log.error(f'Matrix of name "{self.matrix_attr}" not available! Exiting...')
            sys.exit(1)
        if acc_metric is None:
            log.error(f'Accuracy metrix of name "{self.val_acc_attr}" not available! Exiting...')
            sys.exit(1)

        result: np.ndarray = pl_module.matrix.compute().cpu().numpy().T
        fig = px.imshow(result, text_auto=True, x=self.labels, y=self.labels,
                        title=f'Accuracy: {pl_module.val_acc.compute() * 100:2.3f}%')
        fig.update_xaxes(side='top', type='category', title='Actual')
        fig.update_yaxes(type='category', title='Predicted')
        img_bytes = fig.to_image(engine='kaleido')

        # Reset the matrix
        pl_module.matrix.reset()
        # Log it in W and B
        pl_module.logger.log_image('conf_matrix', [Image.open(io.BytesIO(img_bytes))])