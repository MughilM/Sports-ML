# File: src/modules.py
# Author: Mughil Pari
# Creation Date: 2025-04-27
#
# Contains all Lightning Module definitions. By default, each module takes a pure nn.Module
# and a SpockConfig namespace by default. The latter is used to gather the optimizer, because
# this needs to be instantiated using the parameters, which aren't define in Spock.

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import MeanMetric, MaxMetric, ConfusionMatrix
from torchmetrics.classification import BinaryAccuracy

import pytorch_lightning as pl
from spock.backend.wrappers import Spockspace
from src.utils import instantiate_optimizer

class CancerImageClassifier(pl.LightningModule):
    def __init__(self, net: nn.Module, config: Spockspace):
        super().__init__()
        # Save all the hyperparameters, they'll become available as self.hparams
        # self.save_hyperparameters(logger=False, ignore=['net'])

        self.net = net
        self.optimizer: Optional[optim.Optimizer] = None
        self.spock_config = config

        # The loss function
        self.loss = nn.BCEWithLogitsLoss()

        # All tasks will use binary accuracy
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()

        # We also need to average losses across batches, so set MeanMetrics up...
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        # Finally, have a confusion matrix...
        self.matrix = ConfusionMatrix(task='binary')

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def configure_optimizers(self):
        # Use the Spock config to create an optimizer object
        # using the model's parameters (accessed with "self")
        # optimizer = self.optimizer(params=self.parameters())
        optimizer = instantiate_optimizer(self.spock_config, self.parameters())
        # TODO: Add schedulers if you want
        return optimizer

    # def on_train_start(self) -> None:
    #     # Reset the best validation accuracy due to sanity checks that pl does
    #     self.best_val_acc.reset()

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.loss(outputs, targets.unsqueeze(dim=-1).float())
        self.train_loss(loss)  # Update our current loss, will hold average loss so far...
        self.train_acc(outputs.squeeze(), targets)
        self.log('train/loss', self.train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train/acc', self.train_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'acc': self.train_acc}

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.loss(outputs, targets.unsqueeze(dim=-1).float())
        self.val_loss(loss)  # Update our current validation loss
        self.val_acc(outputs.squeeze(), targets)
        self.matrix.update(outputs.squeeze(), targets)
        self.log('val/loss', self.val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val/acc', self.val_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'acc': self.val_acc}

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        inputs, _ = batch
        outputs = self(inputs)
        return nn.Sigmoid()(outputs)