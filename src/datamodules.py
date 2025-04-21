# File: src/datamodules.py
# Author: Mughil Pari
# Creation Date: 2025-04-13
#
# This file contains definitions for PyTorch Lightning DataModule classes.
# In contrast to other projects, I have decided to place everything in a single file,
# to easily take advantage of certain class inheritances. For example, most modules
# have the same basic configuration (validation split, batch size, etc.), while
# DataModules that use Kaggle as a data source generally need the competition name.

import os
from typing import Optional
import zipfile
import kaggle
import glob
import logging

from sklearn.model_selection import train_test_split
import pandas as pd
from PIL import Image

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

from src.datasets import *

log = logging.getLogger('train.datamodules')

class KaggleDataModule(pl.LightningDataModule):
    def __init__(self, comp_name, data_dir, downsample_n: int = -1, train_frac: float = 1.0,
                 validation_split: float = 0.2, batch_size: int = 128, num_workers: int = -1,
                 pin_memory: bool = True):
        super().__init__()
        # Save the hyperparameters
        self.save_hyperparameters()
        self.COMP_DATA_PATH = os.path.join(self.hparams.data_dir, self.hparams.comp_name)
        # Rest is blank. If there are any additional hyperparameters that need to be saved,
        # it will need to be done manually.

class CancerDataModule(KaggleDataModule):
    def __init__(self, comp_name: str = 'histopathologic-cancer-detection', data_dir: str = 'data/',
                 downsample_n: int = 10000, train_frac: float = 1.0, validation_split: float = 0.2,
                 batch_size: int = 2048, num_workers: int = 1, pin_memory: bool = True,
                 image_size: int = 32):
        super().__init__(comp_name, data_dir, downsample_n, train_frac, validation_split,
                         batch_size, num_workers, pin_memory)
        # Saving the hyperparameters allows all the parameters to be accessible with self.hparams
        self.save_hyperparameters()
        # Transforms
        self.train_transform = T.Compose([
            T.CenterCrop(self.hparams.image_size),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ToTensor()
        ])
        self.vali_test_transform = T.Compose([T.CenterCrop(self.hparams.image_size), T.ToTensor()])

        self.train_dataset: Optional[Dataset] = None
        self.vali_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    @property
    def num_classes(self):
        return 1

    def prepare_data(self) -> None:
        """
        Download data from Kaggle if we need. No assignments here.
        :return:
        """
        # Download the competition files if the directory doesn't exist
        if not os.path.exists(self.COMP_DATA_PATH):
            log.info(f'Downloading {self.hparams.comp_name} competition files...')
            kaggle.api.competition_download_files(self.hparams.comp_name, path=self.hparams.data_dir, quiet='False')
            log.info(f'Extracting contents into {self.COMP_DATA_PATH}...')
            with zipfile.ZipFile(os.path.join(self.hparams.data_dir, f'{self.hparams.comp_name}.zip'), 'r') as zip_ref:
                zip_ref.extractall(self.COMP_DATA_PATH)
            log.info('Deleting zip file...')
            os.remove(os.path.join(self.hparams.data_dir, f'{self.hparams.comp_name}.zip'))

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Load the data and actually assign the Dataset objects (train, vali, and test).
        This is where the splitting happens as well.
        :param stage:
        :return:
        """
        if not self.train_dataset and not self.vali_dataset and not self.test_dataset:
            # Read the training labels from the standalone csv
            labels = pd.read_csv(os.path.join(self.COMP_DATA_PATH, 'train_labels.csv'))
            # Downsample however many we need
            # If downsample_n = -1, then we take the entire dataset.
            if self.hparams.downsample_n != -1:
                labels = labels.sample(n=self.hparams.downsample_n)
            # Split training and validation
            train_files, validation_files = train_test_split(labels, test_size=self.hparams.validation_split)
            log.info(f'Number of images in training: {len(train_files)}')
            log.info(f'Number of images in validation: {len(validation_files)}')
            # Now load the Dataset objects
            self.train_dataset = CancerDataset(data_folder=os.path.join(self.COMP_DATA_PATH, 'train'),
                                               file_id_df=train_files,
                                               transform=self.train_transform)
            self.vali_dataset = CancerDataset(data_folder=os.path.join(self.COMP_DATA_PATH, 'train'),
                                              file_id_df=validation_files,
                                              transform=self.vali_test_transform)
            self.test_dataset = CancerDataset(data_folder=os.path.join(self.COMP_DATA_PATH, 'test'),
                                              transform=self.vali_test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True,
                          num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory,
                          persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.vali_dataset, batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory,
                          persistent_workers=True)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory,
                          persistent_workers=True)