# File: src/datasets.py
# Author: Mughil Pari
# Creation: 2025-04-13
#
# This contains the PyTorch Dataset class definitions. This is used by the PyTorch
# Lightning DataModules, but these are pure Dataset classes. At minimum,
# they implement __init__, __len__, and __getitem__.

import os
import glob
import logging
from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as T

logger = logging.getLogger('datasets')

class CancerDataset(Dataset):
    def __init__(self, data_folder, file_id_df=None, transform=T.Compose([T.CenterCrop(32), T.ToTensor()]),
                 dict_labels={}):
        self.data_folder = data_folder
        # Create the file list from the IDs
        if file_id_df is None:
            self.file_ids = [file[:-4] for file in glob.glob(os.path.join(data_folder, '*.tif'))]
            self.labels = [-1] * len(self.file_ids)  # This should not be used in test_step for the model
        else:
            self.file_ids = [os.path.join(self.data_folder, file_id) for file_id in file_id_df['id']]
            self.labels = file_id_df['label'].values
        self.transform = transform
        self.dict_labels = dict_labels

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        # Read the image using the filepath and apply the transform
        image = Image.open(f'{self.file_ids[idx]}.tif')
        image = self.transform(image)
        # Return the label as well...
        return image, self.labels[idx]
