from torch.utils.data import Dataset
import torch
from torchvision import transforms
from torchvision.transforms import functional
import numpy
import numpy as np
import os.path
import pickle
import math
from math import ceil, floor
from cifconvert.lmdb import ciflmdb
import logging


def centercrop(width, height, image):
    spill = (
        max(0, (image.shape[1] - width)/2.), 
        max(0, (image.shape[2] - height)/2.)
    )

    return image[
        :,
        int(floor(spill[0])): -int(ceil(spill[0])) if spill[0] > 0 else None,
        int(floor(spill[1])): -int(ceil(spill[1])) if spill[1] > 0 else None
    ]

def centerpad(width, height, image):
    pad = (width-image.shape[1])/2., (height-image.shape[2])/2.

    tmp_im = numpy.zeros((image.shape[0], width, height), dtype=image.dtype)
    tmp_im[
        :,
        int(floor(pad[0])): -int(ceil(pad[0])) if pad[0] > 0 else None,
        int(floor(pad[1])): -int(ceil(pad[1])) if pad[1] > 0 else None
    ] = image

    return tmp_im


class LMDBDataset(Dataset):

    def __init__(self, db_paths, channels, size, raw_image, transform=None):
        self.db_paths = db_paths
        self.channels = channels
        self.size = size
        self.transform = transform
        self.raw_image = raw_image
        self.dbs = []
        self.db_start_index = []
        
        self.length = 0
        for db_path in self.db_paths:
            self.length += len(ciflmdb(db_path))

    def setup(self):
        i = 0
        for db_path in self.db_paths:
            db = ciflmdb(db_path)
            db.set_channels_of_interest(self.channels)

            self.dbs.append(db)
            self.db_start_index.append(i)
            
            i += len(db)

        self.db_start_index = np.array(self.db_start_index)

    def __getitem__(self, index):
        if len(self.dbs) == 0:
            self.setup()

        db_idx = sum(self.db_start_index - index <= 0)-1
        db = self.dbs[db_idx]
        start_idx = self.db_start_index[db_idx]
        width, height, image, mask = db.get_image(index-start_idx, only_coi=True)

        if not self.raw_image:
            image = np.multiply(
                np.float32(image),
                np.float32(mask)
            )
        else:
            image = np.float32(image)

        size = self.size
        if width > size or height > size:
            image = centercrop(size, size, image)

        if width < size or height < size:
            image = centerpad(size, size, image)

        if self.transform is not None:
            image = self.transform(image)

        return image

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'        
