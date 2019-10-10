from torch.utils.data import Dataset
import torch
from torchvision import transforms
from torchvision.transforms import functional
import numpy
import numpy as np
import lmdb
import os.path
import h5py
import h5pickle
import pickle
import math
from math import ceil, floor


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

    def __init__(self, db_path, channels, size, length, transform=None):
        self.channels = channels
        self.db_path = db_path
        self.size = size
        self.length = length
        self.idx_bytes = int(numpy.ceil(numpy.floor(numpy.log2(self.length))/8.))
        self.env = None
        self.transform = transform

    def setup(self):
        self.env = lmdb.open(self.db_path, subdir=os.path.isdir(self.db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)

    def __getitem__(self, index):
        if self.env is None:
            self.setup()

        env = self.env
        size = self.size

        idx = index.to_bytes(self.idx_bytes, "big")

        with env.begin(write=False) as txn:
            byteflow = txn.get(idx)

        width, height, image, mask = pickle.loads(byteflow)

        image = np.multiply(
            np.float32(image),
            np.float32(mask)
        )

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

class HDF5Dataset(Dataset):

    def __init__(self, handle, channels):
        
        self.handle = h5pickle.File(str(handle), "r", libver="latest", swmr=True)

        self.channels = []
        channel_fmt = "channel_%d"
        for channel in channels:
            if channel_fmt % channel in self.handle:
                self.channels.append(channel_fmt % channel)
            else:
                raise ValueError("Channel %d not present in data-file." % channel)
            
        shape = list(self.handle[self.channels[0]]["images"].shape)
        shape.insert(1, len(self.channels))
        self.shape = tuple(shape)

    def __len__(self):
        return self.get_shape()[0]

    def get_shape(self):
        return self.shape

    def __getitem__(self, idx):
        im = numpy.array([self.handle[channel]["images"][idx] for channel in self.channels], dtype=numpy.float32)
        im *= numpy.array([self.handle[channel]["masks"][idx] for channel in self.channels], dtype=numpy.float32)

        return im

    