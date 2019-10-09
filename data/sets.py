from torch.utils.data import Dataset
import torch
import numpy
import numpy as np
import lmdb
import os.path
import h5py
import h5pickle
import pickle
import math


def _pad_or_crop(image, _w, _h):
    """
    function adapted from Cellprofiler stitching library
    :param image:
    :param image_size:
    :return:
    """

    pad_x = float(max(image.shape[1], _w) - image.shape[1])
    pad_y = float(max(image.shape[2], _h) - image.shape[2])

    pad_width_x = (int(math.floor(pad_x / 2)), int(math.ceil(pad_x / 2)))
    pad_width_y = (int(math.floor(pad_y / 2)), int(math.ceil(pad_y / 2)))

    def normal(vector, pad_width, iaxis, kwargs):
        vector[:pad_width[0]] = 0
        vector[-pad_width[1]:] = 0
        return vector

    if (_h > image.shape[2]) and (_w > image.shape[1]):
        return np.pad(image, ((0,0), pad_width_y, pad_width_x), normal)
    else:
        if _w > image.shape[1]:
            temp_image = np.pad(image, pad_width_x, normal)
        else:
            if _h > image.shape[2]:
                temp_image = np.pad(image, pad_width_y, normal)
            else:
                temp_image = image

        return temp_image[:,
               int((temp_image.shape[2] - _h)/2):int((temp_image.shape[2] + _h)/2),
               int((temp_image.shape[1] - _w)/2):int((temp_image.shape[1] + _w)/2)
               ]


class LMDBDataset(Dataset):

    def __init__(self, db_path, channels):
        self.channels = channels
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=os.path.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = int.from_bytes(txn.get(b'__len__'), "big")

        self.idx_bytes = int(numpy.ceil(numpy.floor(numpy.log2(self.length))/8.))


    def __getitem__(self, index):
        env = self.env

        idx = index.to_bytes(self.idx_bytes, "big")

        with env.begin(write=False) as txn:
            byteflow = txn.get(idx)

        width, height, image, mask = pickle.loads(byteflow)
    
        return _pad_or_crop(np.multiply(image, mask), 90, 90)


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

    