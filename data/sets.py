from torch.utils.data import Dataset
import torch
import h5py
import numpy
import h5pickle

class HDF5Dataset(Dataset):

    def __init__(self, handle, channels, augmentation=None):
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

    