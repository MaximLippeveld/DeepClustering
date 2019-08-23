from torch.utils.data import Dataset
import h5py
import numpy

class HDF5Dataset(Dataset):

    def __init__(self, handle, channels, augmentation=None):
        self.handle = h5py.File(handle, "r")

        self.channels = []
        channel_fmt = "channel_%d"
        for channel in channels:
            if channel_fmt % channel in self.handle:
                self.channels.append(channel_fmt % channel)
            else:
                raise ValueError("Channel %d not present in data-file." % channel)

        if augmentation is None:
            self.getitem = self.getitem
        else:
            shape = list(self.handle[self.channels[0]]["images"].shape[1:])
            shape = [len(self.channels)] + shape
            self.augmentation = augmentation(shape)
            self.getitem = self.getitem_withaug

    def __len__(self):
        return len(self.handle[self.channels[0]]["images"])

    def __getitem__(self, idx):
        self.getitem(idx)

    def getitem(self, idx):
        im = numpy.array([self.handle[channel]["images"][idx] for channel in self.channels], dtype=numpy.float64)
        masks = numpy.array([self.handle[channel]["masks"][idx] for channel in self.channels], dtype=numpy.float64)
        return im*masks

    def getitem_withaug(self, idx):
        im = numpy.array([self.handle[channel]["images"][idx] for channel in self.channels], dtype=numpy.float64)
        masks = numpy.array([self.handle[channel]["masks"][idx] for channel in self.channels], dtype=numpy.float64)
        return self.augmentation(im*masks)
    