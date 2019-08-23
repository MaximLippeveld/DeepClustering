from torch.utils.data import Sampler
import numpy

class FoldIndexFileSampler(Sampler):

    def __init__(self, idx_file, batch_size=256, shuffle=True):
        self.batch_size = batch_size
        self.idx = np.loadtxt(idx_file, dtype=int)
        self.shuffle = shuffle
        self.steps_per_epoch = int(len(idx)//batch_size)

    def __iter__(self):
        if self.shuffle:
            self.idx = numpy.random.shuffle(self.idx)

        for i in range(self.steps_per_epoch):
            yield self.idx[i*self.batch_size:(i+1)*self.batch_size]
