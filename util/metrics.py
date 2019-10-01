import torch
import numpy as np
import scipy.optimize
from sklearn.metrics import normalized_mutual_info_score

nmi = normalized_mutual_info_score

def acc(y_true, y_pred):
    """
    Calculate clustering accuracy.

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(w.max() - w)

    return w[row_ind, col_ind].sum() / y_pred.size

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, shape=None, fmt=':f', cuda=True):
        self.name = name
        self.fmt = fmt
        self.cuda = cuda
        if shape is not None:
            self.reset(shape)

    def reset(self, shape=None):
        if shape == None:
            shape = self.shape
        else:
            self.shape = shape

        if self.cuda:
            device = torch.cuda.current_device()
        else:
            device = torch.device("cpu")

        self.val = torch.zeros(shape, device=device)
        self.avg = torch.zeros(shape, device=device)
        self.sum = torch.zeros(shape, device=device)
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)