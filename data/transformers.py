import torch

class MinMax:
    def __call__(self, x):
        min_ = torch.flatten(x, 1).min(dim=1).values
        max_ = torch.flatten(x, 1).max(dim=1).values

        return ((x.T - min_)/(max_ - min_)).T
