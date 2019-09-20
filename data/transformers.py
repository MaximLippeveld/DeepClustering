import torch

class MinMax:
    def __call__(self, x):
        tmp = torch.flatten(x, 1)

        return (x - tmp.min(dim=0).values)/(tmp.max(dim=0).values-tmp.min(dim=0).values)