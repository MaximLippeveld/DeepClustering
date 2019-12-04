from data import sets
import torch
from torch.utils.data import DataLoader
from augmentation.augmentation_2d import *
from torchvision.transforms import Compose
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision import datasets, transforms

def main(args):
    # prepare data
    if type(args.data) is Path:
        ds = sets.HDF5Dataset(args.data, args.channels)
        shape = ds.get_shape()[1:]
        augs = [ToTensor()]
    else:
        ds = datasets.FashionMNIST("data/datasets/", train=True, download=True).data
        shape = tuple(ds.shape[1:])
        augs = [Stack(), Unsqueeze()]
    
    def_augmenter = transforms.Compose(augs)
    augs += [
        ToFloatTensor(),
        FlipX(shape),
        FlipY(shape),
        RandomDeformation(shape, sampling_interval=7),
        RotateRandom(shape)   
    ]
    augmenter = transforms.Compose(augs)

    loader_aug = DataLoader(
        ds, batch_size=2, shuffle=False, 
        drop_last=False, num_workers=0,
        collate_fn=augmenter
    )
    loader = DataLoader(
        ds, batch_size=2, shuffle=False, 
        drop_last=False, num_workers=0,
        collate_fn=def_augmenter
    )

    batch_aug, batch = next(iter(loader_aug)).cpu(), next(iter(loader)).cpu()
    fig, axes = plt.subplots(2, 2)

    axes[0][0].imshow(batch_aug[0][0])
    # axes[0][1].imshow(batch_aug[0][1])
    axes[1][0].imshow(batch[0][0])
    # axes[1][1].imshow(batch[0][1])
        
    plt.show()
