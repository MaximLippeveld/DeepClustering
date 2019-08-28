from data import sets
import torch
from torch.utils.data import DataLoader
from util.augmentation.augmentation_2d import *
from torchvision.transforms import Compose
import matplotlib.pyplot as plt

def main(args):
    ds = sets.HDF5Dataset(args.data, args.channels)
    shape = ds.get_shape()[1:]

    augmenter = Compose([
        ToTensor(),
        ToFloatTensor(),
        FlipX(shape),
        FlipY(shape),
        RandomDeformation(shape),
        RotateRandom(shape)   
    ])

    loader_aug = DataLoader(
        ds, batch_size=2, shuffle=False, 
        drop_last=False, num_workers=0,
        collate_fn=augmenter
    )
    loader = DataLoader(
        ds, batch_size=2, shuffle=False, 
        drop_last=False, num_workers=0
    )

    batch_aug, batch = next(iter(loader_aug)).cpu(), next(iter(loader)).cpu()
    fig, axes = plt.subplots(2, 2)

    axes[0][0].imshow(batch_aug[0][0])
    axes[0][1].imshow(batch_aug[0][1])
    axes[1][0].imshow(batch[0][0])
    axes[1][1].imshow(batch[0][1])
        
    plt.show()
