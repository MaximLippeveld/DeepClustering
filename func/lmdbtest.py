from data import sets
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path

def main(args):

    ds = sets.LMDBDataset(str(args.data), args.channels, 45)

    loader = DataLoader(
        ds, batch_size=128, shuffle=False, 
        drop_last=False, num_workers=0
    )

    batch = next(iter(loader)).cpu()
    fig, axes = plt.subplots(1, 2)

    axes[0].imshow(batch[0, 0])
    axes[1].imshow(batch[1, 0])
        
    plt.show()
