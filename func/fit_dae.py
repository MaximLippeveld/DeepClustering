import data
import model.dae
import torch.nn.functional as F
import torch.optim
from tqdm import tqdm
from torchvision import datasets, transforms
from pathlib import Path
from util.augmentation.augmentation_2d import *
from torch.utils.data import DataLoader

def main(args):

    # prepare data
    if type(args.data) is Path:
        ds = sets.HDF5Dataset(args.data, args.channels)
        shape = ds.get_shape()[1:]
        augs = [ToTensor()]
    else:
        ds = datasets.FashionMNIST("data/datasets/", train=True, download=True).data
        shape = tuple(ds.shape[1:])
        augs = [Stack()]

    augs += [
        ToFloatTensor(),
        FlipX(shape),
        FlipY(shape),
        RandomDeformation(shape, sampling_interval=12),
        RotateRandom(shape)   
    ]
    augmenter = transforms.Compose(augs)

    loader_aug = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, 
        drop_last=False, num_workers=0,
        collate_fn=augmenter
    )

    # get model, optimizer, loss
    dae = model.dae.DAE(shape, 10)

    dae.cuda()

    opt = torch.optim.Adam(dae.parameters())

    # training loop
    for epoch in tqdm(range(args.epochs)):
        # iterate over data
        for batch in loader_aug:
            opt.zero_grad()

            target = dae(batch)
            loss = torch.nn.functional.cross_entropy(batch, target)
            loss.backward()

            opt.step()
    
