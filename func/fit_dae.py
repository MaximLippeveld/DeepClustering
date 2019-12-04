import data
import model.dae
import torch.nn.functional as F
import torch.optim
from tqdm import tqdm
from torchvision import datasets, transforms
from pathlib import Path
from augmentation.augmentation_2d import *
from torch.utils.data import DataLoader


def main(args):

    # prepare data
    if type(args.data) is Path:
        ds = sets.HDF5Dataset(args.data, args.channels)
        shape = ds.get_shape()[1:]
        pre_augs = [ToTensor(cuda=args.cuda)]
        post_augs = []
    else:
        ds = datasets.FashionMNIST("data/datasets/", train=True, download=True).data
        shape = tuple(ds.shape[1:])
        pre_augs = [Stack(cuda=args.cuda), Unsqueeze(cuda=args.cuda)]
        post_augs = []

    augs = pre_augs + [
        ToFloatTensor(cuda=args.cuda),
        FlipX(shape, cuda=args.cuda),
        FlipY(shape, cuda=args.cuda),
        RandomDeformation(shape, sampling_interval=7, cuda=args.cuda),
        RotateRandom(shape, cuda=args.cuda)
    ] + post_augs
    augmenter = transforms.Compose(augs)

    loader_aug = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, 
        drop_last=False, num_workers=5,
        collate_fn=augmenter
    )

    # get model, optimizer, loss
    dae = model.dae.DAE(shape, 10)

    if args.cuda:
        dae.cuda()

    opt = torch.optim.Adam(dae.parameters())

    # training loop
    for epoch in tqdm(range(args.epochs)):
        # iterate over data
        for batch in tqdm(loader_aug):
            opt.zero_grad()

            target = dae(batch)
            flat = torch.flatten(batch, start_dim=1)
            loss = F.mse_loss(flat, target, reduction="mean")
            loss.backward()

            opt.step()
    
