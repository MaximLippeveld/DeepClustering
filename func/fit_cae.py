import data
import model.cae
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
        img_shape = ds.get_shape()[1:]
        channel_shape = ds.get_shape()[2:]
        pre_augs = [ToTensor(cuda=args.cuda)]
        post_augs = []
    else:
        ds = datasets.FashionMNIST("data/datasets/", train=True, download=True).data
        channel_shape = ds.shape[1:]
        ds = torch.unsqueeze(ds, 1)
        img_shape = ds.shape[1:]
        pre_augs = [Stack(cuda=args.cuda)]
        post_augs = []

    augs = pre_augs + [
        ToFloatTensor(cuda=args.cuda),
        FlipX(channel_shape, cuda=args.cuda),
        FlipY(channel_shape, cuda=args.cuda),
        RandomDeformation(channel_shape, sampling_interval=7, cuda=args.cuda),
        RotateRandom(channel_shape, cuda=args.cuda)
    ] + post_augs
    augmenter = transforms.Compose(augs)

    loader_aug = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, 
        drop_last=False, num_workers=0,
        collate_fn=augmenter
    )

    # get model, optimizer, loss
    cae = model.cae.ConvolutionalAutoEncoder(img_shape, 10)

    if args.cuda:
        cae.cuda()

    opt = torch.optim.Adam(cae.parameters())

    # training loop
    for epoch in tqdm(range(args.epochs)):
        # iterate over data
        for batch in tqdm(loader_aug):
            opt.zero_grad()

            target = cae(batch)
            loss = F.mse_loss(batch, target, reduction="mean")
            loss.backward()

            opt.step()
    
