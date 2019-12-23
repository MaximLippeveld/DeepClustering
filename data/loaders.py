import data.sets, data.transformers
import imgaug as ia
import imgaug.augmenters as iaa
import torch
from torchvision import transforms, datasets
import numpy as np
from torch.utils.data import DataLoader


class IASeq:
    def __init__(self, seq):
        self.seq = seq

    def __call__(self, x):
        aug = self.seq(images=np.moveaxis(np.array(x), 1, -1))
        return np.moveaxis(aug, -1, 1)

class UnsupervisedFMNIST(datasets.FashionMNIST):
    def __getitem__(self, index):
        return super().__getitem__(index)[0]

class unsqueeze:
    def __init__(self, axis=0):
        self.axis = axis
    def __call__(self, x):
        return np.expand_dims(x, self.axis)

class DataLoaderWrapper(DataLoader):

    def __init__(self, args):

        if isinstance(args.data, list):
            ds = data.sets.LMDBDataset(args.data, args.channels, 90, args.raw_image)

            if len(args.channels) == 0:
                args.channels = ds.channels_of_interest
            img_shape = (len(args.channels), 90, 90)
            channel_shape = (90, 90)
        else:
            ds = UnsupervisedFMNIST("data/datasets/", train=True, download=True, transform=unsqueeze())
            channel_shape = ds.data.shape[1:]
            img_shape = [1] + list(channel_shape)

        augs = []
        if args.func not in ["embed"]:
            ia_seq = iaa.Sequential([
                iaa.Affine(rotate=(-160, 160), scale=(0.5, 1.5), translate_percent=(-0.1, 0.1)),
                iaa.HorizontalFlip(),
                iaa.VerticalFlip()
            ])
            augs.append(IASeq(ia_seq))

        augs += [
            torch.Tensor,
            data.transformers.StandardScale(),
        ]
        augmenter = transforms.Compose(augs)

        self.img_shape = img_shape
        self.channel_shape = channel_shape
        self.ds = ds

        super().__init__(
            ds, batch_size=args.batch_size, shuffle=True, 
            drop_last=False, num_workers=args.workers,
            collate_fn=augmenter
        )