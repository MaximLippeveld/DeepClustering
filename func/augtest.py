from data import sets
import torch
from torch.utils.data import DataLoader
from util.augmentation import augmentation_2d
import matplotlib.pyplot as plt

def main(args):
    ds = sets.HDF5Dataset(args.data, args.channels)

    class collate_fn:
        def __init__(self, shape):
            self.transformations = [
                # augmentation_2d.Rotate90(shape, prob=0.5),
                augmentation_2d.RotateRandom(shape),
                augmentation_2d.FlipX(shape),
                augmentation_2d.FlipY(shape),
                augmentation_2d.RandomDeformation(shape, sigma=0.01)
            ]

        def __call__(self, samples):
            samples = torch.as_tensor(samples).cuda()

            for transform in self.transformations:
                samples = transform(samples)            

            return samples

    loader_aug = DataLoader(
        ds, batch_size=2, shuffle=False, 
        drop_last=False, num_workers=0,
        collate_fn=collate_fn(ds.get_shape()[1:])
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
