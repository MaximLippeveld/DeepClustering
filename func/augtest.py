from data import sets
import torch
from torch.utils.data import DataLoader
from util.augmentation import augmentation_2d

def main(args):
    ds = sets.HDF5Dataset(args.data, args.channels)

    augmentation = augmentation_2d.Rotate90(ds.get_shape())

    class collate_fn:
        def __init__(self, augmentation):
            self.augmentation = augmentation

        def __call__(self, samples):
            samples = torch.as_tensor(samples).cuda()
            return self.augmentation(samples)

    loader = DataLoader(
        ds, batch_size=32, shuffle=True, 
        drop_last=False, num_workers=0,
        collate_fn=collate_fn(augmentation)
    )

    for i_batch, sample_batched in enumerate(loader):
        print(
            i_batch, sample_batched.size())
