from data import sets
from torch.utils.data import DataLoader
from util.augmentation import augmentation_2d

def main(args):
    ds = sets.HDF5Dataset(args.data, args.channels, augmentation=augmentation_2d.Rotate90)
    loader = DataLoader(ds, batch_size=32, shuffle=True, drop_last=True, pin_memory=True, num_workers=8)

    for i_batch, sample_batched in enumerate(loader):
        print(
            i_batch, sample_batched.size())
