import data
import model.dae
import torch.nn.functional as F
import torch.optim
import tqdm

def main(args):

    # prepare data
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
        ds, batch_size=args.batch_size, shuffle=False, 
        drop_last=False, num_workers=5,
        collate_fn=augmenter
    )

    # get model, optimizer, loss
    dae = model.dae.DAE(shape, 10)
    opt = torch.optim.Adam(dae.params())

    # training loop
    for epoch in tqdm(range(args.epochs)):
        # iterate over data
        for batch in loader_aug:
            opt.zero_grad()

            target = dae(batch)
            loss = torch.nn.functional.cross_entropy(batch, target)
            loss.backward()

            opt.step()
    
