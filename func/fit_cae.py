import data.sets, data.transformers
import model.cae
import torch.nn.functional as F
import torch.optim
from tqdm import tqdm
from torchvision import datasets, transforms, utils
from pathlib import Path
from util.augmentation.augmentation_2d import *
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import util.metrics
import logging
logging.basicConfig(level=logging.INFO)

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
        pre_augs = [Stack(cuda=args.cuda), ToFloatTensor(cuda=args.cuda), data.transformers.MinMax()]
        post_augs = []

    augs = pre_augs + [
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

    # tensorboard writer
    writer = SummaryWriter(args.output / "tb")

    # get model, optimizer, loss
    cae = model.cae.ConvolutionalAutoEncoder(img_shape, 10)
    if args.cuda:
        cae.cuda()
    writer.add_graph(cae, next(iter(loader_aug)))
    opt = torch.optim.Adam(cae.parameters())

    logging.info("Trainable model parameters: %d" % sum(p.numel() for p in cae.parameters() if p.requires_grad))

    # training loop
    running_loss = util.metrics.AverageMeter("loss", (1,), cuda=args.cuda)
    running_gradients = {
        n: util.metrics.AverageMeter(n, cuda=args.cuda)
        for n,p in cae.named_parameters() if p.requires_grad
    }

    for epoch in tqdm(range(args.epochs)):
        # iterate over data
        for b_i, batch in enumerate(tqdm(loader_aug, leave=False)):
            opt.zero_grad()

            target = cae(batch)
            loss = F.mse_loss(batch, target, reduction="mean")
            loss.backward()

            running_loss.update(loss)
            for n, p in cae.named_parameters():
                if epoch==0 and b_i==0:
                    running_gradients[n].reset(p.grad.shape)

                if p.requires_grad:
                    running_gradients[n].update(p.grad)

            opt.step()

        # reporting 
        input_grid = utils.make_grid(batch[:10], nrow=2, normalize=True)
        output_grid = utils.make_grid(target[:10], nrow=2, normalize=True)

        writer.add_image("training/input", input_grid, global_step=epoch)
        writer.add_image("training/output", output_grid, global_step=epoch)
        writer.add_scalar("training/loss", running_loss.avg, global_step=epoch)
        
        for n, rg in running_gradients.items():
            writer.add_histogram("gradients/%s" % n, rg.avg, global_step=epoch)

        running_loss.reset()
        for k, v in running_gradients.items():
            v.reset()

    torch.save(cae.state_dict(), args.output / "model.pth")

    writer.close()
    
