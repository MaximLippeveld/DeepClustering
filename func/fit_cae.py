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
from math import ceil
import time
from torch import multiprocessing

multiprocessing.set_start_method('spawn', True)

def epoch_reporting(output, queue, event):

    # tensorboard writer
    writer = SummaryWriter(output / "tb")

    while not event.is_set():
        item = queue.get()
        global_step = item["global_step"]
        
        input_grid = utils.make_grid(item["input_grid"], nrow=3, normalize=True)
        output_grid = utils.make_grid(item["output_grid"], nrow=3, normalize=True)

        writer.add_embedding(item["embeddings"], label_img=item["label_imgs"], global_step=global_step)
        
        writer.add_image("training/input", input_grid, global_step=global_step)
        writer.add_image("training/output", output_grid, global_step=global_step)
        writer.add_scalar("training/loss", item["running_loss_avg"], global_step=global_step)
        
        for n, avg in item["running_gradients_avgs"].items():
            writer.add_histogram("gradients/%s" % n, avg, global_step=global_step)

    writer.close()


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

    # get model, optimizer, loss
    cae = model.cae.ConvolutionalAutoEncoder(img_shape, args.embedding_size)
    if args.cuda:
        cae.cuda()
    # writer.add_graph(cae, next(iter(loader_aug)))
    opt = torch.optim.Adam(cae.parameters())

    logging.info("Trainable model parameters: %d" % sum(p.numel() for p in cae.parameters() if p.requires_grad))

    # training loop
    running_loss = util.metrics.AverageMeter("loss", (1,), cuda=args.cuda)
    running_gradients = {
        n: util.metrics.AverageMeter(n, cuda=args.cuda)
        for n,p in cae.named_parameters() if p.requires_grad
    }

    global_step = 0
    embeddings_to_save_per_epoch = 5000
    steps_per_epoch = ceil(len(ds)/args.batch_size)
    embeddings_to_save_per_step = int(embeddings_to_save_per_epoch/steps_per_epoch)

    manager = multiprocessing.Manager()
    queue = manager.Queue()
    event = manager.Event()
    consumer = multiprocessing.Process(target=epoch_reporting, args=(args.output, queue, event))
    consumer.start()

    for epoch in tqdm(range(args.epochs)):
        # iterate over data

        embeddings = torch.empty((embeddings_to_save_per_epoch, args.embedding_size), dtype=np.float)
        label_imgs = torch.empty(tuple([embeddings_to_save_per_epoch] + list(img_shape)), dtype=np.float)
        for b_i, batch in enumerate(tqdm(loader_aug, leave=False)):
            global_step += 1
            opt.zero_grad()

            embedding = cae.encoder(batch)

            embeddings[
                b_i*embeddings_to_save_per_step:
                (b_i+1)*embeddings_to_save_per_step
            ] = embedding.detach().cpu()[:embeddings_to_save_per_step]
            label_imgs[
                b_i*embeddings_to_save_per_step:
                (b_i+1)*embeddings_to_save_per_step
            ] = batch.detach().cpu()[:embeddings_to_save_per_step]

            target = cae.decoder(embedding)
            loss = F.mse_loss(batch, target, reduction="mean")
            loss.backward()
            running_loss.update(loss)
            for n, p in cae.named_parameters():
                if epoch==0 and b_i==0:
                    running_gradients[n].reset(p.grad.shape)

                if p.requires_grad:
                    running_gradients[n].update(p.grad)
            
            opt.step()

            if b_i == 0:
                break

        # reporting
        item = {
            "input_grid": batch.detach().cpu()[:15],
            "output_grid": embedding.detach().cpu()[:15],
            "embeddings": embeddings,
            "label_imgs": label_imgs,
            "running_loss_avg": running_loss.avg.cpu(),
            "running_gradients_avgs": {},
            "global_step": global_step
        }
        for n, rg in running_gradients.items():
            item["running_gradients_avgs"][n] = rg.avg.cpu()

        queue.put(item)
        
        running_loss.reset()
        for k, v in running_gradients.items():
            v.reset()

    torch.save(cae.state_dict(), args.output / "model.pth")

    event.set()
    consumer.join()
    
