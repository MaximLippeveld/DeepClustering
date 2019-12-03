import data.sets, data.transformers
import model.cae
import torch.nn.functional as F
import torch.optim, torch.autograd
from tqdm import tqdm
from torchvision import datasets, transforms, utils
from pathlib import Path
from util.augmentation.augmentation_2d import *
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import util.metrics
import logging
from math import ceil
import time
from torch import multiprocessing
import imgaug as ia
import imgaug.augmenters as iaa
import os
import psutil
from collections.abc import Iterable
import numpy as np

def reporting(output, queue):
    writer = SummaryWriter(output / "tb/rep")

    while True:
        item = queue.get()
        if item == None:
            break

        func, args = item
        getattr(writer, func)(*args)

        for arg in args:
            del arg
    
    writer.close()


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

def main(args):
    logger = logging.getLogger("root")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    handler = logging.FileHandler("root.log", mode="w")
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    # prepare data
    if isinstance(args.data, list):
        ds = data.sets.LMDBDataset(args.data, args.channels, 90, args.raw_image)
        img_shape = (len(args.channels), 90, 90)
        channel_shape = (90, 90)
    else:
        ds = UnsupervisedFMNIST("data/datasets/", train=True, download=True, transform=unsqueeze())
        channel_shape = ds.data.shape[1:]
        img_shape = [1] + list(channel_shape)

    ia_seq = iaa.Sequential([
        iaa.Affine(rotate=(-160, 160), scale=(0.5, 1.5), translate_percent=(-0.1, 0.1)),
        iaa.HorizontalFlip(),
        iaa.VerticalFlip()
    ])

    augs = [
        IASeq(ia_seq),
        torch.Tensor,
        data.transformers.MinMax(),
    ]
    augmenter = transforms.Compose(augs)

    loader_aug = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, 
        drop_last=False, num_workers=args.workers,
        collate_fn=augmenter
    )

    # get model, optimizer, loss
    cae = model.cae.ConvolutionalAutoEncoder(img_shape, args.embedding_size, args.dropout)
    logger.info("Trainable model parameters: %d" % sum(p.numel() for p in cae.parameters() if p.requires_grad))

    running_loss = util.metrics.AverageMeter("loss", (1,), cuda=args.cuda)
    running_gradients = {
        n: util.metrics.AverageMeter(n, cuda=args.cuda)
        for n,p in cae.named_parameters() if p.requires_grad
    }

    global global_step
    global_step = 0
    embeddings_to_save_per_epoch = 5000
    steps_per_epoch = ceil(len(ds)/args.batch_size)
    embeddings_to_save_per_step = int(embeddings_to_save_per_epoch/steps_per_epoch)

    manager = multiprocessing.Manager()
    queue = manager.Queue()
    consumer = multiprocessing.Process(target=reporting, args=(args.output, queue), name="Reporting")
    
    queue.put(("add_graph", (cae, next(iter(loader_aug)))))
    if args.cuda:
        cae.cuda()
    opt = torch.optim.Adam(cae.parameters())
    
    # def activation_hook(self, input, output):
    #     global global_step

    #     if global_step % args.batch_report_frequency == 0:
    #         queue.put(("add_histogram", ("activations/%s" % module_map[id(self)], torch.tensor(output).cpu(), global_step)))
    # module_map = {}
    # for name, module in cae.named_modules():
    #     if len([_ for _ in module.children()]) == 0:
    #         module_map[id(module)] = name
    #         module.register_forward_hook(activation_hook)

    try:
        consumer.start()
        c_process = psutil.Process(consumer.pid)
        this_process = psutil.Process()

        with torch.autograd.detect_anomaly():
            embeddings = torch.empty((embeddings_to_save_per_epoch, args.embedding_size), dtype=np.float)
            label_imgs = torch.empty(tuple([embeddings_to_save_per_epoch] + list(img_shape)), dtype=np.float)
            
            for epoch in tqdm(range(args.epochs)):
                for b_i, batch in enumerate(tqdm(loader_aug, leave=False)):
                    global_step += 1
                    opt.zero_grad()

                    if args.cuda:
                        batch = batch.cuda()

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

                    queue.put(("add_scalar", ("memory/consumer", c_process.memory_info().rss, global_step)))
                    queue.put(("add_scalar", ("memory/this", this_process.memory_info().rss, global_step)))
                    queue.put(("add_scalar", ("memory/all", psutil.virtual_memory().used, global_step)))

                queue.put(("add_embedding", (embeddings, None, torch.unsqueeze(label_imgs[:, 0], 1), global_step)))
                ig = batch[:15].detach().cpu()
                og = target[:15].detach().cpu()
                for i in range(len(args.channels)):
                    input_grid = utils.make_grid(torch.unsqueeze(ig[:, i, ...], 1), nrow=3, normalize=True)
                    output_grid = utils.make_grid(torch.unsqueeze(og[:, i, ...], 1), nrow=3, normalize=True)
                    queue.put(("add_image", ("training/input.%d" % i, input_grid, global_step)))
                    queue.put(("add_image", ("training/output.%d" % i, output_grid, global_step)))
                queue.put(("add_scalar", ("training/loss", running_loss.avg.cpu(), global_step)))
                for n, rg in running_gradients.items():
                    queue.put(("add_histogram", ("gradients/%s" % n, rg.avg.cpu(), global_step)))
                    rg.reset()                
                running_loss.reset()

            torch.save(cae.state_dict(), args.output / "model.pth")
    finally:
        queue.put(None)
        consumer.join()
