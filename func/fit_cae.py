import model.cae
import torch.nn.functional as F
import torch.optim, torch.autograd
from tqdm import tqdm
from torchvision import utils
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import util.metrics, util.reporting
import logging
from math import ceil
import time
from torch import multiprocessing
import os
import psutil
from collections.abc import Iterable
import numpy as np
import data.loaders
import logging


def main(args):

    # logger
    logger = logging.getLogger()

    # data prep
    loader_aug = data.loaders.DataLoaderWrapper(args)

    # get model, optimizer, loss
    cae = model.cae.ConvolutionalAutoEncoder(loader_aug.img_shape, args.embedding_size, args.dropout)
    logger.info("Trainable model parameters: %d" % sum(p.numel() for p in cae.parameters() if p.requires_grad))

    running_loss = util.metrics.AverageMeter("loss", (1,), cuda=args.cuda)
    running_gradients = {
        n: util.metrics.AverageMeter(n, cuda=args.cuda)
        for n,p in cae.named_parameters() if p.requires_grad
    }

    global global_step
    global_step = 0
    embeddings_to_save_per_epoch = 5000
    steps_per_epoch = ceil(len(loader_aug.ds)/args.batch_size)
    embeddings_to_save_per_step = int(embeddings_to_save_per_epoch/steps_per_epoch)

    manager = multiprocessing.Manager()
    queue = manager.Queue()
    consumer = multiprocessing.Process(target=util.reporting.target, args=(args.output, queue), name="Reporting")
    
    queue.put(("add_graph", (cae, next(iter(loader_aug)))))
    if args.cuda:
        cae.cuda()
    opt = torch.optim.Adam(cae.parameters())
    
    try:
        consumer.start()
        c_process = psutil.Process(consumer.pid)
        this_process = psutil.Process()

        with torch.autograd.detect_anomaly():
            embeddings = torch.empty((embeddings_to_save_per_epoch, args.embedding_size), dtype=np.float)
            label_imgs = torch.empty(tuple([embeddings_to_save_per_epoch] + list(loader_aug.img_shape)), dtype=np.float)
            
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
