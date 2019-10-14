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

def epoch_reporting(output, queue, n_channels):
    logger = logging.getLogger("epoch_reporting")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    handler = logging.FileHandler("epoch_reporting.log", mode="w")
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    writer = SummaryWriter(output / "tb/epoch")
    process = psutil.Process(os.getpid())

    while True:
        item = queue.get()
        logger.debug("%d - %d" % (os.getpid(), process.memory_info().rss))
        if item == None:
            break

        global_step = item["global_step"]

        for i in range(n_channels):
            input_grid = utils.make_grid(torch.unsqueeze(item["input_grid"][:, i, ...], 1), nrow=3, normalize=True)
            output_grid = utils.make_grid(torch.unsqueeze(item["output_grid"][:, i, ...], 1), nrow=3, normalize=True)
            
            writer.add_image("training/input.%d" % i, input_grid, global_step=global_step)
            writer.add_image("training/output.%d" % i, output_grid, global_step=global_step)
        del item["input_grid"]
        del item["output_grid"]

        writer.add_embedding(item["embeddings"], label_img=item["label_imgs"], global_step=global_step)
        del item["embeddings"]
        del item["label_imgs"]

        writer.add_scalar("training/loss", item["running_loss_avg"], global_step=global_step)
        del item["running_loss_avg"]

        for n, avg in item["running_gradients_avgs"].items():
            writer.add_histogram("gradients/%s" % n, avg, global_step=global_step)
        del item["running_gradients_avgs"]

        logger.debug("finish")

    writer.close()


def batch_reporting(output, queue):
    process = psutil.Process(os.getpid())
    logger = logging.getLogger("batch_reporting")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    handler = logging.FileHandler("batch_reporting.log", mode="w")
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    
    writer = SummaryWriter(output / "tb/batch")
    
    while True:
        item = queue.get()
        logger.debug("%d - %d" % (os.getpid(), process.memory_info().rss))
        if item == None:
            break

        global_step = item["global_step"]

        writer.add_histogram("activations/%s" % item["name"], item["output"], global_step=global_step)
        del item["output"]

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

    writer = SummaryWriter(args.output / "tb/root")

    # prepare data
    if isinstance(args.data, Path):
        import lmdb

        env = lmdb.open(str(args.data), subdir=args.data.is_dir(),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with env.begin(write=False) as txn:
            length = int.from_bytes(txn.get(b'__len__'), "big")

        ds = data.sets.LMDBDataset(str(args.data), args.channels, 90, length, None)
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
    if args.cuda:
        cae.cuda()
    # writer.add_graph(cae, next(iter(loader_aug)))
    opt = torch.optim.Adam(cae.parameters())

    logger.info("Trainable model parameters: %d" % sum(p.numel() for p in cae.parameters() if p.requires_grad))

    # training loop
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
    consumer = multiprocessing.Process(target=epoch_reporting, args=(args.output, queue, len(args.channels)), name="Epoch reporting")
    batch_queue = manager.Queue()
    batch_consumer = multiprocessing.Process(target=batch_reporting, args=(args.output, batch_queue), name="Batch reporting")
    
    process = psutil.Process(os.getpid())

    module_map = {}
    def activation_hook(self, input, output):
        global global_step
        if global_step % args.batch_report_frequency == 0:
            item = {
                "name": module_map[id(self)],
                "global_step": global_step,
                "output": output.clone().detach().cpu()
            }
            batch_queue.put(item)

    for name, module in cae.named_modules():
        if len([_ for _ in module.children()]) == 0:
            module_map[id(module)] = name
            module.register_forward_hook(activation_hook)

    try:
        consumer.start()
        batch_consumer.start()

        c_process = psutil.Process(consumer.pid)
        bc_process = psutil.Process(batch_consumer.pid)
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

                    writer.add_scalar("memory/consumer", c_process.memory_info().rss, global_step) 
                    writer.add_scalar("memory/batch_consumer", bc_process.memory_info().rss, global_step) 
                    writer.add_scalar("memory/this", this_process.memory_info().rss, global_step) 

                item = {
                    "input_grid": batch[:15].clone().detach().cpu(),
                    "output_grid": target[:15].clone().detach().cpu(),
                    "embeddings": embeddings.clone(),
                    "label_imgs": torch.unsqueeze(label_imgs[:, 0], 1).clone(),
                    "running_loss_avg": running_loss.avg.clone().cpu(),
                    "running_gradients_avgs": {},
                    "global_step": global_step
                }
                for n, rg in running_gradients.items():
                    item["running_gradients_avgs"][n] = rg.avg.clone().cpu()

                queue.put(item)
                
                running_loss.reset()
                for k, v in running_gradients.items():
                    v.reset()

            torch.save(cae.state_dict(), args.output / "model.pth")
    finally:
        queue.put(None)
        batch_queue.put(None)
        consumer.join()
        batch_consumer.join()
