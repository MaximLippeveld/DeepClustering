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

multiprocessing.set_start_method('spawn', True)

def epoch_reporting(output, queue, event):
    logger = logging.getLogger("epoch_reporting")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    handler = logging.FileHandler("epoch_reporting.log", mode="w")
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

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

        logger.debug("finish")

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
    logging.getLogger().setLevel(logging.DEBUG)

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

    logging.info("Trainable model parameters: %d" % sum(p.numel() for p in cae.parameters() if p.requires_grad))

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
    event = manager.Event()
    consumer = multiprocessing.Process(target=epoch_reporting, args=(args.output, queue, event), name="Epoch reporting")

    # register hooks to record activations
    writer = SummaryWriter(args.output / "tb")
    hooks = {}
    module_map = {}
    def activation_hook(self, input, output):
        global global_step
        writer.add_histogram("activations/%s" % module_map[id(self)], output, global_step=global_step)
    for name, module in cae.named_modules():
        if len([_ for _ in module.children()]) == 0:
            module_map[id(module)] = name
            hooks[name] = module.register_forward_hook(activation_hook)

    try:
        consumer.start()

        with torch.autograd.detect_anomaly():
            for epoch in tqdm(range(args.epochs)):
                # iterate over data

                embeddings = torch.empty((embeddings_to_save_per_epoch, args.embedding_size), dtype=np.float)
                label_imgs = torch.empty(tuple([embeddings_to_save_per_epoch] + list(img_shape)), dtype=np.float)
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

                # reporting
                item = {
                    "input_grid": batch[:15].clone().detach().cpu(),
                    "output_grid": target[:15].clone().detach().cpu(),
                    "embeddings": embeddings.clone(),
                    "label_imgs": label_imgs.clone(),
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
        event.set()
        consumer.join()
