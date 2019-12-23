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
import lmdb
import numpy
import pickle


def writetolmdb(output, queue, length):
    db = lmdb.open(output)
    count = 0
    idx_byte_length = int(numpy.ceil(numpy.floor(numpy.log2(length))/8.))

    with db.begin(write=True) as txn:
        while True:
            item = queue.get()
            if item == None:
                break

            txn.put(
                count.to_bytes(idx_byte_length, "big"), 
                pickle.dumps(item)
            )
            count += 1


def main(args):

    # logger
    logger = logging.getLogger()

    # data prep
    loader_aug = data.loaders.DataLoaderWrapper(args)

    # get model, optimizer, loss
    cae = model.cae.ConvolutionalAutoEncoder(loader_aug.img_shape, args.embedding_size, args.dropout)
    cae.load_state_dict(torch.load(args.model))
    logger.info("Trainable model parameters: %d" % sum(p.numel() for p in cae.parameters() if p.requires_grad))

    global_step = 0

    manager = multiprocessing.Manager()
    queue = manager.Queue()
    consumer = multiprocessing.Process(target=writetolmdb, args=(args.output, queue, len(loader_aug.ds)), name="Reporting")
    
    if args.cuda:
        cae.cuda()
    
    try:
        consumer.start()
        c_process = psutil.Process(consumer.pid)
        this_process = psutil.Process()

        with torch.autograd.detect_anomaly():
            for b_i, batch in enumerate(tqdm(loader_aug, leave=False)):
                global_step += 1

                if args.cuda:
                    batch = batch.cuda()

                embedding = cae.encoder(batch)
                queue.put(embedding.cpu())

    finally:
        queue.put(None)
        consumer.join()
