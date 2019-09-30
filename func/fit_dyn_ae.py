import numpy as np 
from sklearn.cluster import MiniBatchKMeans, KMeans
import torch
import data.sets, data.transformers
import model.cae, model.dae
import torch.nn.functional as F
import torch.optim
import torch.nn
from tqdm import tqdm
from torchvision import datasets, transforms, utils
from pathlib import Path
from util.augmentation.augmentation_2d import *
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import util.metrics
import logging
logging.basicConfig(level=logging.DEBUG)


def dynamic_reconstruction_loss(x, z, x_hat, c, s, centroids, reconstructed_centroids, model):
    ret = torch.nn.functional.mse_loss(x[~c], x_hat[~c])

    if torch.sum(c) > 0:
        sigma = torch.tensor(model.predict(z[c].detach().numpy()), dtype=torch.long)
        cc = torch.from_numpy(model.cluster_centers_)
        ret += (
            torch.nn.functional.mse_loss(reconstructed_centroids[sigma], x_hat[c]) + 
            torch.nn.functional.mse_loss(z[c], cc[sigma])
        )

    return ret

def q(z, u, alpha):
    q_ij = 1.0 / (1.0 + (np.sum(np.square(np.expand_dims(z, axis=1) - u), axis=2) / alpha))
    q_ij = np.power(q_ij, (alpha + 1.0) / 2.0)
    return np.transpose(np.transpose(q_ij) / np.sum(q_ij, axis=1))

def h(n, q_ij):
    if n == 1:
        return np.max(q_ij, axis=1)
    else:
        return np.sort(q_ij, axis=1)[:, -n]

def is_conflicted(q_ij, beta1, beta2):
    hi_1 = h(1, q_ij)
    hi_2 = h(2, q_ij)

    return np.logical_and(hi_1 >= beta1, (hi_1 - hi_2) >= beta2)

def is_conflicted_clustering(q_ij, beta1, beta2):
    hi_1 = h(1, q_ij)
    hi_2 = h(2, q_ij)
    hi_3 = h(3, q_ij)

    return np.logical_and(hi_1 >= beta1, (hi_2 - hi_3) >= beta2)

def main(args):

    # hyperparams
    k = 0.3*args.clusters
    delta_k = 0.3*k
    beta1 = k/args.clusters
    beta2 = beta1/2
    alpha = 1.
    
    # load data
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

    # initialize pretrained model
    m = model.cae.ConvolutionalAutoEncoder(img_shape, args.embedding_size)
    m.load_state_dict(torch.load(args.pretrained_model))

    if args.cuda:
        m.cuda()

    # initialize embedded centroids
    embedding = []
    hook = m.encoder.layers[-1].register_forward_hook(lambda s, i, o: embedding.extend(o.cpu().data.numpy()))
    m.eval()
    for i, batch in enumerate(loader_aug):
        if i == 20:
            break
        m.encoder(batch)
    embedding = np.array(embedding)
    hook.remove()

    centroids = KMeans(n_clusters=args.clusters).fit(embedding).cluster_centers_
    mb_kmeans = MiniBatchKMeans(n_clusters=args.clusters)
    mb_kmeans.cluster_centers_ = centroids
    reconstructed_centroids = m.decoder(torch.from_numpy(centroids).cuda()).cpu()

    # initialize loss
    loss = dynamic_reconstruction_loss

    # initalize optimizer
    opt = torch.optim.Adam(m.parameters())

    n_conf_prev = args.batch_size
    stop = False
    epoch = 0
    it = iter(loader_aug)

    m.train()
    while not stop and epoch < args.epochs:
        logging.info("Epoch: %d", epoch)

        for batch in loader_aug:
            if epoch > 0:
                n_conf_prev = nb_conf

            embedding = m.encoder(batch)
            reconstruction = m.decoder(embedding)

            q_ij = q(embedding.cpu().detach().numpy(), centroids, alpha)
            c_i = is_conflicted(q_ij, beta1, beta2)
            nb_conf = np.sum(c_i)
            if nb_conf >= n_conf_prev:
                mb_kmeans = mb_kmeans.partial_fit(embedding.detach().cpu().numpy()[c_i])
                centroids = mb_kmeans.cluster_centers_
                reconstructed_centroids = m.decoder(torch.from_numpy(centroids)).cpu()

                beta1 -= delta_k/args.clusters
                beta2 -= delta_k/args.clusters

            if nb_conf/args.batch_size < args.tolerance:
                logging.info("Stopping after %d epochs" % epoch)
                stop = True
                break

            s_i = is_conflicted_clustering(q_ij, beta1, beta2)
            l = loss(
                batch.cpu(), embedding.cpu(), reconstruction.cpu(), torch.from_numpy(c_i),
                torch.from_numpy(s_i), torch.from_numpy(centroids), 
                reconstructed_centroids, mb_kmeans
            )
            l.backward(retain_graph=True)

            opt.step()
        epoch += 1

