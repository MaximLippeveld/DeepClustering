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
from torch import autograd
from pathlib import Path
from util.augmentation.augmentation_2d import *
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import util.metrics
import logging

from matplotlib import pyplot as plt
import seaborn as sns
import umap


def inspect_clustering(batch, centroids, cluster_assignments):
    embedder = umap.UMAP().fit(batch)
    embedded_centroids = embedder.transform(centroids)

    fig, ax = plt.subplots()
    ax.scatter(embedder.embedding_[:, 0], embedder.embedding_[:, 1], s=5, c=cluster_assignments, cmap="tab10")
    ax.scatter(embedded_centroids[:, 0], embedded_centroids[:, 1], s=30, c=np.arange(centroids.shape[0]), cmap="tab10")

    return fig


def dynamic_reconstruction_loss(x, z, x_hat, c, s, centroids, reconstructed_centroids, model):
    m = 0
    n = 0
    o = 0

    if torch.sum(~c) > 0:
        m = torch.nn.functional.mse_loss(x[~c], x_hat[~c])
    if torch.sum(c) > 0:
        sigma_c = torch.tensor(model.predict(z[c].detach().numpy()), dtype=torch.long)
        n = torch.nn.functional.mse_loss(reconstructed_centroids[sigma_c], x_hat[c]) 
    if torch.sum(s) > 0:
        sigma_s = torch.tensor(model.predict(z[s].detach().numpy()), dtype=torch.long)
        cc = torch.from_numpy(model.cluster_centers_)
        o = torch.nn.functional.mse_loss(z[s], cc[sigma_s])

    return m + n + o

def q(z, u, alpha):
    q_ij = 1.0 / (1.0 + (np.sum(np.square(np.expand_dims(z, axis=1) - u), axis=2) / alpha))
    q_ij = np.power(q_ij, (alpha + 1.0) / 2.0)
    return np.transpose(np.transpose(q_ij) / np.sum(q_ij, axis=1))

def h(n, q_ij):
    if n == 1:
        return np.max(q_ij, axis=1)
    else:
        return np.sort(q_ij, axis=1)[:, -n]

def is_not_conflicted(q_ij, beta1, beta2):
    hi_1 = h(1, q_ij)
    hi_2 = h(2, q_ij)

    return np.logical_and(hi_1 >= beta1, (hi_1 - hi_2) >= beta2)

def is_not_conflicted_clustering(q_ij, beta1, beta2):
    hi_1 = h(1, q_ij)
    hi_2 = h(2, q_ij)
    hi_3 = h(3, q_ij)

    return np.logical_and(hi_1 >= beta1, (hi_2 - hi_3) >= beta2)

def main(args):

    # hyperparams
    k = 0.3*args.clusters
    # delta_k = 0.3*k
    delta_k = 0.001
    # beta1 = k/args.clusters
    beta1 = 0.095
    beta2 = beta1/2
    alpha = 1.
    
    # tensorboard writer
    writer = SummaryWriter(args.output / "tb")
    
    writer.add_text("Hyperparams", str({
        "k": k, "delta_k": delta_k, "beta1": beta1,
        "beta2": beta2, "alpha": alpha}))

    # load data
    class unsqueeze:
        def __init__(self, axis=0):
            self.axis = axis
        def __call__(self, x):
            return np.expand_dims(x, self.axis)

    ds = datasets.FashionMNIST("data/datasets/", train=True, download=True, transform=unsqueeze())
    channel_shape = ds.data.shape[1:]
    img_shape = [1] + list(ds.data.shape[1:])
    pre_augs = [ToTensor(cuda=args.cuda), ToFloatTensor(cuda=args.cuda), data.transformers.MinMax()]
    post_augs = []

    augs = pre_augs + [
        FlipX(channel_shape, cuda=args.cuda),
        FlipY(channel_shape, cuda=args.cuda),
        RandomDeformation(channel_shape, sampling_interval=7, cuda=args.cuda),
        RotateRandom(channel_shape, cuda=args.cuda)
    ] + post_augs

    composition = transforms.Compose(augs)
    def augmenter(x):
        x, y = zip(*x)
        return composition(x), torch.from_numpy(np.array(y))
    
    loader_aug = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, 
        drop_last=False, num_workers=0,
        collate_fn=augmenter
    )

    # initialize pretrained model
    m = model.cae.ConvolutionalAutoEncoder(img_shape, args.embedding_size)
    m.load_state_dict(torch.load(args.pretrained_model))

    if args.cuda:
        m.cuda()

    # initialize embedded centroids
    with autograd.detect_anomaly():
        mb_kmeans = MiniBatchKMeans(n_clusters=args.clusters)
        hook = m.encoder.layers[-1].register_forward_hook(lambda s, i, o: mb_kmeans.partial_fit(o.cpu().data.numpy()))
        m.eval()
        for i, (batch, target) in enumerate(tqdm(loader_aug, total=int(len(ds)/args.batch_size))):
            m.encoder(batch)
        hook.remove()
        reconstructed_centroids = m.decoder(torch.from_numpy(mb_kmeans.cluster_centers_).cuda()).cpu()

    # initialize loss
    loss = dynamic_reconstruction_loss

    # initalize optimizer
    opt = torch.optim.SGD(m.parameters(), lr=0.001, momentum=0.9)

    n_conf_prev = args.batch_size
    stop = False
    epoch = 0
    global_step = 0
    it = iter(loader_aug)

    writer.add_scalar("hyperparams/beta1", beta1, global_step)
    writer.add_scalar("hyperparams/beta2", beta2, global_step)
    
    m.train()
    with autograd.detect_anomaly():
        while not stop and epoch < args.epochs:
            logging.info("Epoch: %d", epoch)

            for i, (batch, target) in enumerate(loader_aug):
                opt.zero_grad()

                logging.info("Batch %d" % i)
                if global_step > 0:
                    n_conf_prev = nb_conf

                embedding = m.encoder(batch)
                reconstruction = m.decoder(embedding)

                embedding_ = embedding.cpu().detach().numpy()
                q_ij = q(embedding_, mb_kmeans.cluster_centers_, alpha)
                cluster_assignments = np.argmax(q_ij, axis=1)

                writer.add_histogram("q_ij_hist", q_ij, global_step)

                ret = sns.clustermap(q_ij, row_cluster=True, col_cluster=False)
                writer.add_figure("q_ij_map", ret.fig, global_step=global_step)

                fig, ax = plt.subplots()
                sns.countplot(cluster_assignments, ax=ax)
                writer.add_figure("Assignments", fig, global_step)

                clustering_fig = inspect_clustering(embedding_, mb_kmeans.cluster_centers_, cluster_assignments)
                writer.add_figure("Clustering", clustering_fig, global_step)
                plt.show()

                writer.add_scalar("ACC", util.metrics.acc(target.numpy(), cluster_assignments), global_step)
                writer.add_scalar("NMI", util.metrics.nmi(target.numpy(), cluster_assignments, average_method="arithmetic"), global_step)

                c_i = is_not_conflicted(q_ij, beta1, beta2)
                nb_conf = np.sum(~c_i)
                writer.add_scalar("nb_conf", nb_conf, global_step)

                if nb_conf >= n_conf_prev:
                    mb_kmeans = mb_kmeans.partial_fit(embedding_)
                    centroids = mb_kmeans.cluster_centers_
                    reconstructed_centroids = m.decoder(torch.from_numpy(mb_kmeans.cluster_centers_).cuda()).cpu()

                    beta1 -= delta_k/args.clusters
                    beta2 -= delta_k/args.clusters

                    writer.add_scalar("hyperparams/beta1", beta1, global_step)
                    writer.add_scalar("hyperparams/beta2", beta2, global_step)

                if nb_conf/args.batch_size < args.tolerance:
                    logging.info("Stopping after %d epochs" % epoch)
                    stop = True
                    break

                s_i = is_not_conflicted_clustering(q_ij, beta1, beta2)
                l = loss(
                    batch.cpu(), embedding.cpu(), reconstruction.cpu(), torch.from_numpy(c_i),
                    torch.from_numpy(s_i), torch.from_numpy(mb_kmeans.cluster_centers_), 
                    reconstructed_centroids, mb_kmeans
                )

                writer.add_scalar("loss", l, global_step=global_step)
                
                l.backward(retain_graph=True)

                opt.step()
                global_step += 1

            epoch += 1

    writer.close()
