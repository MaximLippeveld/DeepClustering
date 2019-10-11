"""
Implementation based on:

https://xifengguo.github.io/papers/ICONIP17-DCEC.pdf
http://openaccess.thecvf.com/content_ICCV_2017/papers/Dizaji_Deep_Clustering_via_ICCV_2017_paper.pdf
"""

import torch
import torch.nn as nn
import numpy
from math import ceil, floor


def conv_output_size(input_size, pad, ks, stride):
    return floor((float(input_size + 2 * pad - (ks-1)-1)/stride)+1)
def convtranspose_output_size(input_size, pad, ks, stride):
    return (input_size-1)*stride - 2*pad + ks


class ConvolutionalEncoder(nn.Module):

    """Convolutional encoder
    """

    def __init__(self, input_shape, embedding_shape, dropout=-1.):
        """Initialize encoder layers.
        
        Arguments:
            input_shape {int} -- Flattened input shapa
            embedding_shape {int} -- Embedding dimension
            dropout {float} -- Dropout probability or <= 0.0 if not wanted
        """
        super(ConvolutionalEncoder, self).__init__()

        filters = [32, 16]
        stride = 2
        padding = 1
        kernel_size = 6 if input_shape[1] == 90 else 4

        a = conv_output_size(
                conv_output_size(
                    input_shape[1], padding, kernel_size, stride
                ), padding, kernel_size, stride
            )
        embedding_input_shape = a*a*filters[-1]

        self.layers = nn.ModuleList()
        self.layers.extend([
            nn.Conv2d(input_shape[0], filters[0], stride=stride, kernel_size=kernel_size, padding=padding),
            nn.LeakyReLU(),
            nn.Conv2d(filters[0], filters[1], stride=stride, kernel_size=kernel_size, padding=padding),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(embedding_input_shape, embedding_shape),
            nn.LeakyReLU()
        ])

        if dropout > 0:
            self.layers.insert(2, nn.Dropout(dropout))
            self.layers.insert(5, nn.Dropout(dropout))


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ConvolutionalDecoder(nn.Module):
    """Convolutional decoder
    """

    def __init__(self, input_shape, reconstruction_shape):
        """[summary]

        Arguments:
            input_shape {tuple} -- [description]
            reconstruction_shape {tuple} -- [description]
        """
        super(ConvolutionalDecoder, self).__init__()
        
        filters = [16, 32]
        stride = 2
        padding = 1
        kernel_size = 6 if reconstruction_shape[1] == 90 else 4
        
        a = conv_output_size(
                conv_output_size(
                    reconstruction_shape[1], padding, kernel_size, stride
                ), padding, kernel_size, stride
            )
        embedding_input_shape = a*a*filters[-1]

        self.shape = (
            filters[0],
            a,
            a
        )
        
        self.linear = nn.Linear(input_shape, numpy.prod(self.shape))
        self.nonlinearity = nn.LeakyReLU()
        self.layers = nn.ModuleList()

        self.layers.extend([
            nn.ConvTranspose2d(filters[0], filters[1], stride=stride, padding=padding, kernel_size=kernel_size),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(filters[1], reconstruction_shape[0], stride=stride, padding=padding, kernel_size=kernel_size),
            nn.LeakyReLU()
        ])


    def forward(self, x):

        x = self.linear(x)
        x = self.nonlinearity(x)
        x = torch.reshape(x, [-1] + list(self.shape))

        for layer in self.layers:
            x = layer(x)
        return x

class ConvolutionalAutoEncoder(nn.Module):

    def __init__(self, input_shape, embedding_shape, dropout):

        """Denoising auto-encoder.
        
        Arguments:
            input_shape {int} -- Input shape
            embedding_shape {int} -- Embedding dimension
            dropout {float} -- [description]
        
        """
        super(ConvolutionalAutoEncoder, self).__init__()
                
        self.encoder = ConvolutionalEncoder(input_shape, embedding_shape, dropout)
        self.decoder = ConvolutionalDecoder(embedding_shape, input_shape)

    def forward(self, x):
        embedding = self.encoder(x)
        return self.decoder(embedding)
