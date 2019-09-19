"""
Implementation based on:

https://xifengguo.github.io/papers/ICONIP17-DCEC.pdf
http://openaccess.thecvf.com/content_ICCV_2017/papers/Dizaji_Deep_Clustering_via_ICCV_2017_paper.pdf
"""

import torch
import torch.nn as nn

class ConvolutionalEncoder:

    """Convolutional encoder
    """

    def __init__(self, input_shape, embedding_shape):
        """Initialize encoder layers.
        
        Arguments:
            input_shape {int} -- Flattened input shape
            embedding_shape {int} -- Embedding dimension
        """
        super(ConvolutionalEncoder, self).__init__()

        filters = [32, 16]
        embedding_input_shape = ((input_shape[-1]//2)//2)*filters[-1]

        self.layers = nn.ModuleList()

        self.layers.extend([
            nn.Conv2d(input_shape[0], filters[0], stride=2, kernel_size=5),
            nn.LeakyReLu(),
            nn.Conv2d(filters[0], filters[1], stride=2, kernel_size=5),
            nn.LeakyReLu(),
            nn.Flatten(),
            nn.Linear(embedding_input_shape, embedding_shape),
            nn.Tanh()
        ])


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ConvolutionalDecoder(nn.Module):
    """Convolutional decoder
    """

    def __init__(self, input_shape, reconstruction_shape):
        super(ConvolutionalDecoder, self).__init__()
        
        filters = [16, 32]

        self.reconstruction_shape = reconstruction_shape
        shape = ((reconstruction_shape[-1]//2)//2)*filters[0]
        
        self.linear = nn.Linear(input_shape, shape)
        self.tanh = nn.Tanh()
        self.layers = nn.ModuleList()

        self.layers.extend([
            nn.ConvTranspose2d(filters[0], filters[1], stride=2, kernel_size=5),
            nn.LeakyReLu(),
            nn.ConvTranspose2d(filters[1], reconstruction_shape[0], stride=2, kernel_size=5),
            nn.LeakyReLu()
        ])


    def forward(self, x):

        x = self.linear(x)
        x = self.tanh(x)
        x = torch.reshape(x, self.reconstruction_shape)

        for layer in self.layers:
            x = layer(x)
        return x

class ConvolutionalAutoEncoder:

    def __init__(self, input_shape, embedding_shape):

        """Denoising auto-encoder.
        
        Arguments:
            input_shape {int} -- Input shape
            embedding_shape {int} -- Embedding dimension
        
        """
        super(ConvolutionalAutoEncoder, self).__init__()
                
        self.encoder = ConvolutionalEncoder(input_shape, embedding_shape)
        self.decoder = ConvolutionalDecoder(embedding_shape, input_shape)

    def forward(self, x):
        embedding = self.encoder(x)
        return self.decoder(embedding)
