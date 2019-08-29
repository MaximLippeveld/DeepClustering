""" Denoising auto-encoder architecture

"""

import torch.nn as nn

class Encoder(nn.Module):
    """Regular encoder with dropout possibility    
    """

    def __init__(self, input_shape, embedding_shape, dropout=False):
        """Initialize encoder layers.
        
        Arguments:
            input_shape {int} -- Flattened input shape
            embedding_shape {int} -- Embedding dimension
            dropout {float} -- Set the drop probability (default: no drops)
        """
        super(NoisyEncoder, self).__init__()

        shapes = [
            input_shape,
            500, 500, 2000,
            embedding_shape
        ]

        self.layers = []
        for i, in_shape in enumerate(shapes[:-1]):
            self.layers.append(nn.Linear(in_shape, shapes[i+1]))

            if i != (len(shapes)-1):
                self.layers.append(nn.ReLU())
                self.layers.append(nn.Dropout(p=dropout))


class Decoder(nn.Module):
    """Regular decoder
    """

    def __init__(self, input_shape, reconstruction_shape):
        super(Decoder, self).__init__()
        
        shapes = [
            input_shape,
            2000, 500, 500,
            reconstruction_shape
        ]

        self.layers = []
        for i, in_shape in enumerate(shapes[:-1]):
            self.layers.append(nn.Linear(in_shape, shapes[i+1]))

            if i != (len(shapes)-1):
                self.layers.append(nn.ReLU())



class DAE(nn.Module):
    
    def __init__(self, input_shape, embedding_shape, dropout=0.5):
        """Denoising auto-encoder.
        
        Arguments:
            input_shape {int} -- Flattened input shape
            embedding_shape {int} -- Embedding dimension
        
        Keyword Arguments:
            dropout {float} -- Probability of dropping random element (default: {0.5})
        """
        self.encoder = Encoder(input_shape, embedding_shape, dropout)
        self.decoder = Decoder(embedding_shape, input_shape)

    def forward(self, x):
        embedding = self.encoder(x)
        return self.decoder(embedding)
