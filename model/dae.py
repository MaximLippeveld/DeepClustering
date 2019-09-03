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
        super(Encoder, self).__init__()

        self.layers = nn.ModuleList()
        
        shapes = [
            input_shape,
            500, 500, 2000,
            embedding_shape
        ]

        for i, in_shape in enumerate(shapes[:-1]):
            self.layers.append(nn.Linear(in_shape, shapes[i+1]))

            if i != (len(shapes)-1):
                self.layers.append(nn.ReLU())
                self.layers.append(nn.Dropout(p=dropout))


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


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

        self.layers = nn.ModuleList()
        for i, in_shape in enumerate(shapes[:-1]):
            self.layers.append(nn.Linear(in_shape, shapes[i+1]))

            if i != (len(shapes)-1):
                self.layers.append(nn.ReLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DAE(nn.Module):
    
    def __init__(self, input_shape, embedding_shape, dropout=0.5):
        """Denoising auto-encoder.
        
        Arguments:
            input_shape {int} -- Input shape
            embedding_shape {int} -- Embedding dimension
        
        Keyword Arguments:
            dropout {float} -- Probability of dropping random element (default: {0.5})
        """
        super(DAE, self).__init__()
        
        if type(input_shape) is tuple:
            self.flatten = nn.Flatten()
            input_shape = input_shape[0]*input_shape[1]
        
        self.encoder = Encoder(input_shape, embedding_shape, dropout)
        self.decoder = Decoder(embedding_shape, input_shape)

    def forward(self, x):
        if len(x.shape) > 2:
            x = self.flatten(x)

        embedding = self.encoder(x)
        return self.decoder(embedding)
