import torch.nn as nn
import torch.nn.functional as F


class RNNAutoencoder(nn.Module):
    """
    Class that defines the RNN-Autoencoder model architecture.
    """

    def __init__(self, input_dim, layers, latent_dim, losses, activation):
        """Initializes the network class

        Args:
            input_dim (int): The number of inputs of the system. 
            layers (list, optional): The dimensions of fully-connected layers for the encoder and decoder. Defaults to [128, 64, 32].
            latent_dim (int, optional): The dimension of the bottleneck for generating the latent variable. Defaults to 1.
            losses (list, optional): The list of losses to use.
            activation (str, optional): The name of the activation function. Defaults to 'relu'.
        """
        
        super(RNNAutoencoder, self).__init__()

        # Save the loss functions
        self.losses = list(losses)

        # Define the activation function
        activations = {'relu': F.relu, 'tanh': F.tanh, 'sigmoid': F.sigmoid}
        self.activation = activations[activation]

        # Build the encoder layer
        self.encoder = nn.ModuleList(
            [nn.Linear(input_dim, layers[0])] + 
            [nn.Linear(layers[i-1], layers[i]) for i in range(1, len(layers))] + 
            [nn.Linear(layers[-1], latent_dim)]
        )

        # Build the RNN layer
        self.rnn = nn.LSTM(
            latent_dim,         # input size 
            latent_dim,         # output size
            num_layers=1, 
            batch_first=True, 
            dropout=0.0, 
            bidirectional=False
        )

        # Build the decoder layer
        self.decoder = nn.ModuleList(
            [nn.Linear(latent_dim, layers[-1])] +
            [nn.Linear(layers[i], layers[i-1]) for i in range(len(layers)-1, 0, -1)] +
            [nn.Linear(layers[0], input_dim)]
        )

    def forward(self, batch):
        """
        The batch to be processed by the network
        """
        
        # Encode the input to bottleneck layer
        x = batch
        for layer in self.encoder:
            x = self.activation(layer(x))

        # Pass the bottleneck layer through the RNN
        x, _ = self.rnn(x)

        # Decode the input
        for layer in self.decoder:
            x = self.activation(layer(x))

        return x
    
    def compute_loss(self, batch, output):
        """
        Computes the loss function for the network
        """

        # Compute the loss
        loss = 0

        for loss_fn, weight in self.losses:
            loss += weight * loss_fn(batch, output)

        return loss