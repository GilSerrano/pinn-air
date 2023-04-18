import torch.nn as nn
import torch.nn.functional as F

# The base implementation for the loss functions
from train.losses import mse_loss, system_loss

class RNNAutoencoder(nn.Module):
    """
    Class that defines the RNN-Autoencoder model architecture.
    """

    def __init__(self, input_dim, output_dim, layers, latent_dim, activation, system_model, device="cpu"):
        """Initializes the network class

        Args:
            input_dim (int): The number of inputs of the system. 
            output_dim (int): The number of outputs of the system.
            layers (list, optional): The dimensions of fully-connected layers for the encoder and decoder. Defaults to [128, 64, 32].
            latent_dim (int, optional): The dimension of the bottleneck for generating the latent variable. Defaults to 1.
            system_model (fn): A function which encodes the state-space equations that partially describe the motion of the vehicle + payload
            activation (str, optional): The name of the activation function. Defaults to 'relu'.
        """
        
        super(RNNAutoencoder, self).__init__()

        # Set the device to run the network train and inference
        self.device = device

        # Set the system model for the state-space equations
        self.system_model = system_model

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
            [nn.Linear(layers[0], output_dim)]
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
    
    def compute_loss(self, x, y, y_hat):
        """
        Computes the loss function for the network

        Args:
            x (torch.Tensor): The input of the network
            y (torch.Tensor): The expected output of the network
            y_hat (torch.Tensor): The output of the network
        """
        
        # Compute the MSE lost
        mse_loss = mse_loss(y, y_hat)

        # Compute the system_model loss
        physics_loss = system_loss()

        # Compute the reconstruction loss
        reconstruction_loss = 0.0

        # TODO - finish this section

        # Compute the mean-square-error loss
        return mse_loss + physics_loss