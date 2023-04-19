import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNAutoencoder(nn.Module):
    """
    Class that defines the RNN-Autoencoder model architecture.
    """

    def __init__(self, input_dim, output_dim, layers, latent_dim, activation, system_model, pooled_classification=True, device="cpu"):
        """Initializes the network class

        Args:
            input_dim (int): The number of inputs of the system. 
            output_dim (int): The number of outputs of the system.
            layers (list, optional): The dimensions of fully-connected layers for the encoder and decoder. Defaults to [128, 64, 32].
            latent_dim (int, optional): The dimension of the bottleneck for generating the latent variable. Defaults to 1.
            system_model (fn): A function which encodes the state-space equations that partially describe the motion of the vehicle + payload
            pooled_classification (bool, optional): Whether to use the pooled classification or not, i.e. we only care about the last prediction of the network for the loss, given a set of inputs. Defaults to True.
            regularization (dict): A dictionary with the regularization constants for the loss function
            activation (str, optional): The name of the activation function. Defaults to 'relu'.


        Note: the input of the network should be such that at timestep k
            [x,y,z | vx,vy,vz | qx,qy,qz,qw | wx,wy,wz,T | px,py,pz]
            # position of the drone (in the inertial frame)
            # velocity of the drone (in the inertial frame)
            # attitude of the drone
            # angular velocity of the drone and thrust, expressed in the drone's body frame
            # position of the payload (in the inertial frame)
            Total input size=17

            the output of the network should be the next state at k+1
            [x,y,z | vx,vy,vz | qx,qy,qz,qw | px,py,pz]
            Total output size=13
        """
        
        super(RNNAutoencoder, self).__init__()

        # Set the device to run the network train and inference
        self.device = device

        # Whether we use the pooled classification or not (i.e. we only care about the last prediction of the network for the loss, given a set of inputs)
        self.pooled_classification = pooled_classification

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
    
    def compute_loss(self, x: torch.Tensor, y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss function for the network

        Args:
            x (torch.Tensor): The input of the network             [x,y,z | vx,vy,vz | qx,qy,qz,qw | wx,wy,wz,T | px,py,pz]
            y (torch.Tensor): The expected output of the network   [x,y,z | vx,vy,vz | qx,qy,qz,qw | px,py,pz]
            y_hat (torch.Tensor): The output of the network        [x,y,z | vx,vy,vz | qx,qy,qz,qw | px,py,pz]
        """

        # If we are predicting only the last state of the sequence, given multiple previous steps of the sequence to the network
        if self.pooled_classification:

            # We only care about the last prediction of the network for the loss, given a set of inputs
            y_hat = y_hat[:, -1, :]
            y_hat = y_hat.unsqueeze(1)

            # We only need the last state given as input to the network, to compute the next state based on a physical law
            x = x[:, -1, :]
            x = x.unsqueeze(1)
        
        # Compute the MSE lost (fitting of the actual data)
        mse = F.mse_loss(y_hat, y)

        # Perform the prediction based on our physical model of the quadrotor
        # to predict [x,y,z | vx,vy,vz | qx,qy,qz,qw]
        physics_pred = self.system_model(x=x[..., 0:10], u=x[...,10:14])

        # Compute the MSE of the physics loss between 
        # the drone physical model and the prediction of the network
        physics_loss = F.mse_loss(physics_pred, y_hat[..., 0:10])

        # Note: the line bellow can be used to check the correctness of the physics model against real data
        # aux = F.mse_loss(physics_pred, y)

        # Quaternion should have norm 1, so we try to enforce that
        # constraint in the loss function
        quat_norm = y_hat[..., 6:10].norm(dim=-1)
        quaternion_norm_loss = F.mse_loss(torch.ones_like(quat_norm), quat_norm)

        # Compute the reconstruction loss
        # reconstruction_loss = 0.0

        # TODO - finish this section

        # Compute the mean-square-error loss
        return mse + physics_loss + quaternion_norm_loss