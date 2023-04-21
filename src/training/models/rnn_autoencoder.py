import torch
import torch.nn as nn
import torch.nn.functional as F

from dynamics.utils import quaternion_multiply, quaternion_invert

class RNNAutoencoder(nn.Module):
    """
    Class that defines the RNN-Autoencoder model architecture.
    """

    def __init__(self, input_dim, output_dim, layers, 
            latent_dim, 
            dropout,
            activation, 
            system_model, 
            pooled_classification=True, 
            regularization={
                "mse_drone_position": 0.6,
                "mse_drone_velocity": 0.3,
                "quaternion_norm": 0.3,
                "attitude_error": 0.6,
                "mse_payload_position": 0.6,
                "physical_model": {
                    "mse_position": 0.4,
                    "mse_velocity": 0.2,
                    "mse_attitude": 0.2
                }
            },
            device="cpu"):
        """Initializes the network class

        Args:
            input_dim (int): The number of inputs of the system. 
            output_dim (int): The number of outputs of the system.
            layers (list, optional): The dimensions of fully-connected layers for the encoder and decoder. Defaults to [128, 64, 32].
            latent_dim (int, optional): The dimension of the bottleneck for generating the latent variable. Defaults to 1.
            dropout (flaot, optional): The dropout rate to be used in the fully-connected layers.
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

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Whether we use the pooled classification or not (i.e. we only care about the last prediction of the network for the loss, given a set of inputs)
        self.pooled_classification = pooled_classification

        # The regularization constants to use in the loss function
        self.regularization = regularization

        # Set the system model for the state-space equations
        self.system_model = system_model

        # Set the dropout probability
        self.dropout = dropout

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
            x = F.dropout(self.activation(layer(x)), p=self.dropout, training=self.training)

        # Pass the bottleneck layer through the RNN
        x, _ = self.rnn(x)

        # Decode the input
        for layer in self.decoder:
            x = F.dropout(self.activation(layer(x)), p=self.dropout, training=self.training)

        # If we are doing pooled classification, we only care about the last prediction of the network
        if self.pooled_classification:
            x = x[:, -1, :].unsqueeze(1)

        return x
    
    def predict_series(self, x: torch.Tensor):

        # Predict the sequence given the input
        predicted_squence = self.forward(x)

    def predict_series_recursively(self, x: torch.Tensor, max_length=5):
        """Predict a sequence recursively using this network. We feed the output of the network to the input of the network at the next timestep.
        Input of the type: (batch, total_time, features)

        Returns:
            torch.Tensor: The predicted sequence
        """

        # Get the dimension of the time sequence of each input
        time_sequence_length = x.shape[1]
        print(time_sequence_length)

        self.eval()

        with torch.no_grad():

            # Initialize the predicted sequence
            predicted_squence = torch.zeros(x.shape[0], time_sequence_length, self.output_dim).to(self.device)

            # If we are dealing with pooled classification, we need to iterate over the sequence and predict one by one
            if self.pooled_classification:
                
                # Predict the first N timesteps from the real input data
                for i in range(max_length):
                    predicted_squence[:, i, :] = self.forward(x[:, i:i+max_length, :])

                # Predict the rest of the sequence using the output of the network as the input of the network
                for i in range(max_length, time_sequence_length-max_length):
                    # Note, we use the latest state prediction fromt the network with the true input data
                    predicted_squence[:, i, :] = self.forward(torch.cat((predicted_squence[:, i-max_length:i, :], x[:, i:i+max_length, 10:14]), dim=2))

            else:
                # Predict the sequence given the input
                for i in range(time_sequence_length):
                    predicted_squence[:, i, :] = self.forward(x[:, i:i+max_length, :])

        return predicted_squence
    
    
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

            # We only need the last state given as input to the network, to compute the next state based on a physical law
            x = x[:, -1, :]
            x = x.unsqueeze(1)

        # --------------------------------------------
        # Compute the fitting loss
        # --------------------------------------------
        
        # Compute the MSE lost (fitting of the actual data)
        mse_drone_position = F.mse_loss(y_hat[..., 0:3], y[..., 0:3])
        mse_drone_velocity = F.mse_loss(y_hat[..., 3:6], y[..., 3:6])
        mse_payload_position = F.mse_loss(y_hat[..., 10:13], y[..., 10:13])

        # Compute the rotation error from the quaternion => errror = (q x q_hat')
        quaternion_error = quaternion_multiply(y[..., 6:10], quaternion_invert(y_hat[..., 6:10]))
        mse_quat_error = torch.mean(quaternion_error ** 2)

        # --------------------------------------------
        # Compute the physics loss
        # --------------------------------------------

        # Perform the prediction based on our physical model of the quadrotor
        # to predict [x,y,z | vx,vy,vz | qx,qy,qz,qw] and compare with the prediction from the network
        physics_pred = self.system_model(x=x[..., 0:10], u=x[...,10:14])
        mse_physics_position = F.mse_loss(physics_pred[..., 0:3], y_hat[..., 0:3])
        mse_physics_velocity = F.mse_loss(physics_pred[..., 3:6], y_hat[..., 3:6])
        physics_quaternion_error = torch.mean(quaternion_multiply(quaternion_invert(physics_pred[..., 6:10]), y_hat[..., 6:10]) ** 2)

        # --------------------------------------------
        # Enforce the unit quaternion norm
        # --------------------------------------------

        # Quaternion should have norm 1, so we try to enforce that constraint in the loss function
        quat_norm = y_hat[..., 6:10].norm(dim=-1)
        quaternion_norm_loss = F.mse_loss(torch.ones_like(quat_norm), quat_norm)

        # --------------------------------------------
        # Extra - Compute the difference between the physics model and reality
        # --------------------------------------------

        # Note: the line bellow can be used to check the correctness of the physics model against real data
        # These terms are not used in the loss function, but can be used to monitor the performance of the physics model        
        real_model_position_loss = F.mse_loss(physics_pred[..., 0:3], y[..., 0:3])
        real_model_velocity_loss = F.mse_loss(physics_pred[..., 3:6], y[..., 3:6])
        real_model_quaternion_error = torch.mean(quaternion_multiply(quaternion_invert(physics_pred[..., 6:10]), y[..., 6:10]) ** 2)

        # --------------------------------------------
        # Compute the total loss
        # --------------------------------------------

        # Compute the total loss (fitting error + unit quaternion norm + physics error)
        total_loss = (self.regularization["mse_drone_position"] * mse_drone_position) + \
            (self.regularization["mse_drone_velocity"] * mse_drone_velocity) + \
            (self.regularization["quaternion_norm"] * quaternion_norm_loss) + \
            (self.regularization["attitude_error"] * mse_quat_error) + \
            (self.regularization["mse_payload_position"] * mse_payload_position) +  \
            (self.regularization["physical_model"]["mse_position"] * mse_physics_position) + \
            (self.regularization["physical_model"]["mse_velocity"] * mse_physics_velocity) + \
            (self.regularization["physical_model"]["mse_attitude"] * physics_quaternion_error)
        
        # Save the individual loss terms
        individual_terms = {
            "mse_drone_position": mse_drone_position,
            "mse_drone_velocity": mse_drone_velocity,
            "quaternion_norm": quaternion_norm_loss,
            "attitude_error": mse_quat_error,
            "mse_payload_position": mse_payload_position,
            "physical_loss": {
                "mse_position": mse_physics_position,
                "mse_velocity": mse_physics_velocity,
                "mse_attitude": physics_quaternion_error
            },
            # Note: the line bellow can be used to check the correctness of the physics model against real data
            "real_model_loss": {
                "real_model_mse_position": real_model_position_loss,
                "real_model_mse_velocity": real_model_velocity_loss,
                "real_model_mse_attitude": real_model_quaternion_error
            }
        }

        return total_loss, individual_terms