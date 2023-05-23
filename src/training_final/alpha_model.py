#!/usr/bin/env python3
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the loss metrics that can be used for training this model
from loss import position_error, velocity_error, position_error_payload, continuity_last_input_first_output, output_continuity, quaternion_norm, quaternion_error, physics_error


class Encoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, bidirectional, num_layers, dropout, device):

        super(Encoder, self).__init__()

        # Set the device
        self.device = device

        # Setup the LSTM layer
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.dropout = dropout if num_layers > 1 else 0

        # Setup the LSTM layer
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)

    def forward(self, x):
            
        # Pass the previous sequence through the lstm
        lstm_out, hidden = self.encoder(x)

        return lstm_out, hidden

class Decoder(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim, bidirectional, num_layers, dropout, device):

        super(Decoder, self).__init__()

        # Setup the device
        self.device = device

        # Setup the LSTM layer
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.dropout = dropout if num_layers > 1 else 0

        # Setup the LSTM layer and an output layer
        self.decoder = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.linear_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, decoder_hidden):
        
        decoder_out, decoder_hidden = self.decoder(x, decoder_hidden)

        # Get the prediction
        prediction = self.linear_out(decoder_out)

        return prediction, decoder_hidden
    

class AlphaModel(nn.Module):

    def __init__(self, output_dim, num_layers, dropout, device):

        super(AlphaModel, self).__init__()

        # Set the device and other parameters
        self.device = device
        self.dropout = dropout if num_layers > 1 else 0

        # Setup the encoder part
        self.input_linear = nn.Linear(17, 100)
        self.input_linear2 = nn.Linear(100, 200)
        self.input_linear3 = nn.Linear(200, 300)
        self.encoder = Encoder(input_dim=300, hidden_dim=20, bidirectional=False, num_layers=num_layers, dropout=dropout, device=device)

        # Setup the decoder part
        self.decoder = Decoder(input_dim=300, hidden_dim=20, output_dim=300 ,bidirectional=False, num_layers=num_layers, dropout=dropout, device=device)
        self.linear_out = nn.Linear(300, 200)
        self.linear_out2 = nn.Linear(200, 100)
        self.linear_out3 = nn.Linear(100, output_dim)

    def forward(self, x, u, target_y=None, teacher_forcing_ratio=0.0):
        
        # Get the target time from the size of the control inputs
        target_time = u.shape[1]

        # Create a tensor to store the outputs of shape (batch_size, target_time, num_states)
        outputs = torch.zeros(x.shape[0], target_time, 13).to(self.device)

        # Pass the input through the linear layers
        z = x
        z = F.dropout(F.relu(self.input_linear(z)), training=self.training, p=self.dropout)
        z = F.dropout(F.relu(self.input_linear2(z)), training=self.training, p=self.dropout)
        z = F.dropout(F.relu(self.input_linear3(z)), training=self.training, p=self.dropout)

        # Pass the previous sequence through the lstm
        _, hidden = self.encoder(z)

        # Get the first input of the decoder, which is the last state [x, u]
        decoder_input = x[:,-1,:].unsqueeze(1)

        # Predict the next sequence recursively
        for t in range(0, target_time):

            # Feed the decoder input thorugh the input linear layers
            decoder_input = F.dropout(F.relu(self.input_linear(decoder_input)), training=self.training, p=self.dropout)
            decoder_input = F.dropout(F.relu(self.input_linear2(decoder_input)), training=self.training, p=self.dropout)
            decoder_input = F.dropout(F.relu(self.input_linear3(decoder_input)), training=self.training, p=self.dropout)
            
            # Feed to the input of the network, the previous predicted state and the current input
            decoder_out, hidden = self.decoder(decoder_input, hidden)

            decoder_out = F.dropout(F.relu(self.linear_out(decoder_out)), training=self.training, p=self.dropout)
            decoder_out = F.dropout(F.relu(self.linear_out2(decoder_out)), training=self.training, p=self.dropout)
            decoder_out = self.linear_out3(decoder_out)

            # Save the output of the decoder
            outputs[:,t,:] = decoder_out[:,-1,:]

            # Decide if we are going to use teacher forcing or not in the next iteration
            teacher_force = True if random.random() < teacher_forcing_ratio else False

            # Use the teacher forcing or not
            decoder_input = target_y[:,t,:].unsqueeze(1) if teacher_force and target_y is not None else torch.cat((decoder_out[:,-1,:], u[:,t,:]), dim=-1).unsqueeze(1)

        return outputs
    
    def predict(self, x, u):
        """Given the N timesteps of the state and inputs of the system X and the next U input of the system (for the next N timesteps),
        predict the next N states of the system.

        Args:
            x (torch.Tensor): A tensor with the previous M state + input of the system of shape (batch_size, M, 17)
            u (torch.Tensor): A tensor with the next input of the system for the next N timesteps of shape (batch_size, N, 4)

        Returns:
            torch.Tensor: A tensor with the next N states of the system of shape (batch_size, N, 13)
        """
        return self.forward(x, u, target_y=None, teacher_forcing_ratio=0.0)

    def set_loss_params(self, position_error, velocity_error, position_error_payload, continuity_last_input_first_output, output_continuity, quaternion_norm, quaternion_error, physics_error):

        self.position_error = position_error
        self.velocity_error = velocity_error
        self.position_error_payload = position_error_payload
        self.continuity_last_input_first_output = continuity_last_input_first_output
        self.output_continuity = output_continuity
        self.quaternion_norm = quaternion_norm
        self.quaternion_error = quaternion_error
        self.physics_error = physics_error

        print("-------------------")
        print("Loss params set to:")
        print("-------------------")
        print("position_error: " + str(self.position_error))
        print("velocity_error: " + str(self.velocity_error))
        print("position_error_payload: " + str(self.position_error_payload))
        print("continuity_last_input_first_output: " + str(self.continuity_last_input_first_output))
        print("output_continuity: " + str(self.output_continuity))
        print("quaternion_norm: " + str(self.quaternion_norm))
        print("quaternion_error: " + str(self.quaternion_error))
        print("physics_error: " + str(self.physics_error))

    def compute_loss(self, y_hat, y, x, target_time, physics_model):

        return self.position_error * position_error(y_hat, y, target_time, decay_factor=0.0) + \
            self.velocity_error * velocity_error(y_hat, y, target_time, decay_factor=0.0) + \
            self.position_error_payload * position_error_payload(y_hat, y, target_time, decay_factor=0.0) + \
            self.continuity_last_input_first_output * continuity_last_input_first_output(y_hat, x) + \
            self.output_continuity * output_continuity(y_hat) + \
            self.quaternion_norm * quaternion_norm(y_hat) + \
            self.quaternion_error * quaternion_error(y_hat, y, target_time, decay_factor=0.0) + \
            self.physics_error * physics_error(y_hat, x, y, target_time, physics_model, decay_factor=0.1)