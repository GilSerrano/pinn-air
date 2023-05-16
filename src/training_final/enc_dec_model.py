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
        self.encoder = nn.LSTM(17, input_dim, num_layers=num_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)

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
        return decoder_out, decoder_hidden
    

class AlphaModel(nn.Module):

    def __init__(self, output_dim, hidden_dim, num_layers, dropout, device):

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

    def forward(self, x, u,  target_time, target_y=None, teacher_forcing_ratio=0.0):
        
        # Create a tensor to store the outputs of shape (batch_size, target_time, num_states)
        outputs = torch.zeros(x.shape[0], target_time, 13).to(self.device)

        # Pass the input through the linear layers
        x = F.relu(self.input_linear(x))
        x = F.relu(self.input_linear2(x))
        x = F.relu(self.input_linear3(x))

        # Pass the previous sequence through the lstm
        _, hidden = self.encoder(x)

        # Get the first input of the decoder, which is the last state [x, u]
        decoder_input = x[:,-1,:].unsqueeze(1)

        # Predict the next sequence recursively
        for t in range(1, target_time):

            # Feed the decoder input thorugh the input linear layers
            decoder_input = F.relu(self.input_linear(decoder_input))
            decoder_input = F.relu(self.input_linear2(decoder_input))
            decoder_input = F.relu(self.input_linear3(decoder_input))
            
            # Feed to the input of the network, the previous predicted state and the current input
            decoder_out, hidden = self.decoder(decoder_input, hidden)
            decoder_out = self.linear_out(decoder_out)
            decoder_out = self.linear_out2(decoder_out)
            decoder_out = self.linear_out3(decoder_out)

            # Save the output of the first sequence
            # TODO: check the output dimension of the decoder
            outputs[:,t,:] = decoder_out[:,-1,:]

            # Decide if we are going to use teacher forcing or not in the next iteration
            teacher_force = True if random.random() < teacher_forcing_ratio else False

            # Use the teacher forcing or not
            decoder_input = target_y[:,t,:].unsqueeze(1) if teacher_force and target_y is not None else torch.cat((decoder_out[:,-1,:], u[:,t,:]), dim=1).unsqueeze(1)

            # Get the next input of the decoder
            decoder_input = decoder_out[:,None,:]

        return outputs
    
    def predict(self, x, u, )

    def compute_loss(self, y_hat, y, x, target_time, physics_model):

        return 3 * position_error(y_hat, y, target_time) + \
            1 * velocity_error(y_hat, y, target_time) + \
            2 * position_error_payload(y_hat, y, target_time) + \
            2 * continuity_last_input_first_output(y_hat, x) + \
            1 * output_continuity(y_hat) + \
            2 * quaternion_norm(y_hat) + \
            1 * quaternion_error(y_hat, y, target_time) + \
            5 * physics_error(y_hat, x, y, target_time, physics_model)