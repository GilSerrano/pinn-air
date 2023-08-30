#!/usr/bin/env python3
"""
| File: pinn_air_model.py
| Authors: Gil Serrano, Marcelo Jacinto, Jose Gomes and Joao Pinto
| License: BSD-3-Clause. Copyright (c) 2023, Gil Serrano, Marcelo Jacinto, Jose Gomes and Joao Pinto. All rights reserved.
| Description: Definition of the encoder-decoder model with slack variables.
"""
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the discrete drone physics model (with and without payload)
from pinn_air.physics.discrete_model import DiscreteModel
from pinn_air.physics.drone_physics import DronePhysics

# Import the loss metrics that can be used for training this model
from pinn_air.models.loss import position_error, velocity_error, position_error_payload, continuity_last_input_first_output, output_continuity, quaternion_norm, quaternion_error, physics_error, physics_error_without_payload, continuity_last_input_first_output_without_payload, output_continuity_without_payload, iterative_physics_error, iterative_physics_error_without_payload, slack_weight_reduction, slack_weight_reduction_without_payload


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

class BahdanauAttention(nn.Module):

    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.hidden_size = hidden_size
        
        self.W = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, encoder_outputs, hidden_state):
        
        hidden_state = torch.unsqueeze(hidden_state[-1], dim=1)

        # Compute the attention scores using the encoder outputs and the hidden state
        percep = torch.tanh(self.W(encoder_outputs) + self.U(hidden_state)) 
        attention_scores = self.v(percep)
        
        # Compute the attention weights using softmax
        attention_weights = torch.softmax(attention_scores, dim=1)
        attention_weights = attention_weights.transpose(1, 2)

        # Compute the context vector as a weighted sum of the encoder outputs
        context = torch.bmm(attention_weights, encoder_outputs)  

        return context


class DecoderWithAttention(Decoder):
    
    def __init__(self, input_dim, output_dim, hidden_dim, bidirectional, num_layers, dropout, device):
        super().__init__(input_dim, output_dim, hidden_dim, bidirectional, num_layers, dropout, device)
        
        self.attention = BahdanauAttention(hidden_dim)
        self.decoder = nn.LSTM(input_dim + hidden_dim, hidden_dim, num_layers=num_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)

    def forward(self, x, decoder_hidden, encoder_outputs):
        
        hidden, cell = decoder_hidden

        # Compute the context vector and the attention weights using the attention mechanism
        context = self.attention(encoder_outputs, hidden)

        # Concatenate context vector and input decoder to be the new input of the LSTM
        x = torch.cat((context, x), dim=2)

        # decoder_out, decoder_hidden = self.decoder(x, (hidden.unsqueeze(0), cell.unsqueeze(0)))
        decoder_out, decoder_hidden = self.decoder(x, (hidden, cell))

        # Get the prediction
        prediction = self.linear_out(decoder_out)

        return prediction, decoder_hidden
    

class PINNAirModel(nn.Module):

    def __init__(self, system_state_dim, system_control_dim, num_layers, dropout, device):
        
        # Call the super constructor
        super(PINNAirModel, self).__init__()

        # Define the dimensions of the dynamical system we are trying to aproximate
        self.system_state_dim = system_state_dim
        self.system_control_dim = system_control_dim

        # Set the physics model depending on the system state dimension (if 10, regular drone physics, if 13, drone physics with payload)
        if self.system_state_dim == 10:
            Ts = 0.03
            mass_quadrotor = 0.772
            self.physics_model = DronePhysics(Ts, mass_quadrotor, device=device)
        elif self.system_state_dim == 13:
            Ts = 0.03
            mass_quadrotor = 1.35
            mass_payload = 0.1
            l = 0.7
            self.physics_model = DiscreteModel(Ts=Ts, mQ=mass_quadrotor, mL=mass_payload, l=l)

        # Set the device and other parameters
        self.device = device
        self.dropout = dropout if num_layers > 1 else 0

        # Setup the encoder part
        self.input_linear = nn.Linear(self.system_state_dim + self.system_control_dim, 100)
        self.input_linear2 = nn.Linear(100, 200)
        self.input_linear3 = nn.Linear(200, 300)
        self.encoder = Encoder(input_dim=300, hidden_dim=20, bidirectional=False, num_layers=num_layers, dropout=dropout, device=device)

        # Setup the decoder part
        self.linear_out = nn.Linear(300, 200)
        self.linear_out2 = nn.Linear(200, 100)

        # Let the output of the system have the size of the system state * 2 (because we want to add one slack variable per system state)
        self.linear_out3 = nn.Linear(100, self.system_state_dim * 2)

        # Setup the kind of loss to use
        self.decoder = DecoderWithAttention(input_dim=300, hidden_dim=20, output_dim=300 ,bidirectional=False, num_layers=num_layers, dropout=dropout, device=device)
        if self.system_state_dim == 10:
            self.loss = self.loss_without_payload
        elif self.system_state_dim == 13:
            self.loss = self.loss_with_payload
        

    def forward(self, x, u, target_y=None, teacher_forcing_ratio=0.0):
        
        # Get the target time from the size of the control inputs
        target_time = u.shape[1]

        # Create a tensor to store the outputs of shape (batch_size, target_time, num_states * 2)
        # The output must have num_state * 2, because we want to add one slack variable per state
        # which are not used explicitly during evaluation time, but are used during training time
        outputs = torch.zeros(x.shape[0], target_time, self.system_state_dim * 2).to(self.device)

        # Pass the input through the linear layers
        z = x
        z = F.dropout(F.gelu(self.input_linear(z)), training=self.training, p=self.dropout)
        z = F.dropout(F.gelu(self.input_linear2(z)), training=self.training, p=self.dropout)
        z = F.dropout(F.gelu(self.input_linear3(z)), training=self.training, p=self.dropout)

        # Pass the previous sequence through the lstm
        enconder_output, hidden = self.encoder(z)

        # Get the first input of the decoder, which is the last state [x, u]
        decoder_input = x[:,-1,:].unsqueeze(1)

        # Predict the next sequence recursively
        for t in range(0, target_time):

            # Feed the decoder input thorugh the input linear layers
            decoder_input = F.dropout(F.gelu(self.input_linear(decoder_input)), training=self.training, p=self.dropout)
            decoder_input = F.dropout(F.gelu(self.input_linear2(decoder_input)), training=self.training, p=self.dropout)
            decoder_input = F.dropout(F.gelu(self.input_linear3(decoder_input)), training=self.training, p=self.dropout)
            
            # Feed to the input of the network, the previous predicted state and the current input
            decoder_out, hidden = self.decoder(decoder_input, hidden, enconder_output)

            decoder_out = F.dropout(F.gelu(self.linear_out(decoder_out)), training=self.training, p=self.dropout)
            decoder_out = F.dropout(F.gelu(self.linear_out2(decoder_out)), training=self.training, p=self.dropout)
            decoder_out = self.linear_out3(decoder_out)

            # Pass the slack variables through a RELU output, to make sure that slack_variables are always positive >= 0
            #decoder_out[..., self.system_state_dim:self.system_state_dim*2] = F.relu(decoder_out[..., self.system_state_dim:self.system_state_dim*2])

            # Save the output of the decoder
            outputs[:,t,:] = decoder_out[:,-1,:]

            # Decide if we are going to use teacher forcing or not in the next iteration
            teacher_force = True if random.random() < teacher_forcing_ratio else False

            # Use the teacher forcing or not
            # Also, discard the last num_state variables from the predicted system state, because they are the slack variables
            # and not actual system states
            decoder_input = target_y[:,t,:].unsqueeze(1) if teacher_force and target_y is not None else torch.cat((decoder_out[:,-1,0:self.system_state_dim], u[:,t,:]), dim=-1).unsqueeze(1)

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


    def set_loss_params(self, position_error, velocity_error, position_error_payload, continuity_last_input_first_output, output_continuity, quaternion_norm, quaternion_error, physics_error, exponential_decay_real, exponential_decay_physics, slack_weight):

        self.position_error = position_error
        self.velocity_error = velocity_error
        self.position_error_payload = position_error_payload
        self.continuity_last_input_first_output = continuity_last_input_first_output
        self.output_continuity = output_continuity
        self.quaternion_norm = quaternion_norm
        self.quaternion_error = quaternion_error
        self.physics_error = physics_error
        self.exponential_decay_real = exponential_decay_real
        self.exponential_decay_physics = exponential_decay_physics
        self.slack_weight = slack_weight

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
        print("exponential_decay_real: " + str(self.exponential_decay_real))
        print("exponential_decay_physics: " + str(self.exponential_decay_physics))
        print("slack_weight: " + str(self.slack_weight))

    def loss_with_payload(self, y_hat, y, x):

        # Compute the loss on the values of the slack variables 
        slack_loss = slack_weight_reduction(y_hat, self.exponential_decay_physics)

        return self.position_error * position_error(y_hat, y, self.exponential_decay_real) + \
            self.velocity_error * velocity_error(y_hat, y, self.exponential_decay_real) + \
            self.position_error_payload * position_error_payload(y_hat, y, self.exponential_decay_real) + \
            self.continuity_last_input_first_output * continuity_last_input_first_output(y_hat, x) + \
            self.output_continuity * output_continuity(y_hat) + \
            self.quaternion_norm * quaternion_norm(y_hat) + \
            self.quaternion_error * quaternion_error(y_hat, y, self.exponential_decay_real) + \
            self.physics_error * iterative_physics_error(y_hat, y, x, self.physics_model, self.exponential_decay_physics) + \
            self.slack_weight * slack_loss, {"slack": slack_loss}
            #self.physics_error * physics_error(y_hat, y_physics, self.exponential_decay_physics)    

    def loss_without_payload(self, y_hat, y, x):

        # Compute the loss on the values of the slack variables
        slack_loss = slack_weight_reduction_without_payload(y_hat, self.exponential_decay_physics)

        return self.position_error * position_error(y_hat, y, self.exponential_decay_real) + \
            self.velocity_error * velocity_error(y_hat, y, self.exponential_decay_real) + \
            self.continuity_last_input_first_output * continuity_last_input_first_output_without_payload(y_hat, x) + \
            self.output_continuity * output_continuity_without_payload(y_hat) + \
            self.quaternion_norm * quaternion_norm(y_hat) + \
            self.quaternion_error * quaternion_error(y_hat, y, self.exponential_decay_real) + \
            self.physics_error * iterative_physics_error_without_payload(y_hat, y, x, self.physics_model, self.exponential_decay_physics) + \
            self.slack_weight * slack_loss(y_hat), {"slack": slack_loss}
            # self.physics_error * physics_error_without_payload(y_hat, y_physics, self.exponential_decay_physics)

    def compute_loss(self, y_hat, y, x):
        return self.loss(y_hat, y, x)