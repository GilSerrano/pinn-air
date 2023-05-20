#!/usr/bin/env python3
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from alpha_model import AlphaModel, Decoder

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
    

class OmegaModel(AlphaModel):

    def __init__(self, output_dim, num_layers, dropout, device):
        super().__init__(output_dim, num_layers, dropout, device)
        self.decoder = DecoderWithAttention(input_dim=300, hidden_dim=20, output_dim=300 ,bidirectional=False, num_layers=num_layers, dropout=dropout, device=device)

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
        enconder_output, hidden = self.encoder(z)

        # Get the first input of the decoder, which is the last state [x, u]
        decoder_input = x[:,-1,:].unsqueeze(1)

        # Predict the next sequence recursively
        for t in range(0, target_time):

            # Feed the decoder input thorugh the input linear layers
            decoder_input = F.dropout(F.relu(self.input_linear(decoder_input)), training=self.training, p=self.dropout)
            decoder_input = F.dropout(F.relu(self.input_linear2(decoder_input)), training=self.training, p=self.dropout)
            decoder_input = F.dropout(F.relu(self.input_linear3(decoder_input)), training=self.training, p=self.dropout)
            
            # Feed to the input of the network, the previous predicted state and the current input
            decoder_out, hidden = self.decoder(decoder_input, hidden, enconder_output)

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