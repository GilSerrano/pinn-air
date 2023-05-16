#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the loss metrics that can be used for training this model
from loss import position_error, velocity_error, position_error_payload, continuity_last_input_first_output, output_continuity, quaternion_norm, quaternion_error, physics_error

class SuperModelo(nn.Module):

    def __init__(self):

        super(SuperModelo, self).__init__()

        # NOTA: adicionar relus no input? no output nao, porque podemos ter valores negativos no output
        
        # Input size and hidden size
        self.input_linear = nn.Linear(17, 100)
        self.input_linear2 = nn.Linear(100, 200)
        self.input_linear3 = nn.Linear(200, 300)

        self.lstm = nn.LSTM(300, 20, batch_first=True)
        self.linear = nn.Linear(20, 300)

        self.linear_out = nn.Linear(300, 200)
        self.linear_out2 = nn.Linear(200, 100)

        # We only want to output the position, velocity, quaternion and payload position for the next time step
        self.linear_out3 = nn.Linear(100, 13)

    def forward(self, x, u, target_time):

        outputs = torch.zeros(x.shape[0], target_time, 13).to("cuda")

        x = F.relu(self.input_linear(x))
        x = F.relu(self.input_linear2(x))
        x = F.relu(self.input_linear3(x))

        # Pass the previous sequence through the lstm
        lstm_out, hidden = self.lstm(x)
        lstm_out = self.linear(lstm_out[:, -1, :])
        lstm_out = lstm_out[:,None,:]

        # Save the output of the first sequence
        output = F.relu(self.linear_out(lstm_out))
        output = F.relu(self.linear_out2(output))
        output = self.linear_out3(output)
        outputs[:,0,:] = output[:,-1,:]

        # Predict the next sequence recursively
        for i in range(1, target_time):
            
            # Feed to the input of the network, the previous predicted state and the current input
            x = torch.cat((outputs[:,i-1,:], u[:,i-1,:]), dim=-1).unsqueeze(1)
            x = F.relu(self.input_linear(x))
            x = F.relu(self.input_linear2(x))
            x = F.relu(self.input_linear3(x))

            # Pass the latent variables through the lstm
            lstm_out, hidden = self.lstm(x, hidden)
            lstm_out = self.linear(lstm_out)

            # Save the ouput of the current sequence
            output = F.relu(self.linear_out(lstm_out))
            output = F.relu(self.linear_out2(output))
            output = self.linear_out3(output)
            outputs[:,i,:] = output[:,-1,:]

        return outputs
    

    def compute_loss(self, y_hat, y, x, target_time, physics_model):

        return 3 * position_error(y_hat, y, target_time) + \
            1 * velocity_error(y_hat, y, target_time) + \
            2 * position_error_payload(y_hat, y, target_time) + \
            2 * continuity_last_input_first_output(y_hat, x) + \
            1 * output_continuity(y_hat) + \
            2 * quaternion_norm(y_hat) + \
            1 * quaternion_error(y_hat, y, target_time) + \
            5 * physics_error(y_hat, x, y, target_time, physics_model)