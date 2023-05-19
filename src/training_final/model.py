#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the loss metrics that can be used for training this model
from loss import position_error, velocity_error, position_error_payload, continuity_last_input_first_output, output_continuity, quaternion_norm, quaternion_error, physics_error

class SuperModelo(nn.Module):

    def __init__(self, device):

        super(SuperModelo, self).__init__()

        self.device = device

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

    def forward(self, x, u, target_y=None, teacher_forcing_ratio=0.0):

        # UNUSED IN THIS MODEL
        target_y = None
        teacher_forcing_ratio = 0.0

        # Get the target time
        target_time = u.shape[1]

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

        return self.position_error * position_error(y_hat, y, target_time) + \
            self.velocity_error * velocity_error(y_hat, y, target_time) + \
            self.position_error_payload * position_error_payload(y_hat, y, target_time) + \
            self.continuity_last_input_first_output * continuity_last_input_first_output(y_hat, x) + \
            self.output_continuity * output_continuity(y_hat) + \
            self.quaternion_norm * quaternion_norm(y_hat) + \
            self.quaternion_error * quaternion_error(y_hat, y, target_time) + \
            self.physics_error * physics_error(y_hat, x, y, target_time, physics_model)