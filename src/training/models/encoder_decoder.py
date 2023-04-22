import random 

import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1):
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        '''
        Args:
            x (torch.Tensor) :The input of the encoder (batch, timeseries_len, features)
        '''
        
        # Note, we use x.view to make sure that we have the input of the type (batch, timeseries_len, features)
        # Note: on an encoder, we do not care about the output of the LSTM but rather the propagation of the hidden states
        _, self.hidden = self.lstm(x.view(x.shape[0], x.shape[1], self.input_size))
        
        return self.hidden
    
class Decoder(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        """_summary_

        Args:
            input_size (int): The number of features
            hidden_size (int): The number of features in the hidden layer
            num_layers (int, optional): The number of stacked LSTM units. Defaults to 1.
        """

        super(Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        
        # Propagate once through the LSTM decoder
        lstm_out, hidden_out = self.lstm(x, hidden)

        # flatten output to (batch_size, hidden_size)
        lstm_out = lstm_out.view(-1, self.hidden_size)

        # Make the prediction with a given output size
        predictions = self.linear(lstm_out)

        return predictions, hidden_out
    

class EncoderDecoder(nn.Module):


    def __init__(self, input_size, target_size, hidden_size):
        """
        Args:
            input_size (int): The number of features in the input
            target_size (int): The number of features in the target output
            hidden_size (int): The number of features in the hidden layer
        """
        super(EncoderDecoder, self).__init__()

        self.input_size = input_size
        self.target_size = target_size
        self.hidden_size = hidden_size
        
        self.encoder = Encoder(input_size, hidden_size)
        self.decoder = Decoder(input_size, hidden_size, target_size)


    def forward(self, x, u, y, training_type='teacher', teacher_ratio=0.3):
        """
        Args:
            x (torch.Tensor): The input to the network of shape (batch, timeseries_len, features)
            [x,y,z | vx,vy,vz | qx,qy,qz,qw | p_load, y_load, z_load | wx,wy,wz,T ]
            ----
            u (torch.Tensor): The input to the network of shape (batch, target_len, [wx,wy,wz,T]) after the encoder
            y (torch.Tensor): The target data with shape (batch, target_len, number features)
            [x,y,z | vx,vy,vz | qx,qy,qz,qw | p_load, y_load, z_load]
            training_type (str, optional): The type of training to use. Defaults to 'teacher'. Can be 'teacher', 'recursive' or 'mixed'
            teacher_ratio (float, optional): The ratio of teacher forcing to use, when using mixed training. Defaults to 0.3.
        """

        # Get the current batch size and the target length to generate
        batch_size = x.shape[0]
        target_len = y.shape[1]

        # Encode the input tensor
        encoder_hidden = self.encoder(x)

        # Initialize tensor for predictions
        outputs = torch.zeros(batch_size, target_len, self.target_size)

        # Now we want to get only the last time sample from the vector and use it as the input to the decoder
        # and use the hidden state from the encoder as the initial hidden state of the decoder
        decoder_input = x[:, -1, :].unsqueeze(1)
        decoder_hidden = encoder_hidden

        # Now we want to predict the next time step: We might want to do this recursively, where the output of the
        # previous time step is used as the input to the next time step.
        if training_type == 'recursive':
            for t in range(target_len):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                outputs[:, t, :] = decoder_output

                # Concatenate the predicted output (which is the next timestep state) with the input angular velocity and thrust of the vehicle
                decoder_input = torch.cat((decoder_output, u[:, t, :]), dim=-1).unsqueeze(1)

        # When using teacher forcing, we will use the target data as the input to the decoder in the next time step
        # instead of using the output from the previous time step
        elif training_type == 'teacher':
            for t in range(target_len):

                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                outputs[:, t, :] = decoder_output

                # Concatenate the next real timestep state with the input angular velocity and thrust of the vehicle
                decoder_input = torch.cat((y[:, t, :], u[:, t, :]), dim=-1).unsqueeze(1)

        elif training_type == "mixed":
            for t in range(target_len):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                outputs[:, t, :] = decoder_output

                # Check in the next timestep whether we want to use teacher forcing or not for the prediction
                if random.random() < teacher_ratio:
                    decoder_input = torch.cat((y[:, t, :], u[:, t, :]), dim=-1).unsqueeze(1)
                else:
                    decoder_input = torch.cat((decoder_output, u[:, t, :]), dim=-1).unsqueeze(1)
        
        return outputs

    def predict_sequence_recursively(self, x, u):
        """
        Args:
            x (torch.Tensor): The input to the network of shape (batch, timeseries_len, features)
            u (torch.Tensor): The input angular velocity and thrust after the encoder of shape (batch, target_len, [wx,wy,wz,T])
        """

        with torch.no_grad():

            # Get the current batch size and the target length to generate
            batch_size = x.shape[0]
            target_len = x.shape[1]

            # Encode the input tensor
            encoder_hidden = self.encoder(x)

            # Initialize tensor for predictions
            outputs = torch.zeros(batch_size, target_len, self.target_size)

            # Now we want to get only the last time sample from the vector and use it as the input to the decoder
            # and use the hidden state from the encoder as the initial hidden state of the decoder
            decoder_input = x[:, -1, :].unsqueeze(1)
            decoder_hidden = encoder_hidden

            # Now we want to predict the next time step: We might want to do this recursively, where the output of the
            # previous time step is used as the input to the next time step.
            for t in range(target_len):

                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                outputs[:, t, :] = decoder_output

                # Concatenate the predicted output (which is the next timestep state) with the input angular velocity and thrust of the vehicle
                decoder_input = torch.cat((decoder_output, u[:, t, :]), dim=-1).unsqueeze(1)

            return outputs

    def compute_loss(self, y, y_hat):

        # Compute the loss
        loss = F.mse_loss(y[..., 0:3], y_hat[..., 0:3])

        return loss, {}


if __name__ == "__main__":

    # Create the model
    model = EncoderDecoder(input_size=17, hidden_size=20, target_size=13)

    # Create a random input tensor
    x = torch.randn(32, 10, 17)         # (batch, timeseries_input_len, features)

    # Create a random target tensor and inputs after the encoder
    y = torch.randn(32, 5, 13)          # (batch, timeseries_target_len, target_features)
    u = torch.randn(32, 5, 4)           # (batch, timeseries_target_len, [wx,wy,wz,T])

    model(x, u, y, training_type='mixed', teacher_ratio=0.3)
    output = model.predict_sequence_recursively(x, u)
    print(output.shape)