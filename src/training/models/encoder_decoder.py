import torch
import torch.nn as nn

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

    def __init__(self, input_size, hidden_size, num_layers=1):
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

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, input_size)

    def forward(self, x, hidden):

        lstm_out, hidden = self.lstm(x, hidden)
        predictions = self.linear(lstm_out)

        return predictions, hidden
    

class EncoderDecoder(nn.Module):


    def __init__(self, input_size, target_size, hidden_size):
        """_

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
        self.decoder = Decoder(target_size, hidden_size)


    def forward(self, x):
        """_summary_

        Args:
            x (torch.Tensor): The input to the network of shape (batch, timeseries_len, features)
        """

        # Encode the input tensor
        encoder_hidden = self.encoder(x)

        # Initialize tensor for predictions
        outputs = torch.zeros(x.shape[0], x.shape[1], self.target_size)

        # TODO: continue from here

        # decode input_tensor
        decoder_input = input_tensor[-1, :, :]
        decoder_hidden = encoder_hidden
        
        for t in range(target_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[t] = decoder_output.squeeze(0)
            decoder_input = decoder_output
            
        np_outputs = outputs.detach().numpy()
        
        return np_outputs

