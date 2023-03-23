import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, n_in, n_out, **kwargs):

        super(MLP, self).__init__()

        hidden_sizes = [100, 150, 100]
        droupout_rate = 0.1

        self.model = nn.Sequential(
            nn.Linear(n_in, hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(droupout_rate),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Dropout(droupout_rate),
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.ReLU(),
            nn.Dropout(droupout_rate),
            nn.Linear(hidden_sizes[2], n_out),
            nn.ReLU()
        )

    def forward(self, x, **kwargs):
        """
        x (batch_size x n_features): a batch of training examples
        """
        return self.model(x)
