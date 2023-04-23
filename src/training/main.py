import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from data_loaders.mocap_swipe_loader import MocapSwipeLoader
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib.pyplot as plt


class SuperModelo(nn.Module):

    def __init__(self):

        super(SuperModelo, self).__init__()
        
        # Input size and hidden size
        self.input_linear = nn.Linear(3, 100)
        self.lstm = nn.LSTM(100, 15, batch_first=True)
        self.linear = nn.Linear(15, 100)


        self.linear_out = nn.Linear(100, 3)

    def forward(self, x, target_time):

        outputs = torch.zeros(x.shape[0], target_time, x.shape[2]).to("cpu")

        x = self.input_linear(x)

        # Pass the previous sequence through the lstm
        lstm_out, hidden = self.lstm(x)
        lstm_out = self.linear(lstm_out[:, -1, :])
        lstm_out = lstm_out[:,None,:]
        
        output = self.linear_out(lstm_out)
        outputs[:,0,:] = output
        

        # Predict the next sequence recursively
        for i in range(1, target_time):

            lstm_out, hidden = self.lstm(lstm_out, hidden)
            lstm_out = self.linear(lstm_out[:,-1,:])

            output = self.linear_out(lstm_out)
            outputs[:,i,:] = output
            lstm_out = lstm_out[:,None,:]

        return outputs

model = SuperModelo().to("cpu")
optimizer = AdamW(model.parameters(), lr=1E-4, weight_decay=0.05)

target_time = 50

# Load the dataset
train_dataset = MocapSwipeLoader(input_window=50, output_window=target_time, stride=1, split="train", device="cpu")

train_loader = DataLoader(
               train_dataset,                                           # The dataset itself
               batch_size=64,                                           # The size of each batch
               shuffle=False,                                           # Whether to shuffle the sequences on the dataset
               num_workers=0,                                           # The number of cpu to use to load the dataset and generate the batches in parallel
               drop_last=False,                                         # Drop the last samples if not enough to make a batch of the desired size
               collate_fn=lambda x: train_dataset.collate(x, "cpu"),    # Custom collate function for the dataset
               multiprocessing_context=None)

epochs = torch.arange(0, 650, 1)



# Enable gradient tracking
with torch.enable_grad():
    
    # Train for the desired number of epochs
    for epoch in epochs:

        print('Training epoch {}'.format(epoch))

        # Train loss
        train_loss = []
        train_individual_losses = []

        # Set the model to be in training mode
        model.train()

        for batch in tqdm(train_loader, desc="Computing batch"):

            # Zero the gradients
            optimizer.zero_grad()

            # Get the input of the network and the expect output from the batch
            x, y = train_dataset[0]
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)

            # Perform a forward pass
            y_hat = model(x, target_time)

            # Forward pass
            total_loss = 0
            for elem in range(x.shape[0]):     # (batch, time, features)
                # For each feature
                for i in range(x.shape[2]):
                    loss = F.mse_loss(y_hat[elem, :, i], y[elem, :, i])
                    total_loss += loss

            # Backward pass
            total_loss.backward()

            # Update the parameters
            optimizer.step()

            print(total_loss.item())

        if epoch >= 600:

            print("----")

            for i in range(len(train_dataset)):

                x, y = train_dataset[i]

                y_hat = model(x[None, :, :], target_time)
                
                print(y_hat.shape)
                time = torch.arange(0, x.shape[0] + y.shape[0]).numpy(force=True)

                # Plot the resulting prediction
                plt.figure()
                plt.plot(time, torch.cat((x[:, 0], y[:, 0]), dim=0).numpy(force=True), label="x")        # x
                plt.plot(time, torch.cat((x[:, 1], y[:, 1]), dim=0).numpy(force=True), label="y")        # y
                plt.plot(time, torch.cat((x[:, 2], y[:, 2]), dim=0).numpy(force=True), label="z")        # z

                time2 = torch.arange(x.shape[0], x.shape[0] + y_hat.shape[1]).numpy(force=True)
                plt.plot(time2, y_hat[0, :, 0].numpy(force=True), label="x_hat")        # x
                plt.plot(time2, y_hat[0, :, 1].numpy(force=True), label="y_hat")        # y
                plt.plot(time2, y_hat[0, :, 2].numpy(force=True), label="z_hat")        # z

                plt.legend()
                plt.show()

