import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from data_loaders.mocap_swipe_loader import MocapSwipeLoader
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib.pyplot as plt

from utils.fix_seed import fix_seed


from torch.utils.tensorboard import SummaryWriter

fix_seed(0)

class SuperModelo(nn.Module):

    def __init__(self):

        super(SuperModelo, self).__init__()
        
        # Input size and hidden size
        self.input_linear = nn.Linear(3, 100)
        self.input_linear2 = nn.Linear(100, 200)
        self.input_linear3 = nn.Linear(200, 300)

        self.lstm = nn.LSTM(300, 15, batch_first=True)
        self.linear = nn.Linear(15, 300)

        self.linear_out = nn.Linear(300, 3)

    def forward(self, x, target_time):

        outputs = torch.zeros(x.shape[0], target_time, x.shape[2]).to("cuda")

        x = self.input_linear(x)
        x = self.input_linear2(x)
        x = self.input_linear3(x)

        # Pass the previous sequence through the lstm
        lstm_out, hidden = self.lstm(x)
        lstm_out = self.linear(lstm_out[:, -1, :])
        lstm_out = lstm_out[:,None,:]
        output = self.linear_out(lstm_out)
        outputs[:,0,:] = output[:,-1,:]

        # Predict the next sequence recursively
        for i in range(1, target_time):

            lstm_out, hidden = self.lstm(lstm_out, hidden)
            lstm_out = self.linear(lstm_out[:,-1,:])

            output = self.linear_out(lstm_out)
            outputs[:,i,:] = output
            lstm_out = lstm_out[:,None,:]

        return outputs

torch.set_default_device("cuda")
model = SuperModelo().to("cuda")
optimizer = AdamW(model.parameters(), lr=1E-4, weight_decay=0.05)

writer = SummaryWriter(log_dir="output")

target_time = 25

# Load the dataset
train_dataset = MocapSwipeLoader(input_window=50, output_window=target_time, stride=1, split="train", device="cuda")
validation_dataset = MocapSwipeLoader(input_window=50, output_window=target_time, stride=1, split="val", device="cuda")

train_loader = DataLoader(
               train_dataset,                                           # The dataset itself
               batch_size=64,                                           # The size of each batch
               shuffle=False,                                           # Whether to shuffle the sequences on the dataset
               num_workers=0,                                           # The number of cuda to use to load the dataset and generate the batches in parallel
               drop_last=False,                                         # Drop the last samples if not enough to make a batch of the desired size
               collate_fn=lambda x: train_dataset.collate(x, "cuda"),    # Custom collate function for the dataset
               multiprocessing_context=None)

validation_loader = DataLoader(
            validation_dataset,                                           # The dataset itself
            batch_size=64,                                                # The size of each batch
                shuffle=False,                                            # Whether to shuffle the sequences on the dataset
                num_workers=0,                                            # The number of cuda to use to load the dataset and generate the batches in parallel
                drop_last=False,                                          # Drop the last samples if not enough to make a batch of the desired size
            collate_fn=lambda x: validation_dataset.collate(x, "cuda"),    # Custom collate function for the dataset
            multiprocessing_context=None)

epochs = torch.arange(0, 2, 1)


def save_model(epoch, train_loss, val_loss, save_dir="output/"):

    checkpoint_path = os.path.join(save_dir, 'checkpoint_{:04d}.pth.tar'.format(epoch))
    print('Saving checkpoint {}'.format(checkpoint_path))

    # Save the current model
    torch.save({
        "epoch": epoch,
        "train_loss": train_loss,
        "validation_loss": val_loss,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }, checkpoint_path)

def save_best_model_idx(epoch, val_loss, save_dir="output/"):

    checkpoint_path = os.path.join(save_dir, 'best_model_idx.pth.tar')
    print('Saving checkpoint {}'.format(checkpoint_path))

    # Save the current model
    torch.save({"best_model_idx": best_model_idx, "best_val_loss": best_val_loss,}, checkpoint_path)

def load_best_model(epoch, save_dir="output/"):
    
    checkpoint_path = os.path.join(save_dir, 'checkpoint_{:04d}.pth.tar'.format(epoch))
    print("Loading: {}".format(checkpoint_path))

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])


def continuity_loss(y_hat, y, target_time):
    exp_decay = 10 * torch.exp(-0.1*torch.arange(0, target_time, 1)).to("cuda")
    return torch.sum((torch.norm(y_hat - y, dim=2) ** 2) * exp_decay)

def continuity_last_input_first_output(y_hat, x):
    return torch.sum(torch.norm(y_hat[:, 0, :] - x[:, -1, :], dim=1) **2, dim=0)

def output_continuity(y_hat):

    total_loss = 0
    for t in range(y.shape[1] - 1):
        total_loss += torch.sum(torch.norm(y_hat[:, t, :] - y_hat[:, t+1, :], dim=1) ** 2, dim=0)

    return total_loss

def compute_loss(y_hat, y, x, target_time):

    return continuity_loss(y_hat, y, target_time) + \
        100 * continuity_last_input_first_output(y_hat, x) + \
        output_continuity(y_hat)
    
# Train the model
best_model_idx = 0
best_val_loss = 1E1000
for epoch in epochs:

    # Enable gradient tracking
    with torch.enable_grad():

        model.train()

        print('Training epoch {}'.format(epoch))

        # Train loss
        train_loss = []

        # Set the model to be in training mode
        model.train()

        for batch in tqdm(train_loader, desc="Computing batch"):

            # Zero the gradients
            optimizer.zero_grad()

            # Get the input of the network and the expect output from the batch
            x, y = batch

            # Perform a forward pass
            y_hat = model(x, target_time)

            # --------------------------------------
            # LOSS
            # --------------------------------------
            total_loss = compute_loss(y_hat, y, x, target_time)

            # --------------------------------------
            train_loss.append(total_loss.item())

            # Backward pass
            total_loss.backward()

            # Update the parameters
            optimizer.step()

    # Perform validation
    model.eval()
    val_loss = []
    with torch.no_grad():

        for x, y in validation_loader:
            
            # Perform a forward pass
            y_hat = model(x, target_time)

            total_val_loss = compute_loss(y_hat, y, x, target_time)

            val_loss.append(total_val_loss.item())
    
    print("Epoch {} - Train loss: {}".format(epoch, torch.tensor(train_loss).mean()))
    print("Epoch {} - Validation loss: {}".format(epoch, torch.tensor(val_loss).mean()))
    writer.add_scalar("Loss/train", torch.tensor(train_loss).mean(), epoch)
    writer.add_scalar("Loss/validation", torch.tensor(val_loss).mean(), epoch)

    # Save the model
    save_model(epoch, torch.tensor(train_loss).mean(), torch.tensor(val_loss).mean())

    # Save the best model
    if torch.tensor(val_loss).mean() < best_val_loss:
        best_val_loss = torch.tensor(val_loss).mean()
        best_model_idx = epoch

# Save the index of the best model
save_best_model_idx(best_model_idx, best_val_loss)

# Load the best model
load_best_model(best_model_idx)

# TODO - Load best model and plot it 

    # if epoch >= 1000:

    #     print("----")

    #     for i in range(len(train_dataset)):

    #         x, y = train_dataset[i]

    #         y_hat = model(x[None, :, :], target_time)
            
    #         print(y_hat.shape)
    #         time = torch.arange(0, x.shape[0] + y.shape[0]).numpy(force=True)

    #         # Plot the resulting prediction
    #         plt.figure()
    #         plt.plot(time, torch.cat((x[:, 0], y[:, 0]), dim=0).numpy(force=True), label="x")        # x
    #         plt.plot(time, torch.cat((x[:, 1], y[:, 1]), dim=0).numpy(force=True), label="y")        # y
    #         plt.plot(time, torch.cat((x[:, 2], y[:, 2]), dim=0).numpy(force=True), label="z")        # z

    #         time2 = torch.arange(x.shape[0], x.shape[0] + y_hat.shape[1]).numpy(force=True)
    #         plt.plot(time2, y_hat[0, :, 0].numpy(force=True), label="x_hat")        # x
    #         plt.plot(time2, y_hat[0, :, 1].numpy(force=True), label="y_hat")        # y
    #         plt.plot(time2, y_hat[0, :, 2].numpy(force=True), label="z_hat")        # z

    #         plt.legend()
    #         plt.show()

