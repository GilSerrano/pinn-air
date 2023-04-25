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
from dynamics.utils import quaternion_multiply, quaternion_invert
from dynamics.discrete_multirotor import DiscreteMultirotor


from torch.utils.tensorboard import SummaryWriter

fix_seed(0)

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

torch.set_default_device("cuda")
model = SuperModelo().to("cuda")
optimizer = AdamW(model.parameters(), lr=1E-4, weight_decay=0.05)
Ts = 0.03
physics_model = DiscreteMultirotor(Ts, 1.0, "cuda")

writer = SummaryWriter(log_dir="output6")

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

epochs = torch.arange(0, 600, 1)


def save_model(epoch, train_loss, val_loss, save_dir="output6/"):

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

def save_best_model_idx(epoch, val_loss, save_dir="output6/"):

    checkpoint_path = os.path.join(save_dir, 'best_model_idx.pth.tar')
    print('Saving checkpoint {}'.format(checkpoint_path))

    # Save the current model
    torch.save({"best_model_idx": best_model_idx, "best_val_loss": best_val_loss,}, checkpoint_path)

def load_best_model(epoch, save_dir="output6/"):
    
    checkpoint_path = os.path.join(save_dir, 'checkpoint_{:04d}.pth.tar'.format(epoch))
    print("Loading: {}".format(checkpoint_path))

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

#def continuity_loss(y_hat, y, target_time):
#    exp_decay = 10 * torch.exp(-0.1*torch.arange(0, target_time, 1)).to("cuda")
#    return torch.sum((torch.norm(y_hat - y, dim=2) ** 2) * exp_decay)

def position_error(y_hat, y, target_time):
    
    # Compute the exponential decay
    exp_decay = 10 * torch.exp(-0.1*torch.arange(0, target_time, 1)).to("cuda")
    
    # Compute the discounted position error
    return torch.sum((torch.norm(y_hat[:, :, 0:3] - y[:, :, 0:3], dim=2) ** 2) * exp_decay)

def position_error_payload(y_hat, y, target_time):
    
    # Compute the exponential decay
    exp_decay = 10 * torch.exp(-0.1*torch.arange(0, target_time, 1)).to("cuda")
    
    # Compute the discounted position error
    return torch.sum((torch.norm(y_hat[:, :, 10:13] - y[:, :, 10:13], dim=2) ** 2) * exp_decay)

def velocity_error(y_hat, y, target_time):

    # Compute the exponential decay
    exp_decay = 10 * torch.exp(-0.1*torch.arange(0, target_time, 1)).to("cuda")

    # Compute the discounted velocity error
    return torch.sum((torch.norm(y_hat[:, :, 3:6] - y[:, :, 3:6], dim=2) ** 2) * exp_decay)

def quaternion_error(y_hat, y, target_time):

    # Compute the exponential decay
    exp_decay = 10 * torch.exp(-0.1*torch.arange(0, target_time, 1)).to("cuda")

    # Compute the discounted quaternion error
    identity_quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0]).to("cuda")

    return torch.sum((torch.norm(quaternion_multiply(y[..., 6:10], quaternion_invert(y_hat[..., 6:10])) - identity_quaternion, dim=2) ** 2) * exp_decay)

def quaternion_norm(y_hat):

    quat_norm = y_hat[..., 6:10].norm(p=2, dim=-1)
    one = torch.ones_like(quat_norm)
    
    return torch.sum((quat_norm - one) ** 2)

def physics_error(y_hat, x, y, target_time, physics_model):

    # Create an empty tensor to store the predictions
    physics_pred = torch.zeros(x.shape[0], target_time, 10).to("cuda")

    # Compute the first prediction of the physical model
    physics_pred[:, 0, :] = physics_model.run(x=x[:, -1, 0:10].unsqueeze(1), u=x[:,-1, 13:17].unsqueeze(1)).squeeze(1)

    # Compute the rest of the predictions of the physical model
    physics_pred[:, 1:target_time, :] = physics_model.run(x=y[:, 0:target_time-1, 0:10], u=y[:, 0:target_time-1, 13:17])

    # Compute the error between the predicted and the physical model
    pos_error = position_error(y_hat, physics_pred, target_time)
    vel_error = velocity_error(y_hat, physics_pred, target_time)
    quat_error = quaternion_error(y_hat, physics_pred, target_time)

    return pos_error + vel_error + quat_error


def continuity_last_input_first_output(y_hat, x):
    return torch.sum(torch.norm(y_hat[:, 0, :] - x[:, -1, 0:13], dim=1) **2, dim=0)

def output_continuity(y_hat):

    total_loss = 0
    for t in range(y.shape[1] - 1):
        total_loss += torch.sum(torch.norm(y_hat[:, t, :] - y_hat[:, t+1, :], dim=1) ** 2, dim=0)

    return total_loss

def compute_loss(y_hat, y, x, target_time, physics_model):

    return 3 * position_error(y_hat, y, target_time) + \
        1 * velocity_error(y_hat, y, target_time) + \
        2 * position_error_payload(y_hat, y, target_time) + \
        2 * continuity_last_input_first_output(y_hat, x) + \
        1 * output_continuity(y_hat) + \
        2 * quaternion_norm(y_hat) + \
        1 * quaternion_error(y_hat, y, target_time) + \
        5 * physics_error(y_hat, x, y, target_time, physics_model)
    

if __name__ == "__main__":
    
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

                # Generate the u from the y vector
                u = y[:, :, 13:17]

                # Perform a forward pass
                y_hat = model(x, u, target_time)

                # --------------------------------------
                # LOSS
                # --------------------------------------
                total_loss = compute_loss(y_hat, y, x, target_time, physics_model)

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

                # Generate the u from the y vector
                u = y[..., 13:17]
                
                # Perform a forward pass
                y_hat = model(x, u, target_time)

                total_val_loss = compute_loss(y_hat, y, x, target_time, physics_model)

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