#!/usr/bin/env python3
import os
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from math import exp
from torch.utils.tensorboard import SummaryWriter

class Trainer:

    def __init__(self, model, physics_model, train_dataloader, eval_dataloader, optimizer, criterion, output_dir, teacher_forcing_ratio, teacher_forcing_decay, device):
        """
        Implement this function to train the model for one epoch.

        Args:
            model (nn.Module): model to train
            physics_model (DiscreteMultirotor): physics model to use
            train_batch (DataLoader): training data loader
            loss (nn.Module): loss function
            optimizer (torch.optim): optimizer for the model parameters
            teacher_forcing_ratio (float): probability of using teacher forcing
            teacher_forcing_decay (float): decay of the teacher forcing ratio
            device (torch.device): device to use for training
        """
        
        # Store the model
        self.model = model.to(device)
        self.physics_model = physics_model

        # Store the training and validation dataloaders
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        # Store the optimizer and criterion and other training parameters
        self.optimizer = optimizer
        self.criterion = criterion
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.teacher_forcing_decay = teacher_forcing_decay

        # Store the device where we want to run the model
        self.device = device

        # Setup a tensorboard summary writer
        self.writer = SummaryWriter(log_dir=output_dir)
        self.output_dir = output_dir

        # Store training metrics
        self.train_epoch_loss = []
        self.val_epoch_loss = []

        # Store the best parameters
        self.best_epoch = 0

    def train_multiple_epochs(self, num_epochs):
        """
        Implement this function to train the model for multiple epochs
        """

        # Train for a given number of epochs
        for epoch in range(num_epochs):

            train_losses = []
            
            # Get the batches of training data
            for input_batch, target_batch in tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                
                # Move the batch to the device
                input_batch = input_batch.to(self.device)
                target_batch = target_batch.to(self.device)

                # Train the model and save the loss value
                train_losses.append(self.train_batch(input_batch, target_batch, epoch))

            # Compute the mean of the loss up to this point and append to the list
            self.train_epoch_loss.append(torch.tensor(train_losses).mean().item())
            self.writer.add_scalar("Loss/train", torch.tensor(train_losses).mean().item(), epoch)

            # Compute the validation loss
            eval_loss = self.comptute_validation_loss()
            self.writer.add_scalar("Loss/validation", eval_loss, epoch)

            # Check if we want to save the model parameters
            if epoch == 0 or eval_loss < min(self.val_epoch_loss):
                self.best_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'model': self.model.state_dict(), 
                    'optimizer': self.optimizer.state_dict(),
                    'train_loss': train_losses[-1],
                    'val_loss': eval_loss}, 
                    os.path.join(self.output_dir, f"epoch_{epoch}_best_model.pt"))
                
                # Write to a txt file the best epoch
                with open(os.path.join(self.output_dir, "best_epoch.txt"), "w") as f:
                    f.write(f"{epoch}")
                
            # Save the validation losses over time
            self.val_epoch_loss.append(eval_loss)

            print(f"Epoch {epoch+1}/{num_epochs} - Train loss: {self.train_epoch_loss[-1]} - Validation loss: {self.val_epoch_loss[-1]}")

        # Load the best model parameters
        checkpoint = torch.load(os.path.join(self.output_dir, f"epoch_{self.best_epoch}_best_model.pt"))
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Best model parameters found at epoch {self.best_epoch} with validation loss {checkpoint['val_loss']}")

    def comptute_validation_loss(self):

        self.model.eval()
        eval_loss = []

        with torch.no_grad():

            # Iterate over the validation set
            for input_batch, target_batch in self.eval_dataloader:

                # Move the batch to the device
                input_batch = input_batch.to(self.device)
                target_batch = target_batch.to(self.device)

                # Generate the input of the system (u=[w_x, w_y, w_z, T]) from the target_batch vector
                x = input_batch
                y = target_batch
                u = y[..., 13:17]

                target_time = y.shape[1]

                # Get the output sequence from the model
                y_hat = self.model(x, u, target_time)

                # Compute the loss
                loss = self.criterion(y_hat, y, x, target_time, self.physics_model)

                # Compute the loss
                eval_loss.append(loss)

        return torch.tensor(eval_loss).mean().item()

    def train_batch(self, input_batch, target_batch, epoch, clip=1):

        # Set the model to train mode
        self.model.train()

        with torch.enable_grad():

            # Zero the gradients
            self.optimizer.zero_grad()

            # Generate the input of the system (u=[w_x, w_y, w_z, T]) from the target_batch vector
            x = input_batch
            y = target_batch
            u = target_batch[:, :, 13:17]

            # Get the target time from the target sequence
            target_time = y.shape[1]

            # Compute the teacher forcing ratio, using an exponential decay
            tf_ratio = self.teacher_forcing_ratio * exp(-epoch * self.teacher_forcing_decay)

            # Get the output sequence from the model
            y_hat = self.model(x, u, target_y=y, teacher_forcing_ratio=tf_ratio)

            # Compute the loss
            loss = self.criterion(y_hat, y, x, target_time, self.physics_model)

            # Update the parameters of the model
            loss.backward()

            # Clip the gradients to avoid exploding gradients
            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)

            self.optimizer.step()

        return loss.item()

    def plot_loss(self):

        # Plot the training and validation loss
        plt.figure(figsize=(5,4))
        plt.plot(self.train_epoch_loss, label='Training loss')
        plt.plot(self.val_epoch_loss, label='Validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, 'training_and_validation_loss.pdf'))
