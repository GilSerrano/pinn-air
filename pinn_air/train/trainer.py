#!/usr/bin/env python3
"""
| File: trainer.py
| Authors: Gil Serrano, Marcelo Jacinto, Jose Gomes and Joao Pinto
| License: BSD-3-Clause. Copyright (c) 2023, Gil Serrano, Marcelo Jacinto, Jose Gomes and Joao Pinto. All rights reserved.
| Description: Trainer class to train the neural network model.
"""
import os
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from math import exp
from torch.utils.tensorboard import SummaryWriter

class Trainer:

    def __init__(self, model, system_state_dim, system_control_dim, train_dataloader, eval_dataloader, optimizer, criterion, output_dir, teacher_forcing_ratio, teacher_forcing_decay, device):
        """
        Implement this function to train the model for one epoch.

        Args:
            model (nn.Module): model to train
            system_state_dim (int): dimension of the system state
            system_control_dim (int): dimension of the system control
            train_batch (DataLoader): training data loader
            loss (nn.Module): loss function
            optimizer (torch.optim): optimizer for the model parameters
            teacher_forcing_ratio (float): probability of using teacher forcing
            teacher_forcing_decay (float): decay of the teacher forcing ratio
            device (torch.device): device to use for training
        """
        
        # Store the model
        self.model = model.to(device)

        # Dimensions of the system we are trying to approximate
        self.system_state_dim = system_state_dim
        self.system_control_dim = system_control_dim

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

        # Check if the output path already exists, if not, create it
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Store training metrics
        self.train_epoch_loss = []
        self.val_epoch_loss = []

        # Other individual loss metrics
        self.train_epoch_individual_losses = {}
        self.val_epoch_individual_losses = {}

        # Store the best parameters
        self.best_epoch = 0

        self.initial_epoch = 0

    def train_multiple_epochs(self, num_epochs):
        """
        Implement this function to train the model for multiple epochs
        """

        # Check if there is an already pre-trained model in the output directory
        if os.path.exists(os.path.join(self.output_dir, "best_epoch.txt")):
            with open(os.path.join(self.output_dir, "best_epoch.txt"), "r") as f:
                self.best_epoch = int(f.read())
            print(f"Loading the best model parameters found at epoch {self.best_epoch}")
            checkpoint = torch.load(os.path.join(self.output_dir, f"epoch_{self.best_epoch}_best_model.pt"))
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.train_epoch_loss = [checkpoint['train_loss']]
            self.val_epoch_loss = [checkpoint['val_loss']]
            self.initial_epoch = checkpoint['epoch'] + 1
            print(f"Best model parameters found at epoch {self.best_epoch} with validation loss {checkpoint['val_loss']} and training loss {checkpoint['train_loss']}")

        # Train for a given number of epochs
        for epoch in range(self.initial_epoch, self.initial_epoch + num_epochs):

            train_losses = []
            individual_losses = {}
            
            # Get the batches of training data
            for input_batch, target_batch in tqdm(self.train_dataloader, desc=f"Epoch {epoch}/{self.initial_epoch + num_epochs}"):
                
                # Move the batch to the device
                input_batch = input_batch.to(self.device)
                target_batch = target_batch.to(self.device)

                # Train the model and save the loss value
                train_loss, train_individual_losses = self.train_batch(input_batch, target_batch, epoch)

                # Append the loss to the list
                train_losses.append(train_loss)

                # Append the individual losses to the dictionary
                for key in train_individual_losses:
                    if key not in individual_losses:
                        individual_losses[key] = []
                    individual_losses[key].append(train_individual_losses[key])

            # Compute the mean of the loss up to this point and append to the list
            self.train_epoch_loss.append(torch.tensor(train_losses).mean().item())

            # Compute the mean of the individual losses up to this point and append to the list
            for key in individual_losses:
                if key not in self.train_epoch_individual_losses:
                    self.train_epoch_individual_losses[key] = []
                self.train_epoch_individual_losses[key].append(torch.tensor(individual_losses[key]).mean().item())

            # Add the loss to the tensorboard
            self.writer.add_scalar("Loss/train", torch.tensor(train_losses).mean().item(), epoch)

            # Add the individual losses to the tensorboard
            for key in individual_losses:
                self.writer.add_scalar(f"Loss/train_{key}", torch.tensor(individual_losses[key]).mean().item(), epoch)

            # Compute the validation loss
            eval_loss, eval_individual_loss = self.compute_validation_loss()
            self.writer.add_scalar("Loss/validation", eval_loss, epoch)

            # Add the individual losses to the tensorboard
            for key in eval_individual_loss:
                self.writer.add_scalar(f"Loss/validation_{key}", eval_individual_loss[key], epoch)

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
                    f.write(f"\n{epoch}")
                
            # Save the validation losses over time
            self.val_epoch_loss.append(eval_loss)

            print(f"Epoch {epoch}/{self.initial_epoch + num_epochs} - Train loss: {self.train_epoch_loss[-1]} - Validation loss: {self.val_epoch_loss[-1]}")

        # Load the best model parameters
        checkpoint = torch.load(os.path.join(self.output_dir, f"epoch_{self.best_epoch}_best_model.pt"))
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Best model parameters found at epoch {self.best_epoch} with validation loss {checkpoint['val_loss']}")


    def compute_validation_loss(self):

        self.model.eval()
        eval_loss = []
        eval_individual_loss = {}

        counter=0

        with torch.no_grad():

            # Iterate over the validation set
            for input_batch, target_batch in self.eval_dataloader:

                # Move the batch to the device
                input_batch = input_batch.to(self.device)
                target_batch = target_batch.to(self.device)

                # The state of the system can either be:
                # Generate the input of the system (u=[w_x, w_y, w_z, T]) from the target_batch vector
                x = input_batch
                y = target_batch

                u = target_batch[:, :, target_batch.shape[-1]-self.system_control_dim:target_batch.shape[-1]]

                target_time = y.shape[1]

                # Get the output sequence from the model
                if x.shape[-1] == 23:
                    x_simple = torch.cat([x[:, :, 0:13], x[:, :, 19:23]], dim=-1)
                    y_hat = self.model(x_simple, u, target_time)
                else:
                    y_hat = self.model(x, u, target_time)

                # Compute the loss
                loss, loss_elements = self.criterion(y_hat, y, x)

                # Check if the loss is Nan
                if torch.isnan(loss):
                    print(counter)

                counter += 1

                # Compute the loss
                eval_loss.append(loss)

                # Compute the individual losses
                for key in loss_elements:
                    if key not in eval_individual_loss:
                        eval_individual_loss[key] = []
                    eval_individual_loss[key].append(loss_elements[key])

            # Compute the mean of the individual loss terms
            for key in eval_individual_loss:
                eval_individual_loss[key] = torch.tensor(eval_individual_loss[key]).mean().item()

        return torch.tensor(eval_loss).mean().item(), eval_individual_loss

    def train_batch(self, input_batch, target_batch, epoch):

        # Set the model to train mode
        self.model.train()

        with torch.enable_grad():

            # Zero the gradients
            self.optimizer.zero_grad()

            # Generate the input of the system (u=[w_x, w_y, w_z, T]) from the target_batch vector
            x = input_batch
            y = target_batch
            
            u = target_batch[:, :, target_batch.shape[-1]-self.system_control_dim:target_batch.shape[-1]]
            
            # Compute the teacher forcing ratio, using an exponential decay
            tf_ratio = self.teacher_forcing_ratio * exp(-epoch * self.teacher_forcing_decay)

            # Get the output sequence from the model
            if x.shape[-1] == 23:
                x_simple = torch.cat([x[:, :, 0:13], x[:, :, 19:23]], dim=-1)
                y_simple = torch.cat([y[:, :, 0:13], y[:, :, 19:23]], dim=-1)
                y_hat = self.model(x_simple, u, target_y=y_simple, teacher_forcing_ratio=tf_ratio)
            else:
                y_hat = self.model(x, u, target_y=y, teacher_forcing_ratio=tf_ratio)

            # Compute the loss
            loss, loss_elements = self.criterion(y_hat, y, x)

            # Update the parameters of the model
            loss.backward()

            self.optimizer.step()

        return loss.item(), loss_elements

    def plot_loss(self):

        # Plot the training and validation loss
        plt.figure(figsize=(5,4))
        plt.plot(self.train_epoch_loss, label='Training loss')
        plt.plot(self.val_epoch_loss, label='Validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, 'training_and_validation_loss.pdf'))
