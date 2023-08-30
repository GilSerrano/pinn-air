#!/usr/bin/env python3
"""
| File: train_model.py
| Authors: Gil Serrano, Marcelo Jacinto, Jose Gomes and Joao Pinto
| License: BSD-3-Clause. Copyright (c) 2023, Gil Serrano, Marcelo Jacinto, Jose Gomes and Joao Pinto. All rights reserved.
| Description: Main script to train the model.
"""
import os
import torch
import random
import numpy as np

# Import our custom libraries here
from pinn_air.train.trainer import Trainer
from pinn_air.models.pinn_air_model import PINNAirModel
from pinn_air.models.baseline_model import BaselineModel
from pinn_air.utils.arg_parser import ArgsParser

# Import the dataset loaders
from torch.utils.data import DataLoader
from pinn_air.dataset.mocap_dataset import MocapDatasetLoader

def fix_seed(seed):
    """
    Auxiliar function to fix the seed of the random, numpy and torch libraries
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():

    # Set the device for performing training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #torch.set_default_device(device)
    print("Using device: {}".format(device))

    # Create the argument parser
    parser = ArgsParser()

    # Fix the seed for reproducibility
    fix_seed(parser.args.seed)

    # Print the arguments used during training
    print("Seed: {}".format(parser.args.seed))
    print("Batch size: {}".format(parser.args.batch_size))
    print("Learning rate: {}".format(parser.args.learning_rate))
    print("Weight decay: {}".format(parser.args.weight_decay))
    print("Number of epochs: {}".format(parser.args.num_epochs))
    print("Teacher forcing ratio: {}".format(parser.args.teacher_forcing_ratio))
    print("Teacher forcing decay: {}".format(parser.args.teacher_forcing_decay))
    print("Data augmentation: {}".format(parser.args.data_augmentation))
    print("Dropout: {}".format(parser.args.dropout))

    # Create a custom output directory
    output = os.path.join(parser.args.output_path, "batch_size_" + str(parser.args.batch_size) + "_lr_" + str(parser.args.learning_rate) + "_wd_" + str(parser.args.weight_decay) + "_epochs_" + str(parser.args.num_epochs) + "_tfr_" + str(parser.args.teacher_forcing_ratio) + "_tfd_" + str(parser.args.teacher_forcing_decay) + "_use_attention_" + str(parser.args.use_attention) + "_seed_" + str(parser.args.seed)) + "/"

    # Set the time windows for the input/output data
    input_window = 50      # seconds
    output_window = 25     # seconds

    # Dimension of the system state and control input
    system_state_dim = 13
    system_control_dim = 4

    # Create the network model
    if parser.args.model == "PINNAirModel":
        model = PINNAirModel(system_state_dim=system_state_dim, system_control_dim=system_control_dim, num_layers=3, dropout=parser.args.dropout, device=device)

        # Set the loss parameters
        model.set_loss_params(
            parser.args.position_error,
            parser.args.velocity_error, 
            parser.args.position_error_payload, 
            parser.args.continuity_last_input_first_output,
            parser.args.output_continuity,
            parser.args.quaternion_norm,
            parser.args.quaternion_error,
            parser.args.physics_error,
            parser.args.exponential_decay_real,
            parser.args.exponential_decay_physics,
            parser.args.slack_weight)

    elif parser.args.model == "BaselineModel":
        model = BaselineModel(system_state_dim=system_state_dim, system_control_dim=system_control_dim, num_layers=3, dropout=parser.args.dropout, device=device)

        # Set the loss parameters
        model.set_loss_params(
            parser.args.position_error,
            parser.args.velocity_error, 
            parser.args.position_error_payload, 
            parser.args.continuity_last_input_first_output,
            parser.args.output_continuity,
            parser.args.quaternion_norm,
            parser.args.quaternion_error,
            parser.args.physics_error,
            parser.args.exponential_decay_real,
            parser.args.exponential_decay_physics)
        
    else:
        # Throw an error if the model is not recognized
        raise ValueError("Model not recognized. Please use one of the following: PINNAirModel, BaselineModel")

    # Create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=parser.args.learning_rate, weight_decay=parser.args.weight_decay)

    # Use as criterion the loss function of the model
    criterion = model.compute_loss

    # ---------------------------------
    # Load the datasets for training
    # ---------------------------------

    stride = 1

    train_dataset = MocapDatasetLoader(input_window=input_window, output_window=output_window, stride=stride, split="train", device="cuda")
    validation_dataset = MocapDatasetLoader(input_window=input_window, output_window=output_window, stride=stride, split="val", device="cuda")

    train_loader = DataLoader(
               train_dataset,                                           # The dataset itself
               batch_size=parser.args.batch_size,                       # The size of each batch
               shuffle=False,                                           # Whether to shuffle the sequences on the dataset
               num_workers=0,                                           # The number of cuda to use to load the dataset and generate the batches in parallel
               drop_last=False,                                         # Drop the last samples if not enough to make a batch of the desired size
               collate_fn=lambda x: train_dataset.collate(x, device),   # Custom collate function for the dataset
               multiprocessing_context=None)

    validation_loader = DataLoader(
            validation_dataset,                                         # The dataset itself
            batch_size=parser.args.batch_size,                          # The size of each batch
            shuffle=False,                                              # Whether to shuffle the sequences on the dataset
            num_workers=0,                                              # The number of cuda to use to load the dataset and generate the batches in parallel
            drop_last=False,                                            # Drop the last samples if not enough to make a batch of the desired size
            collate_fn=lambda x: validation_dataset.collate(x, device), # Custom collate function for the dataset
            multiprocessing_context=None)

    # ---------------------------------
    # Model Training
    # ---------------------------------

    # Create the trainer for the model
    trainer = Trainer(
        model=model,
        system_state_dim=system_state_dim,
        system_control_dim=system_control_dim,
        train_dataloader=train_loader, 
        eval_dataloader=validation_loader, 
        optimizer=optimizer, 
        criterion=criterion, 
        output_dir=output, 
        teacher_forcing_ratio=parser.args.teacher_forcing_ratio, 
        teacher_forcing_decay=parser.args.teacher_forcing_decay,
        device=device
    )

    # Train the model
    trainer.train_multiple_epochs(num_epochs=parser.args.num_epochs)
    trainer.plot_loss()

if __name__ == "__main__":
    main()