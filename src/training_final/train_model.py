#!/usr/bin/env python3
import os
import torch
import random
import numpy as np

# Import our custom libraries here
from trainer import Trainer
from model import SuperModelo
from alpha_model import AlphaModel
from omega_model import OmegaModel
from arg_parser import ArgsParser
from vehicle_model import DiscreteMultirotor

# Import the dataset loaders
from torch.utils.data import DataLoader
from mocap_dataset import MocapDatasetLoader

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
    # torch.set_default_device(device)
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
    output = os.path.join(parser.args.output_path, "batch_size_" + str(parser.args.batch_size) + "_lr_" + str(parser.args.learning_rate) + "_wd_" + str(parser.args.weight_decay) + "_epochs_" + str(parser.args.num_epochs) + "_tfr_" + str(parser.args.teacher_forcing_ratio) + "_tfd_" + str(parser.args.teacher_forcing_decay) + "_da_" + str(parser.args.data_augmentation) + "_al_" + str(parser.args.augmentation_low) + "_ah_" + str(parser.args.augmentation_high) + "_use_attention_" + str(parser.args.use_attention) + "_seed_" + str(parser.args.seed)) + "/"

    # Set the sampling rate and the mass of the vehicle (without slung load)
    Ts = 0.03       # seconds
    mass = 1.35     # kilograms (Intel Aero RTF)

    # Set the time windows for the input/output data
    input_window = 50      # seconds
    output_window = 25     # seconds

    # Create the physics model
    vehicle_model = DiscreteMultirotor(Ts, mass, device)

    # Create the network model
    #model = AlphaModel(output_dim=13, num_layers=3, dropout=parser.args.dropout, device=device)
    model = OmegaModel(output_dim=13, num_layers=3, dropout=parser.args.dropout, device=device)
    # model = SuperModelo(device)

    # Set the loss parameters
    model.set_loss_params(
        parser.args.position_error,
        parser.args.velocity_error, 
        parser.args.position_error_payload, 
        parser.args.continuity_last_input_first_output,
        parser.args.output_continuity,
        parser.args.quaternion_norm,
        parser.args.quaternion_error,
        parser.args.physics_error)

    # Create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=parser.args.learning_rate, weight_decay=parser.args.weight_decay)

    # Use as criterion the loss function of the model
    criterion = model.compute_loss

    # ---------------------------------
    # Load the datasets for training
    # ---------------------------------
    train_dataset = MocapDatasetLoader(input_window=input_window, output_window=output_window, stride=1, split="train", device="cuda", data_augmentation=parser.args.data_augmentation, augmentation_low=parser.args.augmentation_low, augmentation_high=parser.args.augmentation_high)
    validation_dataset = MocapDatasetLoader(input_window=input_window, output_window=output_window, stride=1, split="val", device="cuda", data_augmentation=False)

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
        physics_model=vehicle_model,
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