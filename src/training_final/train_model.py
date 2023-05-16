#!/usr/bin/env python3
import torch
import random
import numpy as np

# Import our custom libraries here
from trainer import Trainer
from model import SuperModelo
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
    torch.set_default_device(device)
    print("Using device: {}".format(device))

    # Create the argument parser
    parser = ArgsParser()

    # Fix the seed for reproducibility
    fix_seed(parser.args.seed)

    # Set the sampling rate and the mass of the vehicle
    Ts = 0.03       # seconds
    mass = 1.0      # kilograms

    # Set the time windows for the input/output data
    input_window = 50      # seconds
    output_window = 25     # seconds

    # Create the physics model
    vehicle_model = DiscreteMultirotor(Ts, mass, device)

    # Create the network model
    model = SuperModelo()

    # Create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=parser.args.learning_rate, weight_decay=parser.args.weight_decay)

    # Use as criterion the loss function of the model
    criterion = model.compute_loss

    # ---------------------------------
    # Load the datasets for training
    # ---------------------------------
    train_dataset = MocapDatasetLoader(input_window=input_window, output_window=output_window, stride=1, split="train", device="cuda")
    validation_dataset = MocapDatasetLoader(input_window=input_window, output_window=output_window, stride=1, split="val", device="cuda")

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
        output_dir=parser.args.output_path, 
        teacher_forcing_ratio=parser.args.teacher_forcing_ratio, 
        device=device
    )

    # Train the model
    trainer.train_multiple_epochs(num_epochs=parser.args.num_epochs)

if __name__ == "__main__":
    main()