#!/usr/bin/env python
import os
import torch
from models import nn_models
from data_loaders import get_dataset_loader
from train.train_loop import TrainLoop
from utils import fix_seed, train_args, check_save_directory

def main():
    """
    Train a model to learn the dynamics of the system using simulation data
    """

    # parse arguments
    args = train_args()

    # Fix the random seed
    fix_seed(args.seed)

    # Import the data loaders
    check_save_directory(args)

    # Check the device
    args.device = 'cuda:0' if args.cuda and torch.cuda.is_available() else 'cpu'

    # Load the dataset
    # TODO - remove this hardcode from the dataset
    args.dataset = "sim_circles"
    args.data_dir = os.path.abspath("./dataset")

    train_dataloader = get_dataset_loader(args.dataset, batch_size=args.batch_size, datapath=args.data_dir, split="train", device=args.device)
    validation_dataloader = get_dataset_loader(args.dataset, batch_size=args.batch_size, datapath=args.data_dir, split="val", device=args.device)
    test_dataloader = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, datapath=args.data_dir, split="test", device=args.device)

    # Create the model
    model = nn_models["rnn_autoencoder"](
        input_dim=10,               # TODO - change this dimension
        layers=[128, 64, 32], 
        latent_dim=10, 
        activation="relu",
        device=args.device
    )

    # Print the total number of parameters of the model
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    # Train the model
    print("Training...")
    training_loop = TrainLoop(args, model, train_dataloader, validation_dataloader)
    training_loop.train()

    # Test the model
    print("Testing...")
    test_loop = training_loop.test()



if __name__ == "__main__":
    main()