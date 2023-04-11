#!/usr/bin/env python
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
    if args.cuda and torch.cuda.is_available():
        args.device = 'cuda:0'

    # Load the dataset
    data = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, datapath=args.data_dir, split="train", device=args.device)

    # Create the model
    model = nn_models["rnn"](
        input_dim=data.dataset.input_dim, 
        layers=[128, 64, 32], 
        latent_dim=10, 
        activation="relu"
    ).to(args.device)

    # Print the total number of parameters of the model
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0))

    # Train the model
    print("Training...")
    TrainLoop(args, model, data).run_loop()


if __name__ == "__main__":
    main()