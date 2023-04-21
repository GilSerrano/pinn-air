#!/usr/bin/env python
import os
import torch
from train.train_loop import TrainLoop
from models.rnn_autoencoder import RNNAutoencoder
from dynamics.discrete_multirotor import DiscreteMultirotor

from data_loaders import get_dataset_loader
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

    # Check the device and set the default
    args.device = 'cuda:0' if args.cuda and torch.cuda.is_available() else 'cpu'
    torch.set_default_device(args.device)
    
    print("Using device: ", args.device)

    # Load the dataset
    # --------------------------------------------
    # TODO - remove this hardcode from the dataset
    # --------------------------------------------
    args.dataset = "mocap_14_04_2023"
    args.data_dir = os.path.abspath("./dataset")

    lookback=5
    pooled_classification=True      # Only predict the last sample of a sequence, given the sequence of inputs

    train_dataloader = get_dataset_loader(args.dataset, batch_size=args.batch_size, datapath=args.data_dir, split="train", lookback=lookback, pooled_classification=pooled_classification, device=args.device)
    validation_dataloader = get_dataset_loader(args.dataset, batch_size=args.batch_size, datapath=args.data_dir, split="val", lookback=lookback, pooled_classification=pooled_classification, device=args.device)
    test_dataloader = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, datapath=args.data_dir, split="test", lookback=lookback, pooled_classification=pooled_classification, device=args.device)

    # NOTES: Input of the network  (x[k]=[p,v,R], u[k]=[w_ref, T_ref]) (14,)
    #        Output of the network (x[k+1]=[p,v,R]) (10,)
    
    # Create the multirotor model (used in the loss function to learn the known physics of the model
    multirotor_model = DiscreteMultirotor(Ts=0.01, mass=1.5, device=args.device) # System sampling period (s), Mass of the vehicle (without payload) Kg

    # Create the RNN autoencoder model
    model = RNNAutoencoder(
        input_dim=17,                           # Size of the input dimensions = (x[k]=[p,v,R], u[k]=[w_ref, T_ref], x_payload=[p]) (17,)
        output_dim=13,                          # Size of the output dimension = (x[k+1]=[p,v,R], x_payload[k+1]=[p]) (13,)
        layers=[128, 64, 32, 16],               # Sizes of the encoder-decoder layers
        latent_dim=10,                          # Latent dimension going inside the RNN layers
        dropout=0.2,                            # Dropout to be used
        activation="relu",                      # The activation function to be used
        system_model=multirotor_model.run,      # Method used to model the multirotor dynamics
        device=args.device                      # The device to which we should send the model
    )

    # Print the total number of parameters of the model
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    # Train the model
    print("Training...")
    training_loop = TrainLoop(args, model, train_dataloader, validation_dataloader, test_dataloader)
    training_loop.train()

    # Check what was the best model
    print("Best model obtained at epoch: ", training_loop.best_model_idx)
    
    # Load the best model
    training_loop.load_best_model()

    # Test the model
    print("Testing...")
    test_mse = training_loop.test()
    print("Test loss: ", test_mse)

    # Test the model predicitions on a timeseries
    print("Testing on a timeseries...")
    

if __name__ == "__main__":
    main()
