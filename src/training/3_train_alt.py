#!/usr/bin/env python
import os
import torch
from torch.utils.data import DataLoader
from train.train_loop2 import TrainLoop2
from models.encoder_decoder import EncoderDecoder

from data_loaders.mocap_swipe_loader import MocapSwipeLoader
from utils import fix_seed, train_args, check_save_directory

import matplotlib.pyplot as plt

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
    args.dataset = "mocap_14_04_2023"
    args.data_dir = os.path.abspath("./dataset")

    train_dataset = MocapSwipeLoader(input_window=50, output_window=50, stride=1, split="train", device=args.device)

    loader = DataLoader(
        train_dataset,                                                 # The dataset itself
        batch_size=args.batch_size,                              # The size of each batch
        shuffle=False,                                           # Whether to shuffle the sequences on the dataset
        num_workers=0,                                           # The number of cpu to use to load the dataset and generate the batches in parallel
        drop_last=False,                                         # Drop the last samples if not enough to make a batch of the desired size
        collate_fn=lambda x: train_dataset.collate(x, args.device),    # Custom collate function for the dataset
        multiprocessing_context=None
    )

    # Create the model
    model = EncoderDecoder(input_size=17, hidden_size=20, target_size=13)
    
    # Train the model
    train_loop = TrainLoop2(args, model, loader)
    train_loop.train()

    # ------------------
    # Test the model
    # ------------------
    test_dataset = MocapSwipeLoader(input_window=50, output_window=50, stride=1, split="test", device=args.device)

    loader = DataLoader(
        test_dataset,                                                 # The dataset itself
        batch_size=args.batch_size,                                   # The size of each batch
        shuffle=False,                                                # Whether to shuffle the sequences on the dataset
        num_workers=0,                                                # The number of cpu to use to load the dataset and generate the batches in parallel
        drop_last=False,                                              # Drop the last samples if not enough to make a batch of the desired size
        collate_fn=lambda x: test_dataset.collate(x, args.device),    # Custom collate function for the dataset
        multiprocessing_context=None
    )

    x, y, u = test_dataset[0]
    print(x.shape)
    print(y.shape)
    print(u.shape)

    y_hat = train_loop.model.predict_sequence_recursively(x.unsqueeze(0), u.unsqueeze(0))
    print(y_hat.shape)

    time = torch.arange(0, x.shape[0] + y.shape[0])

    # Plot the resulting prediction
    plt.figure()
    plt.plot(time, torch.cat((x[:, 0], y[:, 0]), dim=0), label="x")        # x
    plt.plot(time, torch.cat((x[:, 1], y[:, 1]), dim=0), label="y")        # y
    plt.plot(time, torch.cat((x[:, 2], y[:, 2]), dim=0), label="z")        # z

    time2 = torch.arange(x.shape[0], x.shape[0] + y_hat.shape[1])
    plt.plot(time2, y_hat[0, :, 0], label="x_hat")        # x
    plt.plot(time2, y_hat[0, :, 1], label="y_hat")        # y
    plt.plot(time2, y_hat[0, :, 2], label="z_hat")        # z

    plt.legend()

    plt.show()
    print(y_hat.shape)




if __name__ == "__main__":
    main()

