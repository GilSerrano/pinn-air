#!/usr/bin/env python
import os
import torch
from torch.utils.data import DataLoader
from train.train_loop2 import TrainLoop2
from models.encoder_decoder import EncoderDecoder

from data_loaders import get_dataset_loader
from data_loaders.mocap_full_sequences_dataset import MocapSwipeLoader
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

    loader = DataLoader(
        TrainLoop2,                                              # The dataset itself
        batch_size=args.batch_size,                              # The size of each batch
        shuffle=False,                                           # Whether to shuffle the sequences on the dataset
        num_workers=0,                                           # The number of cpu to use to load the dataset and generate the batches in parallel
        drop_last=False,                                         # Drop the last samples if not enough to make a batch of the desired size
        collate_fn=lambda x: TrainLoop2.collate(x, args.device), # Custom collate function for the dataset
        multiprocessing_context=None
    )

    # Create the model
    model = EncoderDecoder(input_size=17, hidden_size=20, target_size=14)


if __name__ == "__main__":
    main()

