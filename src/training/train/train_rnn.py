# Import the NN models and the datasets
from models import nn_models
from data_loaders import get_dataset_loader
from utils import fix_seed

from utils.parser_util import train_args

import os
import json
import torch

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
        args.device = 'cuda:' + args.device

    # Load the dataset
    data = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, datapath=args.data_dir, split="train", device=args.device)

    # Create the model
    model = nn_models["rnn"](
        input_dim=data.dataset.input_dim, 
        layers=[128, 64, 32], 
        latent_dim=10, 
        activation="relu"
    ).to(args.device)

    # Train the model



    

def check_save_directory(args):

    # Check if the save directory was even passed as an argument
    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    
    # Check if the save directory exists and if it should be overwritten
    elif os.path.exists(args.save_dir) and not args.overwrite:
        raise FileExistsError('save_dir [{}] already exists.'.format(args.save_dir))
    
    # Create the save directory if it doesn't exist
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Save the arguments to the save directory
    args_path = os.path.join(args.save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)


if __name__ == "__main__":
    main()
