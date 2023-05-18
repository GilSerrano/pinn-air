#!/usr/bin/env python3

from argparse import ArgumentParser

class ArgsParser:
    def __init__(self):    
        # Create the parser
        self.parser = ArgumentParser()

        # Base options for training
        train_group = self.parser.add_argument_group('training')
        train_group.add_argument("--seed", default=0, type=int, help="Random seed.")
        train_group.add_argument("--batch_size", default=64, type=int, help="Batch size.")
        train_group.add_argument("--learning_rate", default=1E-4, type=float, help="Learning rate.")
        train_group.add_argument("--weight_decay", default=0.05, type=float, help="Weight decay.")
        train_group.add_argument("--num_epochs", default=600, type=int, help="Number of epochs.")
        train_group.add_argument("--teacher_forcing_ratio", default=0.0, type=float, help="Teacher forcing ratio.")
        train_group.add_argument("--teacher_forcing_decay", default=2.0, type=float, help="Teacher forcing decay.")

        # Data augmentation, default is to use data augmentation
        train_group.add_argument("--data_augmentation", dest='data_augmentation', action='store_false', help="Use data augmentation.")
        self.parser.set_defaults(data_augmentation=False)
        train_group.add_argument("--augmentation_low", default=-10.0, type=float, help="Lower bound for the augmentation.")
        train_group.add_argument("--augmentation_high", default=10.0, type=float, help="Upper bound for the augmentation.")

        # Options for the model
        model_group = self.parser.add_argument_group('model')
        model_group.add_argument("--hidden_dim", default=512, type=int, help="Dimension of the hidden state.")
        model_group.add_argument("--dropout", default=0.2, type=float, help="Dropout rate.")

        # Options for the location of the data
        data_group = self.parser.add_argument_group('data')
        data_group.add_argument("--data_path", default='data', type=str, help="Path to the data.")
        data_group.add_argument("--output_path", default='output', type=str, help="Path to the output directory.")

        # Extras for answering the questions in the report
        extras_group = self.parser.add_argument_group('extras')
        extras_group.add_argument("--use_attention", dest='use_attention', action='store_true', help="Use attention in the model.")
        self.parser.set_defaults(use_attention=False)

        self.args = self.parser.parse_args()