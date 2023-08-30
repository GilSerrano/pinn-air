#!/usr/bin/env python3
"""
| File: mocap_dataset.py
| Authors: Gil Serrano, Marcelo Jacinto, Jose Gomes and Joao Pinto
| License: BSD-3-Clause. Copyright (c) 2023, Gil Serrano, Marcelo Jacinto, Jose Gomes and Joao Pinto. All rights reserved.
| Description: Definition of the dataloader for the mocap dataset.
"""
import torch
import numpy as np
import codecs as cs
from tqdm import tqdm
from torch.utils import data
from os.path import join as pjoin, exists

from pinn_air.physics.discrete_model import DiscreteModel

class MocapDatasetLoader(data.Dataset):
    """
    Class used to load the data from the mocap datasets
    """

    def __init__(self, input_window, output_window, stride, dataset_path='', split="test", device="cpu"):
        """
        Initialize the dataset loader.

        Args:
            input_window (int): The number of time steps to use as input for the network.
            output_window (int): The number of time steps to predict with the network.
            stride (int): The number of time steps to skip between each window.
            Ts (float, optional): The sampling time of the mocap dataset. Defaults to 0.03.
            dataset_path (str, optional): The directory of the mocap dataset to use. Defaults to ''.
            split (str, optional): The type of split to use, i.e. train, val or test. Defaults to "test".
            device (str, optional): The device to send the data to. Defaults to "cpu".
        """
        
        # If not path if given, use the default path for the dataset
        if dataset_path == '':
            dataset_path='pinn_air/dataset/mocap_14_04_2023'
        
        print('Loading dataset: ' + dataset_path)

        # Set the paramters of the dataset
        self.Ts = 0.03              # Sampling time of the mocap dataset (30 ms)
        self.mass_quadrotor = 1.35  # Mass of the quadrotor (Kg)
        self.mass_load = 0.100      # Mass of the load (Kg)
        self.l = 0.7                # Length of the cable (m)

        # Create the physics model to generate physics predictions of what the network should output
        self.physics_model = DiscreteModel(Ts=self.Ts, mQ=self.mass_quadrotor, mL=self.mass_load, l=self.l)

        self.input_window = input_window
        self.output_window = output_window
        self.stride = stride

        # Define the device where the data will be stored
        self.device = device

        # List of the files to use for this dataset
        name_list = []

        # Path to the file containing the list of files to use for this dataset (train, test, or val)
        split_file = pjoin(dataset_path, f'{split}.txt')

        # Set the name of the file where the dataset will be saved (if not already saved)
        dataset_file = pjoin(dataset_path, f'data_{split}_{input_window}_{output_window}_{stride}_{self.Ts}.pt')

        # Check if there is already a .pt file containing the dataset
        if exists(dataset_file):
            
            print('Loading dataset from pre-saved .pt file')
            dataset = torch.load(pjoin(dataset_path, f'data_{split}_{input_window}_{output_window}_{stride}_{self.Ts}.pt'))
            
            self.x = dataset['x']
            self.y = dataset['y']
            # self.y_physics = dataset['y_physics']

            # Get the total number of sequences in the dataset
            self.num_sequences = len(self.x)
            
            return

        # split file has the IDs of the sequences to be used for training or testing
        with cs.open(split_file, 'r') as f:
            name_list += [line.strip() for line in f.readlines()]

        # A list with the sequence of lengths of the sequences in the dataset
        length_list = []
        self.dataset = {}

        # Load the data from the files
        for name in tqdm(name_list):
            try:

                # Load the data from the file
                timeseries = np.load(pjoin(dataset_path, 'data', name))
                timeseries_torch = {}

                # Load all the numpy arrays from the timeseries_dict and create the tensors
                for key, value in timeseries.items():
                    
                    # Note: we should not load the t_sampling as a tensor
                    if key != "t_sampling" and key != "__header__" and key != "__version__" and key != "__globals__":
                        timeseries_torch[key] = torch.from_numpy(value).float().to(self.device)

                    # Make sure to create an extra dimensions for the time tensor 
                    # to make it compatible with the other tensors
                    if key == "time":
                        timeseries_torch[key] = timeseries_torch[key].unsqueeze(1)

                # Add the data to the dataset
                self.dataset[name] = timeseries_torch

                # Get the length of the sequences in these timeseries
                length_list.append(self.dataset[name]['time'].shape[0])

            except:
                print("DEBUG - error in loading {}".format(name))

        # Sort the data by length (TODO: check if we still need this. If not, just remove it)
        self.name_list, self.length_list = zip(*sorted(zip(name_list, length_list), key=lambda x: x[1]))

        # Convert the lengths list to a numpy array
        self.length_list = np.array(self.length_list)

        # Check if the dataset is empty
        assert len(self.dataset) >= 1, 'You loaded an empty dataset, '

        # Define the complete sequences with the right features
        # (input of network)  -> x (batch_size, time, state_dim)
        self.x = []             # The input of the network
        self.y = []             # The output of the network + the reference thrust and angular velocity

        counter = 0

        for timeseries in self.dataset.values():
            
            # Get only the features that we care about
            # The states that we really care about (x[k]=[p,v,R, p_load, v_load], u[k]=[w_ref, T_ref])
            #             Dimension:        3

            # Note: the quaternion saved in the npz files are in the standard (x,y,z,w) format
            # but we want to convert to the (w,x,y,z) format
            timeseries["attitude"] = torch.cat([timeseries["attitude"][...,3:4], timeseries["attitude"][...,0:3]], dim=-1)

            # Series is a concatenation of [p, v, R, p_load, v_load, w_load, w_ref, T_ref]
            #                              [3, 3, 4,      3,      3,      3,     3,     1]
            series = torch.cat([timeseries["p"], timeseries["v"], timeseries["attitude"], timeseries["p_load"], torch.zeros_like(timeseries["p_load"]), torch.zeros_like(timeseries["p_load"]), timeseries["w_ref"], timeseries["T_ref"]], dim=-1).to(self.device)
            
            # Compute the velocity of the payload using the discrete difference method (in the inertial frame)
            series[1:, 13:16] = (series[1:, 10:13] - series[:-1, 10:13]) / self.Ts

            # Compute the angular velocity vector of the load, using the equations from the physics model for each individual timestep
            # Note: we use the physics model to compute the angular velocity of the load, because the mocap data does not contain this information
            for t in range(0, series.shape[0]-1):

                # Get the current state of the system
                x = series[t, 0:19]

                # Get the reference input
                u = series[t, 19:23]

                # Compute the angular velocity of the load
                out = self.physics_model.run(x, u)
                
                # Update the real angular velocity of the load in the next timestep of the series
                series[t+1, 16:19] = out[16:19]
            
            # Generate the windows
            x, y = self.window_sequence(series, self.input_window, self.output_window, self.stride)
           
            # Save the mini-batches of the sequence
            self.x.append(x)
            self.y.append(y)

        self.x = torch.cat(self.x, dim=0)
        self.y = torch.cat(self.y, dim=0)

        # Get the total number of sequences in the dataset
        self.num_sequences = len(self.x)

        # Save the torch sequences into a file
        torch.save({"x": self.x, "y": self.y}, dataset_file)


    def window_sequence(self, timeseries, input_window, output_window, stride):
        # ------------------------------------------------------------------------------
        # We must break the sequences into windows:
        # x (time_len, 20) input of the network up until timestep T
        # y (prediction_len, 20) expected prediction from T+1 until T+prediction_len
        # ------------------------------------------------------------------------------
        
        # Get the total length of the sequence
        total_length = timeseries.shape[0]

        # Get the mini-batch size
        size_mini_batch = (total_length - input_window - output_window) // stride + 1

        # Create the sequence that is used as the input of the network
        x = torch.zeros((size_mini_batch, input_window, 23))

        # Create the sequence that is used as the target of the network + the inputs of the system associated with those targets
        y = torch.zeros((size_mini_batch, output_window, 23))
        
        for i in np.arange(size_mini_batch):
            
            # Get the start and end index of the window for the input
            start_x = stride * i
            end_x = start_x + input_window
            
            # Fill the input of the network
            x[i,:,:] = timeseries[start_x:end_x, :]

            # Get the start and end index of the window for the output
            start_y = stride * i + input_window
            end_y = start_y + output_window

            # Fill the target of the network
            y[i,:,:] = timeseries[start_y:end_y, :]

        return x, y

    def __len__(self):
        """Returns the length of the dataset."""
        return self.num_sequences
    
    def __getitem__(self, index):
        """Returns the data for the given index.
        Args:
            index (int): The index of the data to be returned.
        Returns:
            tuple(nn.Tensor): A tupple containing the data for the given index (input, target)

        Note: The input is a tensor of shape (input_window, 20) and the target is a tensor of shape (output_window, 20)

            Content of the tensors:
            x[k] = [p[k], v[k], R[k], p_load[k], v_load[k], w_load[k], w_ref[k], T_ref[k]]
                   [   3,    3,    4,         3,         3,        3,        3,         1]
        
        """

        x = self.x[index].to(self.device)
        y = self.y[index].to(self.device)

        # Note: to train the network we only need: x[k] = [p[k], v[k], R[k], p_load[k], w_ref[k], T_ref[k]]
        #                                          y[k] = [p[k], v[k], R[k], p_load[k], w_ref[k], T_ref[k]]
        #                                          y_physics[k] = [p[k], v[k], R[k], p_load[k]]
        return x, y
        

    @staticmethod
    def collate(batch, device):

        # In this function we want to create a tensor of dimensions (batch_size, max_seq_len, state_dim)
        x = []
        y = []

        # Get the timeseries from the batch
        for data in batch:
            x += [data[0]]
            y += [data[1]]

        return torch.stack(x, dim=0).to(device), torch.stack(y, dim=0).to(device)
        

# ------------------------------
# Used for debugging purposes
# ------------------------------
if __name__ == "__main__":
    input_window = 50      # seconds
    output_window = 25     # seconds

    dataset = MocapDatasetLoader(device="cpu", input_window=input_window, output_window=output_window, stride = 1)
    
    print(len(dataset))

    x, y = dataset[0]

    # Get the first 2 samples from the dataset to generate a batch
    print(x.shape)
    print(y.shape)