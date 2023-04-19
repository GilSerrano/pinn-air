import torch
import numpy as np
import codecs as cs
from tqdm import tqdm
from torch.utils import data
from os.path import join as pjoin

# Used for padding the timeseries
from torch.nn.utils.rnn import pad_sequence

class RealMocap(data.Dataset):
    """
    Base class for real datasets acquired in the Taguspark MOCAP arena
    with drone + payload 

    Args:
        SimDataset (_type_): _description_
    """
    def __init__(self, dataset_path='', split="train", lookback=1, device="cpu"):
        """Initializes the Simulation dataset class

        Args:
            dataset_path (str): Path to the dataset
            split (str): If the split is the training, validation, or test split
            lookback (int): Size of window for prediction
            device (str, optional): _description_. Defaults to "cpu".
        """
        
        # If not path if given, use the default path for the dataset
        if dataset_path == '':
            dataset_path='./dataset/mocap_14_04_2023'
        
        print('Loading dataset: ' + dataset_path)

        # Define the lookback size and the device where the data will be stored
        self.lookback = lookback
        self.device = device

        # List of the files to use for this dataset
        name_list = []

        # Path to the file containing the list of files to use for this dataset (train, test, or val)
        split_file = pjoin(dataset_path, f'{split}.txt')

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

        print(self.length_list)

        # Convert the lengths list to a numpy array
        self.length_list = np.array(self.length_list)

        # Check if the dataset is empty
        assert len(self.dataset) >= 1, 'You loaded an empty dataset, '

    @staticmethod
    def collate(batch):
        '''
        Method used to collate the data from the dataset into batches
        Args:
            batch (list): A list of dicts containing the timeseries to compose the batch
        Returns:
            tuple(nn.Tensor, nn.Tensor): A tuple containing:
                - The batch of timeseries with input of the network 
                - The batch of the timeseries with the expected output
        '''

        # Each timeseries contains: 
        # time (dim=1),                                     # time of the system
        # p (dim=3), v (dim=3), a (dim=3),                  # position velocity and acceleration of the system
        # attitude (dim=4), w (dim=3),                      # attitude quaternion and angular velocity of the system
        # p_ref (dim=3), v_ref (dim=3), a_ref (dim=3),      # reference position, velocity and acceleration
        # attitude_ref (dim=4), w_ref (dim=3),              # reference attitude quaternion and angular velocity
        # T_ref (dim=1), M_ref (dim=3)                      # reference thrust and moment

        # In this function we want to create a tensor of dimensions (batch_size, max_seq_len, 13)
        datasets = []
        expected_outputs = []

        print(data[0])

        # Get the timeseries from the batch
        for data in batch:
            datasets += [data[0]]
            expected_outputs += [data[1]]

        # Return a dataset of dimensions (batch_size, max_seq_len, 13), (batch_size, max_seq_len, 10)
        return pad_sequence(datasets, batch_first=True, padding_value=0.0), pad_sequence(expected_outputs, batch_first=True, padding_value=0.0)
