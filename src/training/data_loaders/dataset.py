import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import codecs as cs
from tqdm import tqdm

# Used for padding the timeseries
from torch.nn.utils.rnn import pad_sequence

class SimDataset(data.Dataset):
    '''
    Base class for the simulation dataset, other datasets based in simulation can inherit from this class,
    or include it in their parameters
    '''
    
    def __init__(self, dataset_path, split, device="cpu"):

        # Define the device where the data will be stored
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
                timeseries = np.load(pjoin(dataset_path, 'data', name + '.npz'))
                timeseries_torch = {}

                # Load all the numpy arrays from the timeseries_dict and create the tensors
                for key, value in timeseries.items():
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

    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.dataset)

    def __getitem__(self, index):
        """Returns the data for the given index.
        Args:
            index (int): The index of the data to be returned.
        Returns:
            tuple(nn.Tensor): A tupple containing the data for the given index.
        """

        # Get the timeserires corresponding to the desired index
        return self.dataset[self.name_list[index]]
    
    @staticmethod
    def collate(batch):
        '''
        Method used to collate the data from the dataset into batches
        Args:
            batch (list): A list of dicts containing the timeseries to compose the batch
        Returns:
            tuple(nn.Tensor, nn.Tensor): A tuple containing the batch of timeseries with input of the system and the expected output
        '''

        # Each timeseries contains: time (dim=1), p (dim=3), v (dim=3), a (dim=3), attitude (dim=4), w (dim=3), 
        # p_ref (dim=3), v_ref (dim=3), a_ref (dim=3), attitude_ref (dim=3), w_ref (dim=3), T_ref (dim=1), M_ref (dim=3)
        # In this function we want to create a tensor of dimensions (batch_size, 13, max_seq_len)

        datasets = []

        # Get the timeseries from the batch
        for timeseries in batch:

            # Note, this could be improved by using torch.cat with a list of tensors
    
            # Create a tensor with the time, p, v, a, attitude, w, p_ref, v_ref, a_ref, attitude_ref, w_ref, T_ref, M_ref
            # of shape (37, seq_len)
            datasets += [torch.cat([timeseries[key] for key in timeseries.keys()], dim=-1).swapaxes(0, 1)]
            

        # Return a dataset of dimensions (batch_size, 13, max_seq_len)
        return pad_sequence(datasets, batch_first=True, padding_value=0.0)
    

class SimCircles(SimDataset):
    """"
    A wrapper class for the sim_circles dataset
    """
    
    def __init__(self, datapath='', split="train", device="cpu"):

        # If no path is given, use the default path for the dataset
        if datapath == '':
            datapath='./dataset/sim_circles'
            
        print('Loading dataset: ' + datapath)

        # Perform the actual initialization of the dataset
        super().__init__(datapath, split, device)

        # Check if the dataset is empty
        assert len(self.dataset) >= 1, 'You loaded an empty dataset, '


if __name__ == "__main__":

    dataset = SimCircles()

    batch = [dataset[0], dataset[1]]
    print(dataset.collate(batch).shape)