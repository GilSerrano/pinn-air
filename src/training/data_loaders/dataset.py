import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import codecs as cs
from tqdm import tqdm


class SimDataset(data.Dataset):
    '''
    Base class for the simulation dataset, other datasets based in simulation can inherit from this class,
    or include it in their parameters
    '''
    
    def __init__(self, dataset_path, split, device="cpu"):

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
                    timeseries_torch[key] = torch.from_numpy(value).float().to(device)

                # Add the data to the dataset
                self.dataset[name] = timeseries_torch

                # Get the length of the sequences in these timeseries
                length_list.append(self.dataset[name]['time'].shape[0])

            except:
                print("DEBUG - error in loading {}".format(name))

        # Sort the data by length
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

        # Get the name of the timeserires corresponding to the desired index
        timeseries = self.dataset[self.name_list[index]]

        # Return the individual tensors with data
        return tuple(timeseries[key] for key in timeseries.keys())
    
    @staticmethod
    def collate(batch):
        '''
        Method used to collate the data from the dataset into batches
        '''
        
        adapted_batch = [{
            'inp': torch.tensor(b[4].T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
            'text': b[2], #b[0]['caption']
            'tokens': b[6],
            'lengths': b[5],
            'babel_text': b[7],
        } for b in batch]
        return collate(adapted_batch)
    

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
