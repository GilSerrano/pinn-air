import torch
import numpy as np
import codecs as cs
from tqdm import tqdm
from torch.utils import data
from os.path import join as pjoin


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

        print(split_file)

        # split file has the IDs of the sequences to be used for training or testing
        with cs.open(split_file, 'r') as f:
            name_list += [line.strip() for line in f.readlines()]

        # A list with the sequence of lengths of the sequences in the dataset
        length_list = []
        self.dataset = {}

        print(name_list)

        # Load the data from the files
        for name in tqdm(name_list):
            try:

                # Load the data from the file
                timeseries = np.load(pjoin(dataset_path, 'data', name))
                timeseries_torch = {}

                print("----")
                print(timeseries[0])
                print("---")

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

        # Check if the dataset is empty
        assert len(self.dataset) >= 1, 'You loaded an empty dataset, '

if __name__ == "__main__":

    dataset = RealMocap(lookback=1)

    batch = [dataset[0], dataset[1]]
    print(dataset.collate(batch)[1].shape)