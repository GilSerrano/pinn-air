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
    
    def __init__(self, dataset_path, split="train", lookback=1, device="cpu"):
        """Initializes the Simulation dataset class

        Args:
            dataset_path (str): Path to the dataset
            split (str): If the split is the training, validation, or test split
            lookback (int): Size of window for prediction
            device (str, optional): _description_. Defaults to "cpu".
        """

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

        # Load the data from the files
        for name in tqdm(name_list):
            try:
                # Load the data from the file
                timeseries = np.load(pjoin(dataset_path, 'data', name))
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
        timeseries = self.dataset[self.name_list[index]]

        # Create a tensor with the time, p, v, a, attitude, w, p_ref, v_ref, a_ref, attitude_ref, w_ref, T_ref, M_ref of shape (37, seq_len)
        #dataset = torch.cat([timeseries[key] for key in timeseries.keys()], dim=-1).swapaxes(0, 1)

        # Only get the states that we really care about (x[k]=[p,v,R], u[k]=[w_ref, T_ref])
        dataset = torch.cat([timeseries["p"], timeseries["v"], timeseries["attitude"], timeseries["w_ref"], timeseries["T_ref"]], dim=-1)

        # Create the feature and target timeseries using the desired lookback
        x, y = [], []

        for i in range(dataset.shape[0] - self.lookback):
            
            feature = dataset[i:i+self.lookback,:]                # We need everything for the input of the network (current state and input of the system)
            target = dataset[i+1:i+self.lookback+1, 0:10]         # We only need at the output the p[k+1], v[k+1], R[k+1] (3+3+4)

            x.append(feature)
            y.append(target)

        # Convert back the list of tensors to a single tensor
        return torch.cat(x, 0), torch.cat(y, 0)

    
    @staticmethod
    def collate(batch, device):
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

        # Get the timeseries from the batch
        for data in batch:
            datasets += [data[0]]
            expected_outputs += [data[1]]

        # Return a dataset of dimensions (batch_size, max_seq_len, 13), (batch_size, max_seq_len, 10)
        return pad_sequence(datasets, batch_first=True, padding_value=0.0).to(device), pad_sequence(expected_outputs, batch_first=True, padding_value=0.0).to(device)
    

class SimCircles(SimDataset):
    """"
    A wrapper class for the sim_circles dataset
    """
    
    def __init__(self, dataset_path='', split="train", lookback=1, device="cpu"):

        # If no path is given, use the default path for the dataset
        if dataset_path == '':
            dataset_path='./dataset/sim_circles'
            
        print('Loading dataset: ' + dataset_path)

        # Perform the actual initialization of the dataset
        super().__init__(dataset_path, split, lookback, device)

        # Check if the dataset is empty
        assert len(self.dataset) >= 1, 'You loaded an empty dataset, '


if __name__ == "__main__":

    dataset = SimCircles(lookback=1)

    batch = [dataset[0], dataset[1]]
    print(dataset.collate(batch)[1].shape)