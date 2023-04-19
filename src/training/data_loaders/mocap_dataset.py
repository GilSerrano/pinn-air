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

    It is assumed that each timeseries contains: 
    t_sampling (float)                                # sampling period of the system
    time (dim=1),                                     # time of the system
    p (dim=3), v (dim=3), a (dim=3),                  # position velocity and acceleration of the system
    attitude (dim=4), w (dim=3),                      # attitude quaternion and angular velocity of the system
    p_ref (dim=3), v_ref (dim=3), a_ref (dim=3),      # reference position, velocity and acceleration
    w_ff (dim=3)                                      # reference angular velocity computed directly only from the trajectory based on the reference jerk (not applied directly)
    attitude_ref (dim=4), w_ref (dim=3),              # reference attitude quaternion and angular velocity for the vehicle to track
    T_ref (dim=1)                                     # reference thrust
    p_load (dim=3), attitude_load(dim=3)              # position and attitude of the payload
    """

    def __init__(self, dataset_path='', split="train", lookback=5, pooled_classification=True, device="cpu"):
        """Initializes the Simulation dataset class

        Args:
            dataset_path (str): Path to the dataset
            split (str): If the split is the training, validation, or test split
            lookback (int): Size of window of samples x[0]...  x[lookback-1] to give to the network for it to give predictions upon
            pooled_classification (bool): Wether we expect the network to output a classification or a regression. I.e. if we should only produce y[lookback] or y[1]...y[lookback]
            device (str, optional): _description_. Defaults to "cpu".
        """
        
        # If not path if given, use the default path for the dataset
        if dataset_path == '':
            dataset_path='./dataset/mocap_14_04_2023'
        
        print('Loading dataset: ' + dataset_path)

        # Define the lookback size and the device where the data will be stored
        self.lookback = lookback
        self.pooled_classification = pooled_classification
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

        # Convert the lengths list to a numpy array
        self.length_list = np.array(self.length_list)

        # Check if the dataset is empty
        assert len(self.dataset) >= 1, 'You loaded an empty dataset, '

        # Now we have a set of large sequences of data. We need to break them into smaller sequences of sizes
        # Break each sequence into smaller sequences of length lookback:
        # (input of network)  -> x (batch_size, lookback, 17)
        # (target of network) -> y (batch_size, 1, 10) or (batch_size, lookback, 10) depending on the pooled_classification flag
        x, y = [], []

        for timeseries in self.dataset.values():
            
            # Get only the features that we care about
            # The states that we really care about (x[k]=[p,v,R], u[k]=[w_ref, T_ref], x_payload[k]=[p] )
            #         Dimension:        3       ,        3       ,          4            ,        3           ,        1           ,        3       
            dataset = torch.cat([timeseries["p"], timeseries["v"], timeseries["attitude"], timeseries["w_ref"], timeseries["T_ref"], timeseries["p_load"]], dim=-1).to(self.device)

            # Break the timeseries into smaller sequences of length lookback
            for i in range(dataset.shape[0] - self.lookback):
                
                # Get the input and target of the network
                x.append(dataset[i:i + self.lookback, :])

                # Note: for the output of the network, we only want the state of the vehicle and the payload (we should not have w_ref and T_ref)
                # If we are doing pooled classification, we only need to get the last element of the sequence
                if self.pooled_classification:
                    y.append(torch.cat([dataset[i + self.lookback, 0:10], dataset[i + self.lookback, 14:17]], dim=-1))
                # Otherwise, we need to get the whole sequence shifted by one unit
                else:
                    y.append(torch.cat([dataset[i + 1:i + self.lookback + 1, 0:10], dataset[i + 1:i + self.lookback + 1, 14:17]], dim=-1))

        # Convert the list of tensors to a tensor (these are the inputs and targets of the network)
        self.x = torch.stack(x).to(self.device)
        self.y = torch.stack(y).to(self.device)

        assert self.x.shape[0] == self.y.shape[0], 'The number of inputs and targets must be the same'

        # If we are doing a pooled classification, we need to reshape the target tensor
        if self.pooled_classification:
            self.y = self.y[:,None,:]

        # Set the total number of sequences in the dataset
        self.num_sequences = self.x.shape[0]

    def __len__(self):
        """Returns the length of the dataset."""
        return self.num_sequences
    
    def __getitem__(self, index):
        """Returns the data for the given index.
        Args:
            index (int): The index of the data to be returned.
        Returns:
            tuple(nn.Tensor): A tupple containing the data for the given index (input, target)

        Note: The input is a tensor of shape (lookback, 17) and the target is a tensor of shape (lookback, 10)

            Content of the input tensor:
            x[k] = [p[k], v[k], R[k], w_ref[k], T_ref[k], p_load[k]]

            Content of the target tensor:
            y[k] = [p[k+1], v[k+1], R[k+1], w_ref[k+1], T_ref[k+1], p_load[k+1]]
        
        """
        # Convert back the list of tensors to a single tensor
        return self.x[index, :, :], self.y[index, :, :]

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

        # In this function we want to create a tensor of dimensions (batch_size, max_seq_len, 13)
        datasets = []
        expected_outputs = []

        # Get the timeseries from the batch
        for data in batch:

            datasets += [data[0]]
            expected_outputs += [data[1]]

        # Return a dataset of dimensions (batch_size, max_seq_len, 13), (batch_size, max_seq_len, 10)
        return torch.stack(datasets, dim=0).to(device), torch.stack(expected_outputs, dim=0).to(device)
        

if __name__ == "__main__":

    dataset = RealMocap(lookback=5, pooled_classification=False, device="cuda")

    # Get the first 2 samples from the dataset to generate a batch
    batch_input = [dataset[0], dataset[1]]

    x1,y1 = dataset[0]
    x2,y2 = dataset[1]

    # Collate the batch
    batch = dataset.collate(batch_input, device="cuda")