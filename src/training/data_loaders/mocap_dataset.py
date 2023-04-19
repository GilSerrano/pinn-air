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

        # Convert the lengths list to a numpy array
        self.length_list = np.array(self.length_list)

        # Check if the dataset is empty
        assert len(self.dataset) >= 1, 'You loaded an empty dataset, '

    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.dataset)
    
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

        # Get the timeserires corresponding to the desired index
        timeseries = self.dataset[self.name_list[index]]

        # Only get the states that we really care about (x[k]=[p,v,R], u[k]=[w_ref, T_ref], x_payload[k]=[p] )
        #         Dimension:        3       ,        3       ,          4            ,        3           ,        1           ,        3       
        dataset = torch.cat([timeseries["p"], timeseries["v"], timeseries["attitude"], timeseries["w_ref"], timeseries["T_ref"], timeseries["p_load"]], dim=-1)

        # Create the feature and target timeseries using the desired lookback
        x, y = [], []

        for i in range(dataset.shape[0] - self.lookback):
            
            feature = dataset[i:i+self.lookback,:]                      # We need everything for the input of the network (current state and input of the system)

            # Generate the targets for the network to predict
            target = dataset[i+1:i+self.lookback+1, 0:10]               # We only need at the output the p[k+1], v[k+1], R[k+1] (3+3+4)
            target_payload_pos = dataset[i+1:i+self.lookback+1, 14:17]  # We only need at the output the p[k+1] of the payload (3)

            # Concatenate the target of the payload position to the target of the vehicle
            target = torch.cat([target, target_payload_pos], dim=-1)

            x.append(feature)
            y.append(target)

        # Convert back the list of tensors to a single tensor
        return torch.cat(x, 0), torch.cat(y, 0)

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

        # In this function we want to create a tensor of dimensions (batch_size, max_seq_len, 13)
        datasets = []
        expected_outputs = []

        # Get the timeseries from the batch
        for data in batch:

            datasets += [data[0]]
            expected_outputs += [data[1]]

        # Return a dataset of dimensions (batch_size, max_seq_len, 13), (batch_size, max_seq_len, 10)
        return pad_sequence(datasets, batch_first=True, padding_value=0.0), pad_sequence(expected_outputs, batch_first=True, padding_value=0.0)


if __name__ == "__main__":

    dataset = RealMocap(lookback=1)

    batch = [dataset[0], dataset[1]]
    print(dataset.collate(batch)[1].shape)