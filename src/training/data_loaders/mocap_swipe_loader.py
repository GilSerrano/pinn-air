import torch
import numpy as np
import codecs as cs
from tqdm import tqdm
from torch.utils import data
from os.path import join as pjoin


class MocapSwipeLoader(data.Dataset):
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

    def __init__(self, input_window, output_window, stride, dataset_path='', split="test", device="cpu"):

        
        # If not path if given, use the default path for the dataset
        if dataset_path == '':
            dataset_path='./dataset/mocap_14_04_2023'
        
        print('Loading dataset: ' + dataset_path)

        self.input_window = input_window
        self.output_window = output_window
        self.stride = stride

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
        # (input of network)  -> x (batch_size, time, 17)
        self.x = []
        self.y = []
        self.u = []

        for timeseries in self.dataset.values():
            
            # Get only the features that we care about
            # The states that we really care about (x[k]=[p,v,R, p_load], u[k]=[w_ref, T_ref])
            #             Dimension:        3       ,        3       ,          4            ,        3            ,        3           ,        1                 
            series = torch.cat([timeseries["p"], timeseries["v"], timeseries["attitude"], timeseries["p_load"], timeseries["w_ref"], timeseries["T_ref"]], dim=-1).to(self.device)
            
            x, y, u = self.window_squence(series)
            
            self.x.append(x)
            self.y.append(y)
            self.u.append(u)

        # Get the total number of sequences in the dataset
        self.num_sequences = len(self.x)

    def window_squence(self, timeseries, input_window, output_window, stride):
        # ------------------------------------------------------------------------------
        # We must break the sequences into windows:
        # x (time_len, 17) input of the network up until timestep T
        # y (prediction_len, 13) expected prediction from T+1 until T+prediction_len
        # u (prediction_len, 4) the inputs of the system from T+1 until T+prediction_len
        # ------------------------------------------------------------------------------
        
        # Get the total length of the sequence
        total_length = timeseries.shape[0]

        # Get the mini-batch size
        size_mini_batch = (total_length - input_window - output_window) // stride + 1

        # Create the sequence that is used as the input of the network
        x = torch.zeros((size_mini_batch, input_window, 17))

        # Create the sequence that is used as the target of the network + the inputs of the system associated with those targets
        y = torch.zeros((size_mini_batch, output_window, 13))
        u = torch.zeros((size_mini_batch, output_window, 4))

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
            # TODO - check if we need to swipe u 1 unit to the left
            y[i,:,:] = timeseries[start_y:end_y, 0:13]
            u[i,:,:] = timeseries[start_y:end_y, 13:17]

        return x, y, u

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
            y[k] = [p[k+1], v[k+1], R[k+1], p_load[k+1]]
            u[k] = [w_ref[k+1], T_ref[k+1]
        
        """
        # Convert back the list of tensors to a single tensor
        return self.x[index], self.y[index], self.u[index]
    
    @staticmethod
    def collate(batch, device):

        # In this function we want to create a tensor of dimensions (batch_size, max_seq_len, 13)
        x = []
        y = []
        u = []

        # Get the timeseries from the batch
        for data in batch:
            x += [data[0]]
            y += [data[1]]
            u += [data[2]]

        return torch.stack(x, dim=0).to(device), torch.stack(y, dim=0).to(device), torch.stack(u, dim=0).to(device)
        

if __name__ == "__main__":

    dataset = MocapSwipeLoader(device="cpu")
    
    print(len(dataset))

    x, y, u = dataset[0]

    # Get the first 2 samples from the dataset to generate a batch
    print(x.shape)
    print(y.shape)
    print(u.shape)