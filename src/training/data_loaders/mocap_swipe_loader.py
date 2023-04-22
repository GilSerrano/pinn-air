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

    def __init__(self, dataset_path='', split="test", device="cpu"):
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
        self.time = []

        for timeseries in self.dataset.values():
            
            # Get only the features that we care about
            # The states that we really care about (x[k]=[p,v,R], u[k]=[w_ref, T_ref], x_payload[k]=[p] )
            #         Dimension:        3       ,        3       ,          4            ,        3           ,        1           ,        3       
            self.x.append(torch.cat([timeseries["p"], timeseries["v"], timeseries["attitude"], timeseries["w_ref"], timeseries["T_ref"], timeseries["p_load"]], dim=-1).to(self.device))
            self.x[-1].unsqueeze_(0)

            self.time.append(timeseries["time"].to(self.device))

        # Now we must break the sequences into:
        # x (time_len, 17) input of the network up until timestep T
        # y (prediction_len, 13) expected prediction from T+1 until T+prediction_len
        # u (prediction_len, 4) the inputs of the system from T+1 until T+prediction_len
        




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
        return self.x[index], self.time[index]
        

if __name__ == "__main__":

    dataset = MocapSwipeLoader(device="cuda")

    # Get the first 2 samples from the dataset to generate a batch
    print(dataset[0][1].shape)