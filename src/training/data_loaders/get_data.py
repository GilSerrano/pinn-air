__all__ = ["get_dataset_loader"]

from os import cpu_count
from os.path import join as pjoin
from torch.utils.data import DataLoader

# Import the data models
from .simulation_dataset import SimCircles
from .mocap_dataset import RealMocap

# A dicitionary of the datasets available
datasets = {
    "sim_circles": SimCircles, 
    "mocap_14_04_2023": RealMocap
}

def get_dataset_class(name):
    try:
        return datasets[name]
    except KeyError:
        raise ValueError(f'Unsupported dataset name [{name}]')

def get_dataset_loader(name, batch_size, datapath, split='train', lookback=5, pooled_classification=True, device="cpu"):
    """
    Generate batches from the dataset.

    Args:
        name (str): Dataset type
        batch_size (int): The size of the batches
        datapath (str): Where the dataset is stored
        split (str, optional): Whether we want the training, validation or test set from this dataset. Defaults to 'train'.
        device (str, optional): The device where the dataset will be placed. Defaults to "cpu".

    Returns:
        Dataloader: A dataloader object that can be used to generate batches from the dataset.
    """

    # Get the dataset class
    dataset_class = get_dataset_class(name)

    # Instantiate the Dataset Object
    dataset = dataset_class(dataset_path=pjoin(datapath, name), split=split, lookback=lookback, pooled_classification=pooled_classification, device=device)

    # Create a dataloader from the dataset (i.e. create batches from the dataset)
    loader = DataLoader(
        dataset,                                          # The dataset itself
        batch_size=batch_size,                            # The size of each batch
        shuffle=False,                                    # Whether to shuffle the sequences on the dataset
        num_workers=0,                                    # The number of cpu to use to load the dataset and generate the batches in parallel
        drop_last=False,                                  # Drop the last samples if not enough to make a batch of the desired size
        collate_fn=lambda x: dataset.collate(x, device),  # Custom collate function for the dataset
        multiprocessing_context=None
    )

    return loader