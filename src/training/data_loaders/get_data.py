__all__ = ["get_dataset_loader"]

from os.path import join as pjoin
from torch.utils.data import DataLoader

# Import the data models
from .dataset import SimCircles

# A dicitionary of the datasets available
datasets = {"sim_circles": SimCircles}

def get_dataset_class(name):

    try:
        return datasets[name]
    except KeyError:
        raise ValueError(f'Unsupported dataset name [{name}]')

def get_dataset_loader(name, batch_size, datapath, split='train', device="cpu"):
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
    dataset = dataset_class(dataset_path=pjoin(datapath, name), split=split, lookback=1, device=device)

    # Create a dataloader from the dataset (i.e. create batches from the dataset)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=8, drop_last=True, collate_fn=dataset.collate 
    )

    return loader