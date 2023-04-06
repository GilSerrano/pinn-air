from torch.utils.data import DataLoader
from data_loaders.tensors import hypermpc_collate

def get_dataset_class(name):
    if name == "sim_circles":
        from training.data_loaders.dataset import SimCircles
        return SimCircles
    else:
        raise ValueError(f'Unsupported dataset name [{name}]')

def get_collate_fn(name):
    if name in ["sim_circles"]:
        return hypermpc_collate
    else:
        raise ValueError(f'Unsupported dataset name, no collate fucntion for [{name}]')

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

    # Get the dataset
    dataset_class = get_dataset_class(name)
    dataset = dataset_class(split, datapath, device)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=8, drop_last=True, collate_fn=dataset.collate
    )

    return loader