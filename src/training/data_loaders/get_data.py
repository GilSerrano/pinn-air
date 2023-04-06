from torch.utils.data import DataLoader
from data_loaders.tensors import hypermpc_collate

def get_dataset_class(name):
    if name == "sim_circles":
        from data_loaders.sim_circles.data.dataset import SIM_CIRCLES
        return SIM_CIRCLES
    else:
        raise ValueError(f'Unsupported dataset name [{name}]')

def get_collate_fn(name):
    if name in ["sim_circles"]:
        return hypermpc_collate
    else:
        raise ValueError(f'Unsupported dataset name, no collate fucntion for [{name}]')

def get_dataset(name, split='train'):
    DATA = get_dataset_class(name)
    dataset = DATA(split=split)
    return dataset


def get_dataset_loader(name, batch_size, split='train'):
    dataset = get_dataset(name, split)
    collate = get_collate_fn(name)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=8, drop_last=True, collate_fn=collate
    )

    return loader