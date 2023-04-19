# Import the method that will return the dataloader
from .get_data import get_dataset_loader

# Import the datasets
from .mocap_dataset import RealMocap
from .simulation_dataset import SimCircles