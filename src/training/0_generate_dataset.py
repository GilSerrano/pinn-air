#!/usr/bin/env python
"""
| File: 0_generate_datase.py
| Authors: Marcelo Jacinto, Gil Serrano, Joao Pinto and Jose Gomes
| Description: Generate the splits for the real and simulation datasets
|              Here it is assumed that you already have the files
|              placed at ./dataset/<Dataset_Name>
"""
from utils.fix_seed import fix_seed
from data_loaders.generate_split import generate_split

# Fix the seed, so that the splits of the datasets is always the same
fix_seed(0)

# ------------------------------------------------------------
# Path to the simulated data of the drone performing circles
# without payload (mass=1.5Kg)
# ------------------------------------------------------------
path_sim_circles_dataset = "./dataset/sim_circles"

generate_split(
    dataset_path=path_sim_circles_dataset,
    percentage_train=0.6,
    percentage_validation=0.2,
    percentage_test=0.2,
    extension=".npz"
)

print("--------------------")

# ------------------------------------------------------------
# Path to the real data of drone with payload of mass=1.0Kg,
# payload=0.103 Kg and tether with 1.0m
# ------------------------------------------------------------
path_mocap_14_04_2023_dataset = "./dataset/mocap_14_04_2023"

generate_split(
    dataset_path=path_mocap_14_04_2023_dataset,
    percentage_train=0.6,
    percentage_validation=0.2,
    percentage_test=0.2,
    extension=".npz"
)