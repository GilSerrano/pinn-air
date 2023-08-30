"""
| File: plot_3dtrajectory.py
| Authors: Gil Serrano, Marcelo Jacinto, Jose Gomes and Joao Pinto
| License: BSD-3-Clause. Copyright (c) 2023, Gil Serrano, Marcelo Jacinto, Jose Gomes and Joao Pinto. All rights reserved.
| Description: Plots the trajectories executed by the quadrotor with the attached slungload in the dataset (used in the paper).
"""
import os

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_3d_trajectory(trajectory, output_dir, name):
    # Create a figure and a 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the trajectory
    ax.plot(trajectory[:,0], trajectory[:,1], trajectory[:,2])
    
    # Set labels for the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    fig.savefig(os.path.join(output_dir, name))
    plt.close('all')

def main():

    dataset_path=os.path.abspath('pinn_air/dataset/mocap_14_04_2023')
    output_path=os.path.abspath('trajectories_plots')

    # Load name list from txt file
    name_list = []
    with open(os.path.join(dataset_path, "train.txt"), 'r') as f:
        for line in f:
            name_list.append(line.strip())

    for name in name_list:
         
        # Set the path to the dataset
        data_path = os.path.join(dataset_path, 'data', name)
        
        # Load the dataset
        timeseries = np.load(data_path)
            
        # Plot the trajectory
        os.makedirs(output_path, exist_ok=True)
        plot_3d_trajectory(timeseries['p'], output_path, name.replace('.npz', '.png'))


if __name__ == '__main__':
    main()