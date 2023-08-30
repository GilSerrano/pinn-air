#!/usr/bin/env python3
"""
| File: rmse_stats.py
| Authors: Gil Serrano, Marcelo Jacinto, Jose Gomes and Joao Pinto
| License: BSD-3-Clause. Copyright (c) 2023, Gil Serrano, Marcelo Jacinto, Jose Gomes and Joao Pinto. All rights reserved.
| Description: Plots histograms of the RMSE metrics and the error histograms.
"""
import os
import torch
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm

# Models
from pinn_air.models.pinn_air_model import PINNAirModel
from pinn_air.models.baseline_model import BaselineModel
from pinn_air.dataset.mocap_dataset import MocapDatasetLoader
from pinn_air.physics.discrete_model import DiscreteModel

# RMSE metrics and utils
from pinn_air.utils.utils import load_best_model, fetch_best_epoch, ArgsParser
from pinn_air.stats_n_plots.rmse import position_rmse, position_rmse_payload, velocity_rmse, quaternion_rmse, rmse_all_state, rmse_all_no_load


def compute_seq_rmse(y_hat, y, expand_y=True, compute_load=True):

    y = y[None,...] if expand_y else y

    rmse_position = position_rmse(y_hat, y)
    rmse_velocity = velocity_rmse(y_hat, y)
    rmse_quaternion = quaternion_rmse(y_hat, y)

    if compute_load:
        rmse_payload = position_rmse_payload(y_hat, y)
        rmse_all = rmse_all_state(y_hat, y)
        return torch.tensor([rmse_position, rmse_velocity, rmse_quaternion, rmse_payload, rmse_all])
    else:
        rmse_all = rmse_all_no_load(y_hat, y)
        return torch.tensor([rmse_position, rmse_velocity, rmse_quaternion, rmse_all])

def get_physics_model_result(x, y, output_window, Ts, mQ, mL, l):
    
        physics_model = DiscreteModel(Ts=Ts, mQ=mQ, mL=mL, l=l)
        physics_model_result = torch.zeros((output_window, 19)).to(x.device)
    
        # Get the last state in time that will be fed into the network along with the first input
        x0 = x[-1, 0:19]
        u0 = x[-1, 19:23]
    
        # Compute the prediction of the physics model over time, given the first real state of the system
        # and the real reference inputs (as if it was an MPC controller)
        for k in range(0, output_window):
    
            # Simulate the model
            physics_model_result[k,:] = physics_model.run(x0, u0)
    
            # Update the initial step and control input for the next iteration
            x0 = physics_model_result[k,:]
            u0 = y[k, 19:23]
        physics_model_result = physics_model_result.unsqueeze(0)
    
        return physics_model_result


def compute_error(y_hat, y):
    # TODO - if we endup using the histogram plots on the paper, fix the quaternion error here - it is not correct to just subtract one from another, but for inspection purposes it is fine
    return y_hat - y


def plot_rmse(model_rmse, physics_rmse, output_dir):
    '''
    Plot histograms of the RMSE metrics
    '''

    # Plot the RMSE for position
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.hist(model_rmse[:, 0], bins=300, alpha=0.5, density=True, color='g', label="Neural Network")
    ax.hist(physics_rmse[:, 0], bins=300, alpha=0.5, density=True, color='r', label="Physics")
    ax.set_title(" RMSE Position")
    ax.legend()
    fig.savefig(os.path.join(output_dir, "rmse_position.png"))

    # Plot the RMSE for velocity
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.hist(model_rmse[:, 1], bins=300, alpha=0.5, density=True, color='g', label="Neural Network")
    ax.hist(physics_rmse[:, 1], bins=300, alpha=0.5, density=True, color='r', label="Physics")
    ax.set_title(" RMSE Velocity")
    ax.legend()
    fig.savefig(os.path.join(output_dir, "rmse_velocity.png"))

    # Plot the RMSE for quaternion
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.hist(model_rmse[:, 2], bins=300, alpha=0.5, density=True, color='g', label="Neural Network")
    ax.hist(physics_rmse[:, 2], bins=300, alpha=0.5, density=True, color='r', label="Physics")
    ax.set_title(" RMSE Quaternion")
    ax.legend()
    fig.savefig(os.path.join(output_dir, "rmse_quaternion.png"))

    # Plot the RMSE for payload position
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.hist(model_rmse[:, 3], bins=300, alpha=0.5, density=True, color='g', label="Neural Network")
    ax.hist(physics_rmse[:, 3], bins=300, alpha=0.5, density=True, color='r', label="Physics")
    ax.set_title(" RMSE Payload Position")
    ax.legend()
    fig.savefig(os.path.join(output_dir, "rmse_load.png"))
    
    # Plot the RMSE for all the states
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.hist(model_rmse[:, 4], bins=300, alpha=0.5, density=True, color='g', label="Neural Network")
    ax.hist(physics_rmse[:, 4], bins=300, alpha=0.5, density=True, color='r', label="Physics")
    ax.set_title(" RMSE All")
    ax.legend()
    fig.savefig(os.path.join(output_dir, "rmse_all.png"))


def plot_error(model_error, physics_error, output_dir):
    '''
    Plot the error histograms
    '''

    # Plot the error of the Position
    fig, ax = plt.subplots(1, 3, figsize=(30, 10))
    ax[0].hist(model_error[:, 0], bins=300, alpha=0.5, density=True, color='g', label="Neural Network")
    ax[0].hist(physics_error[:, 0], bins=300, alpha=0.5, density=True, color='r', label="Physics")
    ax[0].set_title("Error Position - x")
    ax[0].legend()

    ax[1].hist(model_error[:, 1], bins=300, alpha=0.5, density=True, color='g', label="Neural Network")
    ax[1].hist(physics_error[:, 1], bins=300, alpha=0.5, density=True, color='r', label="Physics")
    ax[1].set_title("Error Position - y")
    ax[1].legend()

    ax[2].hist(model_error[:, 2], bins=300, alpha=0.5, density=True, color='g', label="Neural Network")
    ax[2].hist(physics_error[:, 2], bins=300, alpha=0.5, density=True, color='r', label="Physics")
    ax[2].set_title("Error Position - z")
    ax[2].legend()
    fig.savefig(os.path.join(output_dir, "error_position.png"))

    # Plot the error of the Velocity
    fig, ax = plt.subplots(1, 3, figsize=(30, 10))
    ax[0].hist(model_error[:, 3], bins=300, alpha=0.5, density=True, color='g', label="Neural Network")
    ax[0].hist(physics_error[:, 3], bins=300, alpha=0.5, density=True, color='r', label="Physics")
    ax[0].set_title("Error Velocity - x")
    ax[0].legend()

    ax[1].hist(model_error[:, 4], bins=300, alpha=0.5, density=True, color='g', label="Neural Network")
    ax[1].hist(physics_error[:, 4], bins=300, alpha=0.5, density=True, color='r', label="Physics")
    ax[1].set_title("Error Velocity - y")
    ax[1].legend()

    ax[2].hist(model_error[:, 5], bins=300, alpha=0.5, density=True, color='g', label="Neural Network")
    ax[2].hist(physics_error[:, 5], bins=300, alpha=0.5, density=True, color='r', label="Physics")
    ax[2].set_title("Error Velocity - z")
    ax[2].legend()
    fig.savefig(os.path.join(output_dir, "error_velocity.png"))

    # Plot the error of the Quaternion
    fig, ax = plt.subplots(1, 4, figsize=(40, 10))
    ax[0].hist(model_error[:, 6], bins=300, alpha=0.5, density=True, color='g', label="Neural Network")
    ax[0].hist(physics_error[:, 6], bins=300, alpha=0.5, density=True, color='r', label="Physics")
    ax[0].set_title("Error Quaternion - w")
    ax[0].legend()

    ax[1].hist(model_error[:, 7], bins=300, alpha=0.5, density=True, color='g', label="Neural Network")
    ax[1].hist(physics_error[:, 7], bins=300, alpha=0.5, density=True, color='r', label="Physics")
    ax[1].set_title("Error Quaternion - x")
    ax[1].legend()
    
    ax[2].hist(model_error[:, 8], bins=300, alpha=0.5, density=True, color='g', label="Neural Network")
    ax[2].hist(physics_error[:, 8], bins=300, alpha=0.5, density=True, color='r', label="Physics")
    ax[2].set_title("Error Quaternion - y")
    ax[2].legend()

    ax[3].hist(model_error[:, 9], bins=300, alpha=0.5, density=True, color='g', label="Neural Network")
    ax[3].hist(physics_error[:, 9], bins=300, alpha=0.5, density=True, color='r', label="Physics")
    ax[3].set_title("Error Quaternion - z")
    ax[3].legend()
    fig.savefig(os.path.join(output_dir, "error_quaternion.png"))

    # Plot the error of the Payload Position
    fig, ax = plt.subplots(1, 3, figsize=(30, 10))
    ax[0].hist(model_error[:, 10], bins=300, alpha=0.5, density=True, color='g', label="Neural Network")
    ax[0].hist(physics_error[:, 10], bins=300, alpha=0.5, density=True, color='r', label="Physics")
    ax[0].set_title("Error Payload Position - x")
    ax[0].legend()

    ax[1].hist(model_error[:, 11], bins=300, alpha=0.5, density=True, color='g', label="Neural Network")
    ax[1].hist(physics_error[:, 11], bins=300, alpha=0.5, density=True, color='r', label="Physics")
    ax[1].set_title("Error Payload Position - y")
    ax[1].legend()

    ax[2].hist(model_error[:, 12], bins=300, alpha=0.5, density=True, color='g', label="Neural Network")
    ax[2].hist(physics_error[:, 12], bins=300, alpha=0.5, density=True, color='r', label="Physics")
    ax[2].set_title("Error Payload Position - z")
    ax[2].legend()
    fig.savefig(os.path.join(output_dir, "error_payload_position.png"))

def plot_error_through_time(model_error_through_time, physics_error_through_time, output_window, output_dir):
    
    # Calculate the mean through time
    model_error_through_time_mean = np.mean(model_error_through_time, axis=0)
    physics_error_through_time_mean = np.mean(physics_error_through_time, axis=0)

    # Calculate the std through time
    model_error_through_time_std = np.std(model_error_through_time, axis=0)
    physics_error_through_time_std = np.std(physics_error_through_time, axis=0)

    # Calculate the maximum and minimum through time
    model_error_through_time_max = np.max(model_error_through_time, axis=0)
    physics_error_through_time_max = np.max(physics_error_through_time, axis=0)
    model_error_through_time_min = np.min(model_error_through_time, axis=0)
    physics_error_through_time_min = np.min(physics_error_through_time, axis=0)

    timesteps = np.arange(0, output_window)

    '''
    Position
    '''
    fig, ax = plt.subplots(1, 3, figsize=(30, 10))
    ax[0].plot(timesteps, model_error_through_time_mean[:, 0], color='g', label="Neural Network")
    ax[0].fill_between(timesteps, model_error_through_time_min[:, 0], model_error_through_time_max[:, 0], color='g', alpha=0.2)    
    ax[0].plot(timesteps, physics_error_through_time_mean[:, 0], color='r', label="Physics")
    ax[0].fill_between(timesteps, physics_error_through_time_min[:, 0], physics_error_through_time_max[:, 0], color='r', alpha=0.2)
    ax[0].set_title("Error Position - x")
    ax[0].legend()

    ax[1].plot(timesteps, model_error_through_time_mean[:, 1], color='g', label="Neural Network")
    ax[1].fill_between(timesteps, model_error_through_time_min[:, 1], model_error_through_time_max[:, 1], color='g', alpha=0.2)
    ax[1].plot(timesteps, physics_error_through_time_mean[:, 1], color='r', label="Physics")
    ax[1].fill_between(timesteps, physics_error_through_time_min[:, 1], physics_error_through_time_max[:, 1], color='r', alpha=0.2)
    ax[1].set_title("Error Position - y")
    ax[1].legend()

    ax[2].plot(timesteps, model_error_through_time_mean[:, 2], color='g', label="Neural Network")
    ax[2].fill_between(timesteps, model_error_through_time_min[:, 2], model_error_through_time_max[:, 2], color='g', alpha=0.2)
    ax[2].plot(timesteps, physics_error_through_time_mean[:, 2], color='r', label="Physics")
    ax[2].fill_between(timesteps, physics_error_through_time_min[:, 2], physics_error_through_time_max[:, 2], color='r', alpha=0.2)
    ax[2].set_title("Error Position - z")
    ax[2].legend()

    fig.savefig(os.path.join(output_dir, "error_minmax_through_time_position.png"))


    fig, ax = plt.subplots(1, 3, figsize=(30, 10))
    ax[0].plot(timesteps, model_error_through_time_mean[:, 0], color='g', label="Neural Network")
    ax[0].fill_between(timesteps, model_error_through_time_mean[:, 0] - model_error_through_time_std[:, 0], model_error_through_time_mean[:, 0] + model_error_through_time_std[:, 0], color='g', alpha=0.2)
    ax[0].plot(timesteps, physics_error_through_time_mean[:, 0], color='r', label="Physics")
    ax[0].fill_between(timesteps, physics_error_through_time_mean[:, 0] - physics_error_through_time_std[:, 0], physics_error_through_time_mean[:, 0] + physics_error_through_time_std[:, 0], color='r', alpha=0.2)
    ax[0].set_title("Error Position - x")
    ax[0].legend()

    ax[1].plot(timesteps, model_error_through_time_mean[:, 1], color='g', label="Neural Network")
    ax[1].fill_between(timesteps, model_error_through_time_mean[:, 1] - model_error_through_time_std[:, 1], model_error_through_time_mean[:, 1] + model_error_through_time_std[:, 1], color='g', alpha=0.2)
    ax[1].plot(timesteps, physics_error_through_time_mean[:, 1], color='r', label="Physics")
    ax[1].fill_between(timesteps, physics_error_through_time_mean[:, 1] - physics_error_through_time_std[:, 1], physics_error_through_time_mean[:, 1] + physics_error_through_time_std[:, 1], color='r', alpha=0.2)
    ax[1].set_title("Error Position - y")
    ax[1].legend()

    ax[2].plot(timesteps, model_error_through_time_mean[:, 2], color='g', label="Neural Network")
    ax[2].fill_between(timesteps, model_error_through_time_mean[:, 2] - model_error_through_time_std[:, 2], model_error_through_time_mean[:, 2] + model_error_through_time_std[:, 2], color='g', alpha=0.2)
    ax[2].plot(timesteps, physics_error_through_time_mean[:, 2], color='r', label="Physics")
    ax[2].fill_between(timesteps, physics_error_through_time_mean[:, 2] - physics_error_through_time_std[:, 2], physics_error_through_time_mean[:, 2] + physics_error_through_time_std[:, 2], color='r', alpha=0.2)
    ax[2].set_title("Error Position - z")
    ax[2].legend()

    fig.savefig(os.path.join(output_dir, "error_std_through_time_position.png"))



    '''
    Velocity
    '''
    fig, ax = plt.subplots(1, 3, figsize=(30, 10))
    ax[0].plot(timesteps, model_error_through_time_mean[:, 4], color='g', label="Neural Network")
    ax[0].fill_between(timesteps, model_error_through_time_min[:, 4], model_error_through_time_max[:, 4], color='g', alpha=0.2)
    ax[0].plot(timesteps, physics_error_through_time_mean[:, 4], color='r', label="Physics")
    ax[0].fill_between(timesteps, physics_error_through_time_min[:, 4], physics_error_through_time_max[:, 4], color='r', alpha=0.2)
    ax[0].set_title("Error Velocity - x")
    ax[0].legend()

    ax[1].plot(timesteps, model_error_through_time_mean[:, 5], color='g', label="Neural Network")
    ax[1].fill_between(timesteps, model_error_through_time_min[:, 5], model_error_through_time_max[:, 5], color='g', alpha=0.2)
    ax[1].plot(timesteps, physics_error_through_time_mean[:, 5], color='r', label="Physics")
    ax[1].fill_between(timesteps, physics_error_through_time_min[:, 5], physics_error_through_time_max[:, 5], color='r', alpha=0.2)
    ax[1].set_title("Error Velocity - y")
    ax[1].legend()

    ax[2].plot(timesteps, model_error_through_time_mean[:, 6], color='g', label="Neural Network")
    ax[2].fill_between(timesteps, model_error_through_time_min[:, 6], model_error_through_time_max[:, 6], color='g', alpha=0.2)    
    ax[2].plot(timesteps, physics_error_through_time_mean[:, 6], color='r', label="Physics")
    ax[2].fill_between(timesteps, physics_error_through_time_min[:, 6], physics_error_through_time_max[:, 6], color='r', alpha=0.2)
    ax[2].set_title("Error Velocity - z")
    ax[2].legend()

    fig.savefig(os.path.join(output_dir, "error_minmax_through_time_velocity.png"))

    fig, ax = plt.subplots(1, 3, figsize=(30, 10))
    ax[0].plot(timesteps, model_error_through_time_mean[:, 4], color='g', label="Neural Network")
    ax[0].fill_between(timesteps, model_error_through_time_mean[:, 4] - model_error_through_time_std[:, 4], model_error_through_time_mean[:, 4] + model_error_through_time_std[:, 4], color='g', alpha=0.2)
    ax[0].plot(timesteps, physics_error_through_time_mean[:, 4], color='r', label="Physics")
    ax[0].fill_between(timesteps, physics_error_through_time_mean[:, 4] - physics_error_through_time_std[:, 4], physics_error_through_time_mean[:, 4] + physics_error_through_time_std[:, 4], color='r', alpha=0.2)
    ax[0].set_title("Error Velocity - x")
    ax[0].legend()

    ax[1].plot(timesteps, model_error_through_time_mean[:, 5], color='g', label="Neural Network")
    ax[1].fill_between(timesteps, model_error_through_time_mean[:, 5] - model_error_through_time_std[:, 5], model_error_through_time_mean[:, 5] + model_error_through_time_std[:, 5], color='g', alpha=0.2)
    ax[1].plot(timesteps, physics_error_through_time_mean[:, 5], color='r', label="Physics")
    ax[1].fill_between(timesteps, physics_error_through_time_mean[:, 5] - physics_error_through_time_std[:, 5], physics_error_through_time_mean[:, 5] + physics_error_through_time_std[:, 5], color='r', alpha=0.2)
    ax[1].set_title("Error Velocity - y")
    ax[1].legend()

    ax[2].plot(timesteps, model_error_through_time_mean[:, 6], color='g', label="Neural Network")
    ax[2].fill_between(timesteps, model_error_through_time_mean[:, 6] - model_error_through_time_std[:, 6], model_error_through_time_mean[:, 6] + model_error_through_time_std[:, 6], color='g', alpha=0.2)
    ax[2].plot(timesteps, physics_error_through_time_mean[:, 6], color='r', label="Physics")
    ax[2].fill_between(timesteps, physics_error_through_time_mean[:, 6] - physics_error_through_time_std[:, 6], physics_error_through_time_mean[:, 6] + physics_error_through_time_std[:, 6], color='r', alpha=0.2)
    ax[2].set_title("Error Velocity - z")
    ax[2].legend()

    fig.savefig(os.path.join(output_dir, "error_std_through_time_velocity.png"))

    '''
    Load
    '''
    fig, ax = plt.subplots(1, 3, figsize=(30, 10))
    ax[0].plot(timesteps, model_error_through_time_mean[:, 10], color='g', label="Neural Network")
    ax[0].fill_between(timesteps, model_error_through_time_min[:, 10], model_error_through_time_max[:, 10], color='g', alpha=0.2)
    ax[0].plot(timesteps, physics_error_through_time_mean[:, 10], color='r', label="Physics")
    ax[0].fill_between(timesteps, physics_error_through_time_min[:, 10], physics_error_through_time_max[:, 10], color='r', alpha=0.2)
    ax[0].set_title("Error Payload Position - x")
    ax[0].legend()

    ax[1].plot(timesteps, model_error_through_time_mean[:, 11], color='g', label="Neural Network")
    ax[1].fill_between(timesteps, model_error_through_time_min[:, 11], model_error_through_time_max[:, 11], color='g', alpha=0.2)
    ax[1].plot(timesteps, physics_error_through_time_mean[:, 11], color='r', label="Physics")
    ax[1].fill_between(timesteps, physics_error_through_time_min[:, 11], physics_error_through_time_max[:, 11], color='r', alpha=0.2)
    ax[1].set_title("Error Payload Position - y")
    ax[1].legend()

    ax[2].plot(timesteps, model_error_through_time_mean[:, 12], color='g', label="Neural Network")
    ax[2].fill_between(timesteps, model_error_through_time_min[:, 12], model_error_through_time_max[:, 12], color='g', alpha=0.2)
    ax[2].plot(timesteps, physics_error_through_time_mean[:, 12], color='r', label="Physics")
    ax[2].fill_between(timesteps, physics_error_through_time_min[:, 12], physics_error_through_time_max[:, 12], color='r', alpha=0.2)
    ax[2].set_title("Error Payload Position - z")
    ax[2].legend()

    fig.savefig(os.path.join(output_dir, "error_minmax_through_time_load.png"))

    fig, ax = plt.subplots(1, 3, figsize=(30, 10))
    ax[0].plot(timesteps, model_error_through_time_mean[:, 10], color='g', label="Neural Network")
    ax[0].fill_between(timesteps, model_error_through_time_mean[:, 10] - model_error_through_time_std[:, 10], model_error_through_time_mean[:, 10] + model_error_through_time_std[:, 10], color='g', alpha=0.2)
    ax[0].plot(timesteps, physics_error_through_time_mean[:, 10], color='r', label="Physics")
    ax[0].fill_between(timesteps, physics_error_through_time_mean[:, 10] - physics_error_through_time_std[:, 10], physics_error_through_time_mean[:, 10] + physics_error_through_time_std[:, 10], color='r', alpha=0.2)
    ax[0].set_title("Error Payload Position - x")
    ax[0].legend()

    ax[1].plot(timesteps, model_error_through_time_mean[:, 11], color='g', label="Neural Network")
    ax[1].fill_between(timesteps, model_error_through_time_mean[:, 11] - model_error_through_time_std[:, 11], model_error_through_time_mean[:, 11] + model_error_through_time_std[:, 11], color='g', alpha=0.2)
    ax[1].plot(timesteps, physics_error_through_time_mean[:, 11], color='r', label="Physics")
    ax[1].fill_between(timesteps, physics_error_through_time_mean[:, 11] - physics_error_through_time_std[:, 11], physics_error_through_time_mean[:, 11] + physics_error_through_time_std[:, 11], color='r', alpha=0.2)
    ax[1].set_title("Error Payload Position - y")
    ax[1].legend()

    ax[2].plot(timesteps, model_error_through_time_mean[:, 12], color='g', label="Neural Network")
    ax[2].fill_between(timesteps, model_error_through_time_mean[:, 12] - model_error_through_time_std[:, 12], model_error_through_time_mean[:, 12] + model_error_through_time_std[:, 12], color='g', alpha=0.2)
    ax[2].plot(timesteps, physics_error_through_time_mean[:, 12], color='r', label="Physics")
    ax[2].fill_between(timesteps, physics_error_through_time_mean[:, 12] - physics_error_through_time_std[:, 12], physics_error_through_time_mean[:, 12] + physics_error_through_time_std[:, 12], color='r', alpha=0.2)
    ax[2].set_title("Error Payload Position - z")
    ax[2].legend()

    fig.savefig(os.path.join(output_dir, "error_std_through_time_load.png"))


def main():

    args_parser = ArgsParser()

    # -------------------------------------------------------------------
    # Plots for the regular test were we perform the recursive prediction
    # ------------------------------------------------------------------- 
    output_dir = args_parser.args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    best_epoch = fetch_best_epoch(output_dir)
    
    # Set the device for performing training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device: {}".format(device))

    system_state_dim = 13
    system_control_dim = 4

    models = { 
        "BaselineModel": BaselineModel(system_state_dim=system_state_dim, system_control_dim=system_control_dim, num_layers=3, dropout=0.1, device=device),
        "PINNAirModel": PINNAirModel(system_state_dim=system_state_dim, system_control_dim=system_control_dim, num_layers=3, dropout=0.1, device=device)
    }
    
    model = models[args_parser.args.model]
    model = load_best_model(best_epoch, model, output_dir=output_dir, device=device)
    model.eval()
    
    input_window = 50
    output_window = 25
    test_dataset = MocapDatasetLoader(input_window=input_window, output_window=output_window, stride=1, split="test", device=device)
    
    print(f'Dataset length {len(test_dataset)}')

    # Check if rmses have already been computed and then load them
    if os.path.exists(os.path.join(output_dir, "model_rmse.npy")) and os.path.exists(os.path.join(output_dir, "physics_rmse.npy")):
        model_rmse = np.load(os.path.join(output_dir, "model_rmse.npy"))
        physics_rmse = np.load(os.path.join(output_dir, "physics_rmse.npy"))
        model_error = np.load(os.path.join(output_dir, "model_error.npy"))
        physics_error = np.load(os.path.join(output_dir, "physics_error.npy"))
        model_error_through_time = np.load(os.path.join(output_dir, "model_error_through_time.npy"))
        physics_error_through_time = np.load(os.path.join(output_dir, "physics_error_through_time.npy"))
    else:
        model_rmse = np.zeros((len(test_dataset), 5))
        physics_rmse = np.zeros((len(test_dataset), 5))
        model_error = np.zeros((len(test_dataset)*output_window, 13))
        physics_error = np.zeros((len(test_dataset)*output_window, 13))
        model_error_through_time = np.zeros((len(test_dataset), output_window, 13))
        physics_error_through_time = np.zeros((len(test_dataset), output_window, 13))

        for seq in tqdm(range(0, len(test_dataset))):

            x, y = test_dataset[seq]
            x = x.to(device)
            y = y.to(device)

            physics_model_result = get_physics_model_result(x, y, output_window, Ts=test_dataset.Ts, mQ=1.35, mL=0.1, l=0.7)
        
            u = y[..., 19:23]
            x = torch.cat([x[:, 0:13], x[:, 19:23]], dim=-1)
            y_hat = model(x[None, :, :], u[None, :, :])

            x = x.to("cpu")
            y = y.to("cpu")
            y_hat = y_hat.to("cpu")
            physics_model_result = physics_model_result.to("cpu")

            # Compute all the RMSE metrics for this particular trajectory
            model_rmse[seq, :] = compute_seq_rmse(y_hat, y).detach().numpy()

            # Compute all the RMSE metrics for the physics model
            physics_rmse[seq, :] = compute_seq_rmse(y[None,...], physics_model_result, expand_y=False, compute_load=True).detach().numpy()
            
            # Compute the error of the model
            model_error[seq*output_window:(seq+1)*output_window, :] =  compute_error(y_hat=y_hat[..., 0:13].squeeze(), y=y[:, 0:13]).detach().numpy()
            model_error_through_time[seq, :, :] = model_error[seq*output_window:(seq+1)*output_window, :]
            
            # Compute the error of the physics model
            physics_error[seq*output_window:(seq+1)*output_window, :] = compute_error(y_hat=physics_model_result.squeeze()[:, 0:13], y=y[:, 0:13]).detach().numpy()
            physics_error_through_time[seq, :, :] = physics_error[seq*output_window:(seq+1)*output_window, :]

        # Save RMSE metrics to file
        np.save(os.path.join(output_dir, "model_rmse.npy"), model_rmse)
        np.save(os.path.join(output_dir, "physics_rmse.npy"), physics_rmse)
        np.save(os.path.join(output_dir, "model_error.npy"), model_error)
        np.save(os.path.join(output_dir, "physics_error.npy"), physics_error)
        np.save(os.path.join(output_dir, "model_error_through_time.npy"), model_error_through_time)
        np.save(os.path.join(output_dir, "physics_error_through_time.npy"), physics_error_through_time)

    
    '''
    RMSE metrics
    '''
    # Compute the mean and std of the RMSE metrics
    model_rmse_mean = np.mean(model_rmse, axis=0)
    model_rmse_std = np.std(model_rmse, axis=0)

    physics_rmse_mean = np.mean(physics_rmse, axis=0)
    physics_rmse_std = np.std(physics_rmse, axis=0)

    # Print the results
    print("Model RMSE mean: {}".format(model_rmse_mean))
    print("Model RMSE std: {}".format(model_rmse_std))
    print("Physics RMSE mean: {}".format(physics_rmse_mean))
    print("Physics RMSE std: {}".format(physics_rmse_std))

    # Plot the RMSE metrics
    plot_rmse(model_rmse, physics_rmse, output_dir)

    
    '''
    Error metrics
    '''
    # Compute the mean and std of the error metrics
    model_error_mean = np.mean(model_error, axis=0)
    model_error_std = np.std(model_error, axis=0)

    physics_error_mean = np.mean(physics_error, axis=0)
    physics_error_std = np.std(physics_error, axis=0)

    # Print the results
    print("Model error mean: {}".format(model_error_mean))
    print("Model error std: {}".format(model_error_std))
    print("Physics error mean: {}".format(physics_error_mean))
    print("Physics error std: {}".format(physics_error_std))    

    # Plot the error metrics
    plot_error(model_error, physics_error, output_dir)

    # Plot the error through time
    plot_error_through_time(model_error_through_time, physics_error_through_time, output_window, output_dir)

if __name__ == "__main__":
    main()
    
