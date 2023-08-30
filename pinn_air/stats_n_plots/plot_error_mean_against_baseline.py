#!/usr/bin/env python3
"""
| File: plot_error_mean_against_baseline.py
| Authors: Gil Serrano, Marcelo Jacinto, Jose Gomes and Joao Pinto
| License: BSD-3-Clause. Copyright (c) 2023, Gil Serrano, Marcelo Jacinto, Jose Gomes and Joao Pinto. All rights reserved.
| Description: Plots the mean error and standard deviation of the error of the proposed model against the baseline model (used in the paper).
"""
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from pinn_air.utils.math_utils import quaternion_multiply, quaternion_invert

from tqdm import tqdm

# Models
from pinn_air.models.pinn_air_model import PINNAirModel
from pinn_air.models.baseline_model import BaselineModel
from pinn_air.dataset.mocap_dataset import MocapDatasetLoader

# RMSE metrics and utils
from argparse import ArgumentParser
from pinn_air.utils.utils import load_best_model, fetch_best_epoch


class ArgsParser:

    def __init__(self):    
        # Create the parser
        self.parser = ArgumentParser()

        # Base options for training
        train_group = self.parser.add_argument_group('training')
        train_group.add_argument("--output_dir", default="./output", type=str, help="Output directory.")
        train_group.add_argument("--output_baseline", default="./output_baseline", type=str, help="Output directory of the baseline model.")

        self.args = self.parser.parse_args()


def compute_error(y_hat, y):

    pos_error = y_hat[..., 0:3] - y[...,0:3]
    vel_error = y_hat[..., 3:6] - y[...,3:6]
    payload_error = y_hat[...,10:13] - y[...,10:13]

    # Compute the quaternion error
    identity_quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0]).to(pos_error.device)
    quat_error = quaternion_multiply(y[..., 6:10], quaternion_invert(y_hat[..., 6:10])) - identity_quaternion

    # Computer the rest of the errors for the other states
    return torch.cat((pos_error, vel_error, quat_error, payload_error), dim=-1)


def plot_error_through_time(model_error_through_time, baseline_error_through_time, output_window, output_dir):

    Ts = 0.03

    # Compute the norm of the error from the errors through time
    pos_error_through_time = np.linalg.norm(model_error_through_time[:,:,0:3], ord=2, axis=-1)
    vel_error_through_time = np.linalg.norm(model_error_through_time[:,:,3:6], ord=2, axis=-1)
    quat_error_through_time = np.linalg.norm(model_error_through_time[:,:,6:10], ord=2, axis=-1)
    payload_error_through_time = np.linalg.norm(model_error_through_time[:,:,10:13], ord=2, axis=-1)

    # Compute the norm of the physics error from the errors through time
    pos_baseline_error_through_time = np.linalg.norm(baseline_error_through_time[:,:,0:3], ord=2, axis=-1)
    vel_baseline_error_through_time = np.linalg.norm(baseline_error_through_time[:,:,3:6], ord=2, axis=-1)
    quat_baseline_error_through_time = np.linalg.norm(baseline_error_through_time[:,:,6:10], ord=2, axis=-1)
    payload_baseline_error_through_time = np.linalg.norm(baseline_error_through_time[:,:,10:13], ord=2, axis=-1)

    # Compute the mean through time
    pos_error_through_time_mean = np.mean(pos_error_through_time, axis=0)
    vel_error_through_time_mean = np.mean(vel_error_through_time, axis=0)
    quat_error_through_time_mean = np.mean(quat_error_through_time, axis=0)
    payload_error_through_time_mean = np.mean(payload_error_through_time, axis=0)

    # Compute the std through time
    pos_error_through_time_std = np.std(pos_error_through_time, axis=0)
    vel_error_through_time_std = np.std(vel_error_through_time, axis=0)
    quat_error_through_time_std = np.std(quat_error_through_time, axis=0)
    payload_error_through_time_std = np.std(payload_error_through_time, axis=0)

    # Compute the min through time
    pos_error_through_time_min = np.min(pos_error_through_time, axis=0)
    vel_error_through_time_min = np.min(vel_error_through_time, axis=0)
    quat_error_through_time_min = np.min(quat_error_through_time, axis=0)
    payload_error_through_time_min = np.min(payload_error_through_time, axis=0)

    # Compute the max through time
    pos_error_through_time_max = np.max(pos_error_through_time, axis=0)
    vel_error_through_time_max = np.max(vel_error_through_time, axis=0)
    quat_error_through_time_max = np.max(quat_error_through_time, axis=0)
    payload_error_through_time_max = np.max(payload_error_through_time, axis=0)

    # Compute the physics mean through time
    pos_baseline_error_through_time_mean = np.mean(pos_baseline_error_through_time, axis=0)
    vel_baseline_error_through_time_mean = np.mean(vel_baseline_error_through_time, axis=0)
    quat_baseline_error_through_time_mean = np.mean(quat_baseline_error_through_time, axis=0)
    payload_baseline_error_through_time_mean = np.mean(payload_baseline_error_through_time, axis=0)

    # Compute the physics std through time
    pos_baseline_error_through_time_std = np.std(pos_baseline_error_through_time, axis=0)
    vel_baseline_error_through_time_std = np.std(vel_baseline_error_through_time, axis=0)
    quat_baseline_error_through_time_std = np.std(quat_baseline_error_through_time, axis=0)
    payload_baseline_error_through_time_std = np.std(payload_baseline_error_through_time, axis=0)

    # Compute the physics min through time
    pos_baseline_error_through_time_min = np.min(pos_baseline_error_through_time, axis=0)
    vel_baseline_error_through_time_min = np.min(vel_baseline_error_through_time, axis=0)
    quat_baseline_error_through_time_min = np.min(quat_baseline_error_through_time, axis=0)
    payload_baseline_error_through_time_min = np.min(payload_baseline_error_through_time, axis=0)

    # Compute the physics max through time
    pos_baseline_error_through_time_max = np.max(pos_baseline_error_through_time, axis=0)
    vel_baseline_error_through_time_max = np.max(vel_baseline_error_through_time, axis=0)
    quat_baseline_error_through_time_max = np.max(quat_baseline_error_through_time, axis=0)
    payload_baseline_error_through_time_max = np.max(payload_baseline_error_through_time, axis=0)

    # Create the time vector using a sample rate of 0.03 seconds
    timesteps = np.arange(0, output_window * Ts, Ts)

    '''
    Position
    '''

    fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=False)
    # No Physics
    ax[0,0].plot(timesteps, pos_baseline_error_through_time_mean, color='orange', label="NN (without physics)")
    ax[0,0].fill_between(timesteps, pos_baseline_error_through_time_mean - pos_baseline_error_through_time_std, pos_baseline_error_through_time_mean + pos_baseline_error_through_time_std, color='orange', alpha=0.2)
    # Neural Network
    ax[0,0].plot(timesteps, pos_error_through_time_mean, color='g', label="Neural Network")
    ax[0,0].fill_between(timesteps, pos_error_through_time_mean - pos_error_through_time_std, pos_error_through_time_mean + pos_error_through_time_std, color='g', alpha=0.2)
    ax[0,0].set(ylabel="Position Error (m)")
    ax[0,0].axvline(x = 25 * Ts, color = 'black', ls='--')
    ax[0,0].legend()
    #ax[0,0].grid()

    '''
    Velocity
    '''
    # No Physics
    ax[0,1].plot(timesteps, vel_baseline_error_through_time_mean, color='orange', label="NN (without physics)")
    ax[0,1].fill_between(timesteps, vel_baseline_error_through_time_mean - vel_baseline_error_through_time_std, vel_baseline_error_through_time_mean + vel_baseline_error_through_time_std, color='orange', alpha=0.2)
    # Neural Network
    ax[0,1].plot(timesteps, vel_error_through_time_mean, color='g', label="Neural Network")
    ax[0,1].fill_between(timesteps, vel_error_through_time_mean - vel_error_through_time_std, vel_error_through_time_mean + vel_error_through_time_std, color='g', alpha=0.2)
    ax[0,1].set(ylabel="Velocity Error (m/s)")
    ax[0,1].axvline(x = 25 * Ts, color = 'black', ls='--')
    ax[0,1].legend()
    #ax[0,1].grid()

    '''
    Quaternion
    '''
    # No Physics
    ax[1,0].plot(timesteps, quat_baseline_error_through_time_mean, color='orange', label="NN (without physics)")
    ax[1,0].fill_between(timesteps, quat_baseline_error_through_time_mean - quat_baseline_error_through_time_std, quat_baseline_error_through_time_mean + quat_baseline_error_through_time_std, color='orange', alpha=0.2)
    # Neural Network
    ax[1,0].plot(timesteps, quat_error_through_time_mean, color='g', label="Neural Network")
    ax[1,0].fill_between(timesteps, quat_error_through_time_mean - quat_error_through_time_std, quat_error_through_time_mean + quat_error_through_time_std, color='g', alpha=0.2)
    ax[1,0].set(ylabel="Quaternion Error Norm")
    ax[1,0].axvline(x = 25 * Ts, color = 'black', ls='--')
    ax[1,0].legend()
    #ax[1,0].grid()

    '''
    Payload position
    '''

    # No Physics
    ax[1,1].plot(timesteps, payload_baseline_error_through_time_mean, color='orange', label="NN (without physics)")
    ax[1,1].fill_between(timesteps, payload_baseline_error_through_time_mean - payload_baseline_error_through_time_std, payload_baseline_error_through_time_mean + payload_baseline_error_through_time_std, color='orange', alpha=0.2)
    # Neural Network
    ax[1,1].plot(timesteps, payload_error_through_time_mean, color='g', label="Neural Network")
    ax[1,1].fill_between(timesteps, payload_error_through_time_mean - payload_error_through_time_std, payload_error_through_time_mean + payload_error_through_time_std, color='g', alpha=0.2)
    ax[1,1].set(ylabel="Load Position Error (m)")
    ax[1,1].axvline(x = 25 * Ts, color = 'black', ls='--')
    ax[1,1].legend()
    #ax[1,1].grid()

    #fig.savefig(os.path.join(output_dir, "error_std_norm_states.png"))
    plt.setp(ax[-1, :], xlabel='Time (s)')
    fig.tight_layout()
    plt.savefig(output_dir + '/' + "std_avg_against_baseline.pdf")


    # Create a new figure
    plt.close('all')
    fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=False)
    # -------------------------------------------------------------------
    # Plots for the Min and Max values
    # -------------------------------------------------------------------

    '''
    Position
    '''
    # No Physics
    ax[0,0].plot(timesteps, pos_baseline_error_through_time_mean, color='orange', label="NN (without physics)")
    ax[0,0].fill_between(timesteps, pos_baseline_error_through_time_min, pos_baseline_error_through_time_max, color='orange', alpha=0.2)
    # Neural Network
    ax[0,0].plot(timesteps, pos_error_through_time_mean, color='g', label="Neural Network")
    ax[0,0].fill_between(timesteps, pos_error_through_time_min, pos_error_through_time_max, color='g', alpha=0.2)
    ax[0,0].set(ylabel="Position Error (m)")
    ax[0,0].axvline(x = 25 * Ts, color = 'black', ls='--')
    ax[0,0].legend()
    #ax[0,0].grid()

    '''
    Velocity
    '''
    # No Physics
    ax[0,1].plot(timesteps, vel_baseline_error_through_time_mean, color='orange', label="NN (without physics)")
    ax[0,1].fill_between(timesteps, vel_baseline_error_through_time_min, vel_baseline_error_through_time_max, color='orange', alpha=0.2)
    # Neural Network
    ax[0,1].plot(timesteps, vel_error_through_time_mean, color='g', label="Neural Network")
    ax[0,1].fill_between(timesteps, vel_error_through_time_min, vel_error_through_time_max, color='g', alpha=0.2)
    ax[0,1].set(ylabel="Velocity Error (m)")
    ax[0,1].axvline(x = 25 * Ts, color = 'black', ls='--')
    ax[0,1].legend()
    #ax[0,1].grid()

    '''
    Quaternion
    '''
    # No Physics
    ax[1,0].plot(timesteps, quat_baseline_error_through_time_mean, color='orange', label="NN (without physics)")
    ax[1,0].fill_between(timesteps, quat_baseline_error_through_time_min, quat_baseline_error_through_time_max, color='orange', alpha=0.2)
    # Neural Network
    ax[1,0].plot(timesteps, quat_error_through_time_mean, color='g', label="Neural Network")
    ax[1,0].fill_between(timesteps, quat_error_through_time_min, quat_error_through_time_max, color='g', alpha=0.2)
    ax[1,0].set(ylabel="Quaternion Error Norm (m)")
    ax[1,0].axvline(x = 25 * Ts, color = 'black', ls='--')
    ax[1,0].legend()
    #ax[1,0].grid()

    '''
    Payload
    '''
    # No Physics
    ax[1,1].plot(timesteps, payload_baseline_error_through_time_mean, color='orange', label="NN (without physics)")
    ax[1,1].fill_between(timesteps, payload_baseline_error_through_time_min, payload_baseline_error_through_time_max, color='orange', alpha=0.2)
    # Neural Network
    ax[1,1].plot(timesteps, payload_error_through_time_mean, color='g', label="Neural Network")
    ax[1,1].fill_between(timesteps, payload_error_through_time_min, payload_error_through_time_max, color='g', alpha=0.2)
    ax[1,1].set(ylabel="Load Position Error (m)")
    ax[1,1].axvline(x = 25 * Ts, color = 'black', ls='--')
    ax[1,1].legend()
    #ax[1,1].grid()

    plt.setp(ax[-1, :], xlabel='Time (s)')
    fig.tight_layout()
    plt.savefig(output_dir + '/' + "min_max_avg_against_baseline.pdf")


def main():

    args_parser = ArgsParser()

    # -------------------------------------------------------------------
    # Plots for the regular test were we perform the recursive prediction
    # ------------------------------------------------------------------- 

    # Load the best epoch of the proposed model
    output_dir = args_parser.args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    best_epoch = fetch_best_epoch(output_dir)

    # Load the best epoch of the baseline model
    output_baseline = args_parser.args.output_baseline
    os.makedirs(output_baseline, exist_ok=True)
    best_epoch_baseline = fetch_best_epoch(output_baseline)
    
    # Set the device for performing training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device: {}".format(device))

    system_state_dim = 13
    system_control_dim = 4

    # Load the proposed model
    model = PINNAirModel(system_state_dim=system_state_dim, system_control_dim=system_control_dim, num_layers=3, dropout=0.1, device=device)
    model = load_best_model(best_epoch, model, output_dir=output_dir, device=device)
    model.eval()

    # Load the baseline model
    baseline_model = BaselineModel(system_state_dim=system_state_dim, system_control_dim=system_control_dim, num_layers=3, dropout=0.1, device=device)
    baseline_model = load_best_model(best_epoch_baseline, baseline_model, output_dir=output_baseline, device=device)
    baseline_model.eval()
    
    input_window = 50
    output_window = 50

    # Check if rmses have already been computed and then load them
    if os.path.exists(os.path.join(output_dir, "model_error_through_time.npy")) and os.path.exists(os.path.join(output_baseline, "baseline_error_through_time.npy")):
        model_error_through_time = np.load(os.path.join(output_dir, "model_error_through_time.npy"))
        baseline_error_through_time = np.load(os.path.join(output_baseline, "baseline_error_through_time.npy"))
    else:

        # Load the dataset
        test_dataset = MocapDatasetLoader(input_window=input_window, output_window=output_window, stride=1, split="test", device=device)
        print(f'Dataset length {len(test_dataset)}')

        # Initialize the error tensors
        model_error_through_time = np.zeros((len(test_dataset), output_window, 13))
        baseline_error_through_time = np.zeros((len(test_dataset), output_window, 13))

        # For each dataset in the test set
        for seq in tqdm(range(0, len(test_dataset))):
            
            # Load the data
            x, y = test_dataset[seq]
            x = x.to(device)
            y = y.to(device)            

            # Predict the output of the model using the neural network and transfer it to the cpu
            # Note: The model does not use all the variables that are in the dataset
            u = y[..., 19:23]
            x = torch.cat([x[:, 0:13], x[:, 19:23]], dim=-1)

            # Predict the output of the model using only the baseline model (trained without physics information) and transfer it to the cpu
            baseline_model_result = baseline_model(x[None, :, :], u[None, :, :]).to("cpu")

            # Predict using the proposed model
            y_hat = model(x[None, :, :], u[None, :, :]).to("cpu")

            # Send the grountruth back to the CPU to compute the prediction errors of both models
            y = y.to("cpu")
            
            # Compute the error of the model
            model_error_through_time[seq, :, :] = compute_error(y_hat=y_hat.squeeze(), y=y[:, 0:13]).detach().numpy()
            
            # Compute the error of the baseline model
            baseline_error_through_time[seq, :, :] = compute_error(y_hat=baseline_model_result.squeeze()[:, 0:13], y=y[:, 0:13]).detach().numpy()

        # Save RMSE metrics to file
        np.save(os.path.join(output_dir, "model_error_through_time.npy"), model_error_through_time)
        np.save(os.path.join(output_baseline, "baseline_error_through_time.npy"), baseline_error_through_time)
    
    '''
    Error metrics
    '''

    # Plot the error through time
    plot_error_through_time(model_error_through_time, baseline_error_through_time, output_window, output_dir)

if __name__ == "__main__":
    main()
    
