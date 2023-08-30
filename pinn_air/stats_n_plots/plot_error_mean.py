#!/usr/bin/env python3
"""
| File: plot_error_mean.py
| Authors: Gil Serrano, Marcelo Jacinto, Jose Gomes and Joao Pinto
| License: BSD-3-Clause. Copyright (c) 2023, Gil Serrano, Marcelo Jacinto, Jose Gomes and Joao Pinto. All rights reserved.
| Description: Plots the mean error and standard deviation of the error of the proposed model against the physics model (used in the paper).
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
from pinn_air.physics.discrete_model import DiscreteModel
from pinn_air.physics.discrete_model import DiscreteModel

# RMSE metrics and utils
from pinn_air.utils.utils import load_best_model, fetch_best_epoch, ArgsParser

def get_physics_model_result(x, y, output_window, Ts, mQ, mL, l):
    # Given a sequence of inputs and outputs, compute the prediction of the physics model
    # over the output_window period horizon
    
    # Create the physics model and an empty tensor to store the results
    physics_model = DiscreteModel(Ts=Ts, mQ=mQ, mL=mL, l=l)
    physics_model_result = torch.zeros((output_window, 19)).to(x.device)

    # Get the last state in time that will be fed into the network along with the first input
    x0 = x[-1, 0:19]       # [x,y,z | vx,vy,vz | qw,qx,qy,qz | plx,ply,plz | vlx, vly, vlz | wlx, wly, wlz]
    u0 = x[-1, 19:23]      # [wx, wy, wz | F]

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

    pos_error = y_hat[..., 0:3] - y[...,0:3]
    vel_error = y_hat[..., 3:6] - y[...,3:6]
    payload_error = y_hat[...,10:13] - y[...,10:13]

    # Compute the quaternion error
    identity_quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0]).to(pos_error.device)
    quat_error = quaternion_multiply(y[..., 6:10], quaternion_invert(y_hat[..., 6:10])) - identity_quaternion

    # Computer the rest of the errors for the other states
    return torch.cat((pos_error, vel_error, quat_error, payload_error), dim=-1)


def plot_error_through_time(model_error_through_time, physics_error_through_time, output_window, output_dir):

    Ts = 0.03

    # Compute the norm of the error from the errors through time
    pos_error_through_time = np.linalg.norm(model_error_through_time[:,:,0:3], ord=2, axis=-1)
    vel_error_through_time = np.linalg.norm(model_error_through_time[:,:,3:6], ord=2, axis=-1)
    quat_error_through_time = np.linalg.norm(model_error_through_time[:,:,6:10], ord=2, axis=-1)
    payload_error_through_time = np.linalg.norm(model_error_through_time[:,:,10:13], ord=2, axis=-1)

    # Compute the norm of the physics error from the errors through time
    pos_physics_error_through_time = np.linalg.norm(physics_error_through_time[:,:,0:3], ord=2, axis=-1)
    vel_physics_error_through_time = np.linalg.norm(physics_error_through_time[:,:,3:6], ord=2, axis=-1)
    quat_physics_error_through_time = np.linalg.norm(physics_error_through_time[:,:,6:10], ord=2, axis=-1)
    payload_physics_error_through_time = np.linalg.norm(physics_error_through_time[:,:,10:13], ord=2, axis=-1)

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
    pos_physics_error_through_time_mean = np.mean(pos_physics_error_through_time, axis=0)
    vel_physics_error_through_time_mean = np.mean(vel_physics_error_through_time, axis=0)
    quat_physics_error_through_time_mean = np.mean(quat_physics_error_through_time, axis=0)
    payload_physics_error_through_time_mean = np.mean(payload_physics_error_through_time, axis=0)

    # Compute the physics std through time
    pos_physics_error_through_time_std = np.std(pos_physics_error_through_time, axis=0)
    vel_physics_error_through_time_std = np.std(vel_physics_error_through_time, axis=0)
    quat_physics_error_through_time_std = np.std(quat_physics_error_through_time, axis=0)
    payload_physics_error_through_time_std = np.std(payload_physics_error_through_time, axis=0)

    # Compute the physics min through time
    pos_physics_error_through_time_min = np.min(pos_physics_error_through_time, axis=0)
    vel_physics_error_through_time_min = np.min(vel_physics_error_through_time, axis=0)
    quat_physics_error_through_time_min = np.min(quat_physics_error_through_time, axis=0)
    payload_physics_error_through_time_min = np.min(payload_physics_error_through_time, axis=0)

    # Compute the physics max through time
    pos_physics_error_through_time_max = np.max(pos_physics_error_through_time, axis=0)
    vel_physics_error_through_time_max = np.max(vel_physics_error_through_time, axis=0)
    quat_physics_error_through_time_max = np.max(quat_physics_error_through_time, axis=0)
    payload_physics_error_through_time_max = np.max(payload_physics_error_through_time, axis=0)

    # Create the time vector using a sample rate of 0.03 seconds
    timesteps = np.arange(0, output_window * Ts, Ts)

    '''
    Position
    '''

    fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=False)
    # Neural Network
    ax[0,0].plot(timesteps, pos_error_through_time_mean, color='g', label="Neural Network")
    ax[0,0].fill_between(timesteps, pos_error_through_time_mean - pos_error_through_time_std, pos_error_through_time_mean + pos_error_through_time_std, color='g', alpha=0.2)
    # Physics
    ax[0,0].plot(timesteps, pos_physics_error_through_time_mean, color='r', label="Physical Model")
    ax[0,0].fill_between(timesteps, pos_physics_error_through_time_mean - pos_physics_error_through_time_std, pos_physics_error_through_time_mean + pos_physics_error_through_time_std, color='r', alpha=0.2)
    ax[0,0].set(ylabel="Position Error (m)")
    ax[0,0].axvline(x = 25 * Ts, color = 'black', ls='--')
    ax[0,0].legend()
    #ax[0,0].grid()

    '''
    Velocity
    '''

    # Neural Network
    ax[0,1].plot(timesteps, vel_error_through_time_mean, color='g', label="Neural Network")
    ax[0,1].fill_between(timesteps, vel_error_through_time_mean - vel_error_through_time_std, vel_error_through_time_mean + vel_error_through_time_std, color='g', alpha=0.2)
    # Physics
    ax[0,1].plot(timesteps, vel_physics_error_through_time_mean, color='r', label="Physical Model")
    ax[0,1].fill_between(timesteps, vel_physics_error_through_time_mean - vel_physics_error_through_time_std, vel_physics_error_through_time_mean + vel_physics_error_through_time_std, color='r', alpha=0.2)
    ax[0,1].set(ylabel="Velocity Error (m/s)")
    ax[0,1].axvline(x = 25 * Ts, color = 'black', ls='--')
    ax[0,1].legend()
    #ax[0,1].grid()

    '''
    Quaternion
    '''
    ax[1,0].plot(timesteps, quat_error_through_time_mean, color='g', label="Neural Network")
    ax[1,0].fill_between(timesteps, quat_error_through_time_mean - quat_error_through_time_std, quat_error_through_time_mean + quat_error_through_time_std, color='g', alpha=0.2)
    # Physics
    ax[1,0].plot(timesteps, quat_physics_error_through_time_mean, color='r', label="Physical Model")
    ax[1,0].fill_between(timesteps, quat_physics_error_through_time_mean - quat_physics_error_through_time_std, quat_physics_error_through_time_mean + quat_physics_error_through_time_std, color='r', alpha=0.2)
    ax[1,0].set(ylabel="Quaternion Error Norm")
    ax[1,0].axvline(x = 25 * Ts, color = 'black', ls='--')
    ax[1,0].legend()
    #ax[1,0].grid()

    '''
    Payload position
    '''

    # Neural Network
    ax[1,1].plot(timesteps, payload_error_through_time_mean, color='g', label="Neural Network")
    ax[1,1].fill_between(timesteps, payload_error_through_time_mean - payload_error_through_time_std, payload_error_through_time_mean + payload_error_through_time_std, color='g', alpha=0.2)
    # Physics
    ax[1,1].plot(timesteps, payload_physics_error_through_time_mean, color='r', label="Physical Model")
    ax[1,1].fill_between(timesteps, payload_physics_error_through_time_mean - payload_physics_error_through_time_std, payload_physics_error_through_time_mean + payload_physics_error_through_time_std, color='r', alpha=0.2)
    ax[1,1].set(ylabel="Load Position Error (m)")
    ax[1,1].axvline(x = 25 * Ts, color = 'black', ls='--')
    ax[1,1].legend()
    #ax[1,1].grid()

    #fig.savefig(os.path.join(output_dir, "error_std_norm_states.png"))
    plt.setp(ax[-1, :], xlabel='Time (s)')
    fig.tight_layout()
    plt.savefig(output_dir + '/' + "std_avg.pdf")


    # Create a new figure
    plt.close('all')
    fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=False)
    # -------------------------------------------------------------------
    # Plots for the Min and Max values
    # -------------------------------------------------------------------

    '''
    Position
    '''

    # Neural Network
    ax[0,0].plot(timesteps, pos_error_through_time_mean, color='g', label="Neural Network")
    ax[0,0].fill_between(timesteps, pos_error_through_time_min, pos_error_through_time_max, color='g', alpha=0.2)
    # Physics
    ax[0,0].plot(timesteps, pos_physics_error_through_time_mean, color='r', label="Physical Model")
    ax[0,0].fill_between(timesteps, pos_physics_error_through_time_min, pos_physics_error_through_time_max, color='r', alpha=0.2)
    ax[0,0].set(ylabel="Position Error (m)")
    ax[0,0].axvline(x = 25 * Ts, color = 'black', ls='--')
    ax[0,0].legend()
    #ax[0,0].grid()

    '''
    Velocity
    '''

    # Neural Network
    ax[0,1].plot(timesteps, vel_error_through_time_mean, color='g', label="Neural Network")
    ax[0,1].fill_between(timesteps, vel_error_through_time_min, vel_error_through_time_max, color='g', alpha=0.2)
    # Physics
    ax[0,1].plot(timesteps, vel_physics_error_through_time_mean, color='r', label="Physical Model")
    ax[0,1].fill_between(timesteps, vel_physics_error_through_time_min, vel_physics_error_through_time_max, color='r', alpha=0.2)
    ax[0,1].set(ylabel="Velocity Error (m)")
    ax[0,1].axvline(x = 25 * Ts, color = 'black', ls='--')
    ax[0,1].legend()
    #ax[0,1].grid()

    '''
    Quaternion
    '''

    # Neural Network
    ax[1,0].plot(timesteps, quat_error_through_time_mean, color='g', label="Neural Network")
    ax[1,0].fill_between(timesteps, quat_error_through_time_min, quat_error_through_time_max, color='g', alpha=0.2)
    # Physics
    ax[1,0].plot(timesteps, quat_physics_error_through_time_mean, color='r', label="Physical Model")
    ax[1,0].fill_between(timesteps, quat_physics_error_through_time_min, quat_physics_error_through_time_max, color='r', alpha=0.2)
    ax[1,0].set(ylabel="Quaternion Error Norm (m)")
    ax[1,0].axvline(x = 25 * Ts, color = 'black', ls='--')
    ax[1,0].legend()
    #ax[1,0].grid()

    '''
    Payload
    '''

    # Neural Network
    ax[1,1].plot(timesteps, payload_error_through_time_mean, color='g', label="Neural Network")
    ax[1,1].fill_between(timesteps, payload_error_through_time_min, payload_error_through_time_max, color='g', alpha=0.2)
    # Physics
    ax[1,1].plot(timesteps, payload_physics_error_through_time_mean, color='r', label="Physical Model")
    ax[1,1].fill_between(timesteps, payload_physics_error_through_time_min, payload_physics_error_through_time_max, color='r', alpha=0.2)
    ax[1,1].set(ylabel="Load Position Error (m)")
    ax[1,1].axvline(x = 25 * Ts, color = 'black', ls='--')
    ax[1,1].legend()
    #ax[1,1].grid()

    plt.setp(ax[-1, :], xlabel='Time (s)')
    fig.tight_layout()
    plt.savefig(output_dir + '/' + "min_max_avg.pdf")


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
    output_window = 50

    # Check if rmses have already been computed and then load them
    if os.path.exists(os.path.join(output_dir, "model_error_through_time_output_window.npy")) and os.path.exists(os.path.join(output_dir, "physics_error_through_time_output_window.npy")):
        model_error_through_time = np.load(os.path.join(output_dir, "model_error_through_time_output_window.npy"))
        physics_error_through_time = np.load(os.path.join(output_dir, "physics_error_through_time_output_window.npy"))
    else:

        # Load the dataset
        test_dataset = MocapDatasetLoader(input_window=input_window, output_window=output_window, stride=1, split="test", device=device)
        print(f'Dataset length {len(test_dataset)}')

        # Initialize the error tensors
        model_error_through_time = np.zeros((len(test_dataset), output_window, 13))
        physics_error_through_time = np.zeros((len(test_dataset), output_window, 13))

        # For each dataset in the test set
        for seq in tqdm(range(0, len(test_dataset))):
            
            # Load the data
            x, y = test_dataset[seq]
            x = x.to(device)
            y = y.to(device)

            # Predict the output of the model using only the physics model and transfer it to the cpu
            physics_model_result = get_physics_model_result(x, y, output_window, Ts=test_dataset.Ts, mQ=1.35, mL=0.1, l=0.7).to("cpu")

            # Predict the output of the model using the neural network and transfer it to the cpu
            # Note: The model does not use all the variables that are in the dataset
            u = y[..., 19:23]
            x = torch.cat([x[:, 0:13], x[:, 19:23]], dim=-1)
            y_hat = model(x[None, :, :], u[None, :, :])

            # Send everything back to the CPU to compute the prediction errors of both models
            x = x.to("cpu")
            y = y.to("cpu")
            y_hat = y_hat.to("cpu")
            
            # Compute the error of the model
            model_error_through_time[seq, :, :] = compute_error(y_hat=y_hat.squeeze(), y=y[:, 0:13]).detach().numpy()
            
            # Compute the error of the physics model
            physics_error_through_time[seq, :, :] = compute_error(y_hat=physics_model_result.squeeze()[:, 0:13], y=y[:, 0:13]).detach().numpy()

        # Save RMSE metrics to file
        np.save(os.path.join(output_dir, "model_error_through_time_output_window.npy"), model_error_through_time)
        np.save(os.path.join(output_dir, "physics_error_through_time_output_window.npy"), physics_error_through_time)
    
    '''
    Error metrics
    '''

    # Plot the error through time
    plot_error_through_time(model_error_through_time, physics_error_through_time, output_window, output_dir)

if __name__ == "__main__":
    main()
    
