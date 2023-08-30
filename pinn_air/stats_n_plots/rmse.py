#!/usr/bin/env python3
"""
| File: rmse.py
| Authors: Gil Serrano, Marcelo Jacinto, Jose Gomes and Joao Pinto
| License: BSD-3-Clause. Copyright (c) 2023, Gil Serrano, Marcelo Jacinto, Jose Gomes and Joao Pinto. All rights reserved.
| Description: Compute the RMSE of the neural network model and the physics model over the entire test set (used in the paper).
"""
import os
import torch

# Models, datasets and physics models
from pinn_air.physics.discrete_model import DiscreteModel
from pinn_air.models.pinn_air_model import PINNAirModel
from pinn_air.models.baseline_model import BaselineModel
from pinn_air.dataset.mocap_dataset import MocapDatasetLoader

# Utilities
from pinn_air.utils.utils import load_best_model, fetch_best_epoch, ArgsParser
from pinn_air.utils.math_utils import quaternion_multiply, quaternion_invert

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

def position_rmse(y_hat, y):
    
    # Compute the discounted position error
    return torch.sqrt(torch.sum((torch.norm(y_hat[:, :, 0:3] - y[:, :, 0:3], dim=2) ** 2)) / (y.shape[0] * y.shape[1]))

def position_rmse_payload(y_hat, y):
    
    # Compute the discounted position error
    return torch.sqrt(torch.sum((torch.norm(y_hat[:, :, 10:13] - y[:, :, 10:13], dim=2) ** 2)) / (y.shape[0] * y.shape[1]))

def velocity_rmse(y_hat, y):

    # Compute the discounted velocity error
    return torch.sqrt(torch.sum((torch.norm(y_hat[:, :, 3:6] - y[:, :, 3:6], dim=2) ** 2)) / (y.shape[0] * y.shape[1]))

def quaternion_rmse(y_hat, y):

    # Compute the discounted quaternion error
    identity_quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0]).to(y.device)

    return torch.sqrt(torch.sum((torch.norm(quaternion_multiply(y[..., 6:10], quaternion_invert(y_hat[..., 6:10])) - identity_quaternion, dim=2) ** 2)) / (y.shape[0] * y.shape[1]))

def rmse_all_state(y_hat, y):

    identity_quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0]).to(y.device)

    error_pos_vel = y_hat[:, :, 0:6] - y[:, :, 0:6]
    error_payload = y_hat[:, :, 10:13] - y[:, :, 10:13]
    error_quaternion = quaternion_multiply(y[..., 6:10], quaternion_invert(y_hat[..., 6:10])) - identity_quaternion

    # Concatenate the errors
    error = torch.cat([error_pos_vel, error_payload, error_quaternion], dim=2)

    # Compute the RMSE
    return torch.sqrt(torch.sum((torch.norm(error, dim=2) ** 2)) / (y.shape[0] * y.shape[1]))

def rmse_all_no_load(y_hat, y):
    
    identity_quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0])

    error_pos_vel = y_hat[:, :, 0:6] - y[:, :, 0:6]
    error_quaternion = quaternion_multiply(y[..., 6:10], quaternion_invert(y_hat[..., 6:10])) - identity_quaternion

    # Concatenate the errors
    error = torch.cat([error_pos_vel, error_quaternion], dim=2)

    # Compute the RMSE
    return torch.sqrt(torch.sum((torch.norm(error, dim=2) ** 2)) / (y.shape[0] * y.shape[1]))

def compute_all_rmse(y_hat, y, expand_y=True, compute_load=True):

    y = y[None,...] if expand_y else y

    rmse_position = position_rmse(y_hat, y)
    rmse_velocity = velocity_rmse(y_hat, y)
    rmse_quaternion = quaternion_rmse(y_hat, y)
    print("RMSE position: {}".format(rmse_position))
    print("RMSE velocity: {}".format(rmse_velocity))
    print("RMSE quaternion: {}".format(rmse_quaternion))

    if compute_load:
        rmse_payload = position_rmse_payload(y_hat, y)
        rmse_all = rmse_all_state(y_hat, y)
        print("RMSE all: {}".format(rmse_all))
        print("RMSE payload: {}".format(rmse_payload))
        print("RMSE sum: {}".format(rmse_position + rmse_payload + rmse_velocity + rmse_quaternion))
    else:
        rmse_all = rmse_all_no_load(y_hat, y)
        print("RMSE all: {}".format(rmse_all))
        print("RMSE sum: {}".format(rmse_position + rmse_velocity + rmse_quaternion))


def main():

    args_parser = ArgsParser()

    output_dir = args_parser.args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    best_epoch = fetch_best_epoch(output_dir)

    # Set the device for performing training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device: {}".format(device))

    system_state_dim = 13
    system_control_dim = 4

    models = {
        "PINNAirModel": PINNAirModel(system_state_dim=system_state_dim, system_control_dim=system_control_dim, num_layers=3, dropout=0.1, device=device),
        "BaselineModel": BaselineModel(system_state_dim=system_state_dim, system_control_dim=system_control_dim, num_layers=3, dropout=0.1, device=device)
    }

    model = models[args_parser.args.model]
    model = load_best_model(best_epoch, model, output_dir=output_dir, device=device)
    model.eval()

    # Load the model
    input_window = 50
    output_window = 50

    test_dataset = MocapDatasetLoader(input_window=input_window, output_window=output_window, stride=1, split="test", device=device)

    # Acumulate the real and predicted values
    total_y = []
    total_y_hat = []
    total_y_physics = []

    # Perform the predictions across the entire test set
    for i in range(len(test_dataset)):

        print("Processing sample {}/{}".format(i, len(test_dataset)))

        x, y = test_dataset[i]
        x = x.to(device)
        y = y.to(device)

        # Get the physics model result
        physics_model_result = get_physics_model_result(x, y, output_window, Ts=test_dataset.Ts, mQ=1.35, mL=0.1, l=0.7).to("cpu")

        # Generate the u from the y vector
        u = y[..., 19:23]
        x = torch.cat([x[:, 0:13], x[:, 19:23]], dim=-1)
        y_hat = model(x[None, :, :], u[None, :, :], output_window)

        total_y.append(y)
        total_y_hat.append(y_hat)
        total_y_physics.append(physics_model_result)

    total_y = torch.stack(total_y, dim=0).to(device)
    total_y_hat = torch.cat(total_y_hat, dim=0).to(device)
    total_y_physics = torch.cat(total_y_physics, dim=0).to(device)

    # Compute the RMSE
    print("Neural network model")
    compute_all_rmse(total_y_hat, total_y, expand_y=False)

    print("Physics model")
    compute_all_rmse(total_y_physics, total_y, expand_y=False)

if __name__ == "__main__":
    main()