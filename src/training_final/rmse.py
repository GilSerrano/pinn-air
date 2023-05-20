#!/usr/bin/env python3
import os
import torch
from model import SuperModelo
from mocap_dataset import MocapDatasetLoader
from utils import load_best_model, fetch_best_epoch
from math_utils import quaternion_multiply, quaternion_invert

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
    identity_quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0])

    return torch.sqrt(torch.sum((torch.norm(quaternion_multiply(y[..., 6:10], quaternion_invert(y_hat[..., 6:10])) - identity_quaternion, dim=2) ** 2)) / (y.shape[0] * y.shape[1]))

def rmse_all_state(y_hat, y):

    identity_quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0])

    error_pos_vel = y_hat[:, :, 0:6] - y[:, :, 0:6]
    error_payload = y_hat[:, :, 10:13] - y[:, :, 10:13]
    error_quaternion = quaternion_multiply(y[..., 6:10], quaternion_invert(y_hat[..., 6:10])) - identity_quaternion

    # Concatenate the errors
    error = torch.cat([error_pos_vel, error_payload, error_quaternion], dim=2)

    # Compute the RMSE
    return torch.sqrt(torch.sum((torch.norm(error, dim=2) ** 2)) / (y.shape[0] * y.shape[1]))

def compute_all_rmse(y_hat, y):

    rmse_position = position_rmse(y_hat, y[None,...])
    rmse_velocity = velocity_rmse(y_hat, y[None,...])
    rmse_quaternion = quaternion_rmse(y_hat, y[None,...])
    rmse_payload = position_rmse_payload(y_hat, y[None,...])
    rmse_all = rmse_all_state(y_hat, y[None,...])

    print("RMSE position: {}".format(rmse_position))
    print("RMSE payload: {}".format(rmse_payload))
    print("RMSE velocity: {}".format(rmse_velocity))
    print("RMSE quaternion: {}".format(rmse_quaternion))
    print("RMSE all: {}".format(rmse_all))
    print("RMSE sum: {}".format(rmse_position + rmse_payload + rmse_velocity + rmse_quaternion))


def main():

    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)
    best_epoch = fetch_best_epoch(output_dir)

    # Set the device for performing training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device: {}".format(device))

    model = SuperModelo(device)
    model = load_best_model(best_epoch, model, output_dir=output_dir, device=device)

    # Load the model
    target_time = 25

    test_dataset = MocapDatasetLoader(input_window=50, output_window=target_time, stride=1, split="val", device="cuda")

    # Acumulate the real and predicted values
    total_y = []
    total_y_hat = []

    # Perform the predictions across the entire test set
    for i in range(len(test_dataset)):

        print("Processing sample {}/{}".format(i, len(test_dataset)))

        x, y = test_dataset[i]

        Ts = 0.03

        # Generate the u from the y vector
        u = y[..., 13:17]
        y_hat = model(x[None, :, :], u[None, :, :], target_time)

        total_y.append(y)
        total_y_hat.append(y_hat)

    total_y = torch.stack(total_y, dim=0)
    total_y_hat = torch.cat(total_y_hat, dim=0)

    # Compute the RMSE
    compute_all_rmse(total_y_hat, total_y)


if __name__ == "__main__":
    main()