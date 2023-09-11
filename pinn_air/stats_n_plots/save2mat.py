#!/usr/bin/env python3
"""
| File: save2mat.py
| Authors: Gil Serrano, Marcelo Jacinto, Jose Gomes and Joao Pinto
| License: BSD-3-Clause. Copyright (c) 2023, Gil Serrano, Marcelo Jacinto, Jose Gomes and Joao Pinto. All rights reserved.
| Description: Evaluate the model on the test set and save the results to a .mat file (used to analsyze the results in MATLAB).
"""
import os
import torch
import scipy.io as sio

# Models, datasets and physics models
from pinn_air.models.pinn_air_model import PINNAirModel
from pinn_air.models.baseline_model import BaselineModel
from pinn_air.dataset.mocap_dataset import MocapDatasetLoader

# Utilities
from pinn_air.utils.utils import load_best_model, fetch_best_epoch, ArgsParser

def save2mat(dataset, model, output_dir="./output", device="cpu"):

    output_dir = os.path.join(output_dir, "mat")
    os.makedirs(output_dir, exist_ok=True)

    for i in range(len(dataset)):

        print("Processing sample {}/{}".format(i, len(dataset)))

        x, y = dataset[i]
        x = x.to(device)
        y = y.to(device)

        # NOTE: we must change these values if using the baseline model without a payload and just a drone
        u = y[..., 19:23]
        x = torch.cat([x[:, 0:13], x[:, 19:23]], dim=-1)    # The model the does not use load linear and angular velocities
        y_hat = model(x[None, :, :], u[None, :, :])

        x = x.to("cpu")
        y = y.to("cpu")
        y_hat = y_hat.to("cpu")

        with torch.no_grad():

            # create dictionary
            dict = {
                'pos_input': x[..., 0:3].numpy(),
                'vel_input': x[..., 3:6].numpy(),
                'qtr_input': x[..., 6:10].numpy(),
                'ld_input' : x[..., 10:13].numpy(),
                'pos_output': y[..., 0:3].numpy(),
                'vel_output': y[..., 3:6].numpy(),
                'qtr_output': y[..., 6:10].numpy(),
                'ld_output' : y[..., 10:13].numpy(),
                'pos_pred': y_hat[..., 0:3].squeeze(0).numpy(),
                'vel_pred': y_hat[..., 3:6].squeeze(0).numpy(),
                'qtr_pred': y_hat[..., 6:10].squeeze(0).numpy(),
                'ld_pred' : y_hat[..., 10:13].squeeze(0).numpy(),
            }

        # save dataset as MAT file
        filename = os.path.join(output_dir,'test_'+'{:0>4}'.format(i)+'.mat')
        sio.savemat(filename, dict)

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

    # Load the model
    model = models[args_parser.args.model]
    model = load_best_model(best_epoch, model, output_dir=output_dir, device=device)
    model.eval()

    # Load the test dataset
    input_window = 50
    output_window = 25
    test_dataset = MocapDatasetLoader(input_window=input_window, output_window=output_window, stride=1, split="test", device=device)

    # Evaluate the model and save the results to a .mat file (used to analsyze the results in MATLAB)
    save2mat(test_dataset, model, output_dir=output_dir, device=device)

if __name__ == "__main__":
    main()