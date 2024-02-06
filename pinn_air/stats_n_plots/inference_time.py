#!/usr/bin/env python3
"""
| File: inference_time.py
| Authors: Gil Serrano, Marcelo Jacinto, Jose Gomes and Joao Pinto
| License: BSD-3-Clause. Copyright (c) 2023, Gil Serrano, Marcelo Jacinto, Jose Gomes and Joao Pinto. All rights reserved.
| Description: Plot the predictions of the model against the real values over a prediction horizon.
"""
import os
import torch
import numpy as np

# Models
from pinn_air.models.pinn_air_model import PINNAirModel
from pinn_air.models.baseline_model import BaselineModel
from pinn_air.dataset.mocap_dataset import MocapDatasetLoader

# RMSE metrics and utils
from pinn_air.utils.utils import load_best_model, fetch_best_epoch, ArgsParser


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

    # Load the model
    input_window = 50
    output_window = 50

    model = models[args_parser.args.model]
    model = load_best_model(best_epoch, model, output_dir=output_dir, device=device)
    model.eval()
    
    # Load the test dataset
    test_dataset = MocapDatasetLoader(input_window=input_window, output_window=output_window, stride=1, split="test", device=device)

    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = len(test_dataset)
    timings=np.zeros((repetitions,1))

    # Warmup procedure
    for i in range(10):
        
        x, y = test_dataset[i]
        x = x.to(device)
        y = y.to(device)

        # Generate the u from the y vector
        u = y[..., 19:23]
        x = torch.cat([x[:, 0:13], x[:, 19:23]], dim=-1)
        
        _ = model(x[None, :, :], u[None, :, :], output_window)

    # MEASURE THE INFERENCE PERFORMANCE
    with torch.no_grad():
        for i in range(repetitions):

            print("Processing sample {}/{}".format(i, len(test_dataset)))

            x, y = test_dataset[i]
            x = x.to(device)
            y = y.to(device)

            # Generate the u from the y vector
            u = y[..., 19:23]
            x = torch.cat([x[:, 0:13], x[:, 19:23]], dim=-1)

            # Record the time it takes to make the prediction
            starter.record()
            _ = model(x[None, :, :], u[None, :, :], output_window)
            ender.record()
            
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[i] = curr_time

    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)

    print("Average inference time for M={:.2f} and N={:.2f}:  {:.2f} ms.".format(input_window, output_window, mean_syn))
    print("Standard deviation: {:.2f} ms".format(std_syn))

if __name__ == "__main__":
    main()
    
