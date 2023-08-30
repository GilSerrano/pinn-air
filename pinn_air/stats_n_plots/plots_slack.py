#!/usr/bin/env python3
"""
| File: plots_slack.py
| Authors: Gil Serrano, Marcelo Jacinto, Jose Gomes and Joao Pinto
| License: BSD-3-Clause. Copyright (c) 2023, Gil Serrano, Marcelo Jacinto, Jose Gomes and Joao Pinto. All rights reserved.
| Description: Plots the slack variables of the system over a time horizon.
"""
import os
import torch
import matplotlib.pyplot as plt

# Models
from pinn_air.models.pinn_air_model import PINNAirModel
from pinn_air.models.baseline_model import BaselineModel
from pinn_air.dataset.mocap_dataset import MocapDatasetLoader
from pinn_air.physics.discrete_model import DiscreteModel

# RMSE metrics and utils
from pinn_air.utils.utils import load_best_model, fetch_best_epoch, ArgsParser

class Plot:

    def __init__(self, y_hat, time, time2, name='plot'):
        self.name = name
        self.y_hat = y_hat
        self.time = time
        self.time2 = time2

    def draw(self, sub_fig, **kwargs):

        train_time=25
        idx = [0, 1, 2]
        
        gw_lbl = kwargs.get('gw_lbl', "$w$")
        gx_lbl = kwargs.get('gx_lbl', "$x$")
        gy_lbl = kwargs.get('gy_lbl', "$y$")
        gz_lbl = kwargs.get('gz_lbl', "$z$")

        # Create a vector of zeros for the time
        zeros = torch.zeros(self.time.shape[0]).to("cpu").detach().numpy()

        if self.name == 'quaternion':
            sub_fig.plot(self.time2, self.y_hat[0, :, 3], color="yellow", label=gw_lbl)
        
        #sub_fig.plot(self.time, zeros, color="red")
        sub_fig.plot(self.time2, self.y_hat[0, :, idx[0]], color="red", label=gx_lbl)
        #sub_fig.plot(self.time, zeros, color="green")
        sub_fig.plot(self.time2, self.y_hat[0, :, idx[1]], color="green", label=gy_lbl)
        #sub_fig.plot(self.time, zeros, color="blue")
        sub_fig.plot(self.time2, self.y_hat[0, :, idx[2]], color="blue", label=gz_lbl)

       

        # Plot the real quantity
        sub_fig.axvline(x=self.time2[train_time], color = 'black', ls='--')

        ylabel = kwargs.get('ylabel', "Position (m)")
        lgd_location = kwargs.get('lgd_location', "upper left")

        sub_fig.legend(loc=lgd_location, fontsize=10)
        sub_fig.set(ylabel=ylabel)
        sub_fig.grid()


def plot_slack_variables(y_hat, time, time2, output_dir):

    plt.close('all')

    pos_plot = Plot(y_hat[..., 13:16], time, time2, name='position')
    vel_plot = Plot(y_hat[..., 16:19], time, time2, name='velocity')
    qtr_plot = Plot(y_hat[..., 19:23], time, time2, name='quaternion')
    ld_plot  = Plot(y_hat[..., 23:26], time, time2, name='load')

    pos_args = {'gx_lbl': "$p_x$", 'gy_lbl': "$p_y$", 'gz_lbl': "$p_z$", 'ylabel': "Position (m)"}
    vel_args = {'gx_lbl': "$v_x$", 'gy_lbl': "$v_y$", 'gz_lbl': "$v_z$", 'ylabel': "Velocity (m/s)"}
    qtr_args = {'gw_lbl': "$q_w$", 'gx_lbl': "$q_x$", 'gy_lbl': "$q_y$", 'gz_lbl': "$q_z$", 'ylabel': "Quaternion"}
    ld_args = {'gx_lbl': "$p_x^L$", 'gy_lbl': "$p_y^L$", 'gz_lbl': "$p_z^L$", 'ylabel': "Load Position (m)"}

    # Create the figure with subfigures to make the plots
    fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=False)

    plots = [pos_plot, vel_plot, qtr_plot, ld_plot]
    plots_args = [pos_args, vel_args, qtr_args, ld_args]

    # Perform the plots
    i = 0
    for row in ax:
        for col in row:
            plots[i].draw(col, **plots_args[i])
            i += 1

    plt.setp(ax[-1, :], xlabel='Time (s)')
    fig.tight_layout()
    plt.savefig(output_dir+'/slack_variables.pdf')
    

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
    test_dataset = MocapDatasetLoader(input_window=input_window, output_window=output_window, stride=1, split="test", device=device)
    
    x, y = test_dataset[(len(test_dataset)//35)]
    x = x.to(device)
    y = y.to(device)

    print(f'Dataset length {len(test_dataset)}')

    '''
    Physics Model result calculation
    '''
    physics_model = DiscreteModel(Ts=0.03, mQ=1.35, mL=0.1, l=0.7)
    physics_model_result = torch.zeros((output_window, 19)).to(device)
    print(f'physics_model_result shape {physics_model_result.shape}')

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
    print(f'physics_model_result shape {physics_model_result.shape}')

    u = y[..., 19:23]
    x = torch.cat([x[:, 0:13], x[:, 19:23]], dim=-1)
    y_hat = model(x[None, :, :], u[None, :, :])

    # Time for the inputs of the system
    time = test_dataset.Ts * torch.arange(0, x.shape[0] + 1).numpy()

    # Time for the outputs of the network
    time2 = test_dataset.Ts * torch.arange(x.shape[0], x.shape[0] + y.shape[0]).numpy()
    y_hat = y_hat.to("cpu").detach().numpy()

    plot_slack_variables(y_hat, time, time2, output_dir)

if __name__ == "__main__":
    main()
    
