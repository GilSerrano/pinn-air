#!/usr/bin/env python3
"""
| File: plots.py
| Authors: Gil Serrano, Marcelo Jacinto, Jose Gomes and Joao Pinto
| License: BSD-3-Clause. Copyright (c) 2023, Gil Serrano, Marcelo Jacinto, Jose Gomes and Joao Pinto. All rights reserved.
| Description: Plot the predictions of the model against the real values over a prediction horizon.
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
from pinn_air.stats_n_plots.rmse import compute_all_rmse

class Plot:

    def __init__(self, x, y, y_hat, time, time2, y_physics=[], name='plot'):
        self.name = name
        self.name_3d = self.name+'3d'
        self.x = x
        self.y = y
        self.y_hat = y_hat
        self.y_physics = y_physics
        self.time = time
        self.time2 = time2

        # (plane, (elev, azim, roll))
        Plot.VIEWS = [('XY',   (90, -90, 0)),
                      ('XZ',    (0, -90, 0)),
                      ('YZ',    (0,   0, 0))]

        self.fig = plt.figure(num=self.name, figsize=(4, 2), dpi=300)
        self.fig_3d_dict = {
            'XY': plt.figure(num=self.name_3d+'XY', figsize=(5, 5), dpi=300),
            'XZ': plt.figure(num=self.name_3d+'XZ', figsize=(5, 5), dpi=300),
            'YZ': plt.figure(num=self.name_3d+'YZ', figsize=(5, 5), dpi=300)
        }

    def draw(self, output_dir, **kwargs):

        sz = self.x.size(1)
        if sz > 3:
            idx = [1, 2, 3]
        else:
            idx = [0, 1, 2]
        
        gw_lbl = kwargs.get('gw_lbl', "$w$")
        gx_lbl = kwargs.get('gx_lbl', "$x$")
        gy_lbl = kwargs.get('gy_lbl', "$y$")
        gz_lbl = kwargs.get('gz_lbl', "$z$")

        with torch.no_grad():

            # set right figure
            plt.figure(num=self.name)
        
            if sz > 3:
                w_last = self.x[-1, 0] * torch.ones(1)
                plt.plot(self.time2, torch.cat((w_last, self.y_hat[0, :, 0]), dim=0).numpy(), linestyle="--", color="orange", label=gw_lbl)
                plt.plot(self.time, torch.cat((self.x[:, 0], self.y[:, 0]), dim=0).numpy(), color=(1, 220/255, 0.0, 1.0))

                if len(self.y_physics): # check if physics evolution is to be plotted
                    plt.plot(self.time, torch.cat((self.x[:, 0], self.y_physics[0, :, 0]), dim=0).numpy(), linestyle=":", color=(1, 180/255, 0.0, 1.0))

            x_last, y_last, z_last = self.x[-1, idx[0]] * torch.ones(1), \
                                        self.x[-1, idx[1]] * torch.ones(1), self.x[-1, idx[2]] * torch.ones(1)
            
            plt.plot(self.time2, torch.cat((x_last, self.y_hat[0, :, idx[0]]), dim=0).numpy(), linestyle="--", color="red", label=gx_lbl)
            plt.plot(self.time2, torch.cat((y_last, self.y_hat[0, :, idx[1]]), dim=0).numpy(), linestyle="--", color="green", label=gy_lbl)
            plt.plot(self.time2, torch.cat((z_last, self.y_hat[0, :, idx[2]]), dim=0).numpy(), linestyle="--", color="blue", label=gz_lbl)

            # Plot the real quantity
            plt.plot(self.time, torch.cat((self.x[:, idx[0]], self.y[:, idx[0]]), dim=0).numpy(), color=(1,0,0,0.6))
            plt.plot(self.time, torch.cat((self.x[:, idx[1]], self.y[:, idx[1]]), dim=0).numpy(), color=(0,1,0,0.7))
            plt.plot(self.time, torch.cat((self.x[:, idx[2]], self.y[:, idx[2]]), dim=0).numpy(), color=(0,0,1,0.5))

            # Plot the time evolution of the multirotor dynamics without load
            if len(self.y_physics):
                plt.plot(self.time, torch.cat((self.x[:, idx[0]], self.y_physics[0, :, idx[0]]), dim=0).numpy(), linestyle=":", color=(1,0,0,0.3))
                plt.plot(self.time, torch.cat((self.x[:, idx[1]], self.y_physics[0, :, idx[1]]), dim=0).numpy(), linestyle=":", color=(0,1,0,0.4))
                plt.plot(self.time, torch.cat((self.x[:, idx[2]], self.y_physics[0, :, idx[2]]), dim=0).numpy(), linestyle=":", color=(0,0,1,0.2))

            ylabel = kwargs.get('ylabel', "Position (m)")
            lgd_location = kwargs.get('lgd_location', "upper left")

            plt.xlabel("Time (s)", fontsize=10)
            plt.ylabel(ylabel, fontsize=10)
            plt.legend(loc=lgd_location, fontsize=10)
            plt.rc('xtick', labelsize=10)
            plt.rc('ytick', labelsize=10)
            self.fig.subplots_adjust(bottom=0.25, top=0.9, left=0.15, right=0.9)
            plt.grid()

            plt.savefig(output_dir+'/'+self.name+'.pdf')

    def draw3d_primary_planes(self, output_dir, **kwargs):

        gx_lbl = kwargs.get('gx_lbl', "X position (m)")
        gy_lbl = kwargs.get('gy_lbl', "Y position (m)")
        gz_lbl = kwargs.get('gz_lbl', "Z position (m)")
        ax_title = kwargs.get('ax_title', "Position")
        lgd_location = kwargs.get('lgd_location', "best")

        for plane, angles in Plot.VIEWS:
            # set right figure
            plt.figure(num=self.name_3d+plane)

            ax = plt.axes(projection='3d')
            ax.set_xlabel(gx_lbl, labelpad=20)
            ax.set_ylabel(gy_lbl, labelpad=20)
            ax.set_zlabel(gz_lbl, labelpad=20)
            ax.set_proj_type('ortho')
            ax.view_init(elev=angles[0], azim=angles[1])

            if plane == 'XY':
                ax.set_zticklabels([])
                ax.set_zlabel('')
                kwargs["lgd_box_to_anchor"] = (0.5, 0.4)
            elif plane == 'XZ':
                ax.set_yticklabels([])
                ax.set_ylabel('')
                kwargs["lgd_box_to_anchor"] = (0.6, 0.5)
            else: #YZ
                ax.set_xticklabels([])
                ax.set_xlabel('')
                kwargs["lgd_box_to_anchor"] = (0.8, 0.5)

            self.draw3d(ax, output_dir, name=self.name_3d+plane, **kwargs)

    def draw3d(self, ax, output_dir, name, **kwargs):

        with torch.no_grad():

            x_last, y_last, z_last = self.x[-1, 0] * torch.ones(1), \
                                    self.x[-1, 1] * torch.ones(1), self.x[-1, 2] * torch.ones(1)

            px_hat = torch.cat((x_last, self.y_hat[0, :, 0]), dim=0).numpy()
            py_hat = torch.cat((y_last, self.y_hat[0, :, 1]), dim=0).numpy()
            pz_hat = torch.cat((z_last, self.y_hat[0, :, 2]), dim=0).numpy()
            ax.plot3D(px_hat, py_hat, pz_hat, linestyle="--", label='Prediction')

            # Initial point
            ax.scatter(x_last, y_last, z_last, c='green', marker='*', s=50)

            px = torch.cat((x_last, self.y[:, 0]), dim=0).numpy()
            py = torch.cat((y_last, self.y[:, 1]), dim=0).numpy()
            pz = torch.cat((z_last, self.y[:, 2]), dim=0).numpy()
            ax.plot3D(px, py, pz, label='Actual')

            # ax.set_xlabel(gx_lbl, labelpad=20)
            # ax.set_ylabel(gy_lbl, labelpad=20)
            # ax.set_zlabel(gz_lbl, labelpad=20)

            lgd_location = kwargs.get('lgd_location', "best")
            lgd_box_to_anchor = kwargs.get('lgd_box_to_anchor', (0.55, 0.8))
            ax.legend(loc=lgd_location, bbox_to_anchor=lgd_box_to_anchor, fontsize=10)
            # ax.legend(loc=lgd_location)
            # ax.set_title(ax_title, fontsize=12)

            ax.tick_params(labelsize=10)
            ax.grid()

            plt.savefig(output_dir + '/' + name + '.pdf')


def plot_estimated_against_real(x, y, y_hat, time, time2, output_dir, y_physics=[]):

    plt.close('all')

    if len(y_physics):
        print('Plotting with physics')
        pos_plot = Plot(x[..., 0:3], y[..., 0:3], y_hat[..., 0:3], time, time2, y_physics=y_physics[..., 0:3], name='position')
        vel_plot = Plot(x[..., 3:6], y[..., 3:6], y_hat[..., 3:6], time, time2,  y_physics=y_physics[..., 3:6], name='velocity')
        qtr_plot = Plot(x[..., 6:10], y[..., 6:10], y_hat[..., 6:10], time, time2, y_physics=y_physics[..., 6:10], name='quaternion')
        ld_plot  = Plot(x[..., 10:13], y[..., 10:13], y_hat[..., 10:13], time, time2, y_physics=y_physics[..., 10:13], name='load')
    else:
        print('Plotting without physics')
        pos_plot = Plot(x[..., 0:3], y[..., 0:3], y_hat[..., 0:3], time, time2, name='position')
        vel_plot = Plot(x[..., 3:6], y[..., 3:6], y_hat[..., 3:6], time, time2, name='velocity')
        qtr_plot = Plot(x[..., 6:10], y[..., 6:10], y_hat[..., 6:10], time, time2, name='quaternion')
        ld_plot  = Plot(x[..., 10:13], y[..., 10:13], y_hat[..., 10:13], time, time2, name='load')

    pos_args = {'gx_lbl': "$p_x$", 'gy_lbl': "$p_y$", 'gz_lbl': "$p_z$", 'ylabel': "Position (m)"}
    pos_plot.draw(output_dir, **pos_args)
    # pos_plot.draw3d_primary_planes(output_dir)

    vel_args = {'gx_lbl': "$v_x$", 'gy_lbl': "$v_y$", 'gz_lbl': "$v_z$", 'ylabel': "Velocity (m/s)"}
    vel_plot.draw(output_dir, **vel_args)

    qtr_args = {'gw_lbl': "$q_w$", 'gx_lbl': "$q_x$", 'gy_lbl': "$q_y$", 'gz_lbl': "$q_z$", 'ylabel': "Quaternion"}
    qtr_plot.draw(output_dir, **qtr_args)

    ld_args = {'gx_lbl': "$p_x^L$", 'gy_lbl': "$p_y^L$", 'gz_lbl': "$p_z^L$", 'ylabel': "Load Position (m)"}
    ld_plot.draw(output_dir, **ld_args)

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
    
    x, y = test_dataset[(len(test_dataset)//2)]
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

    time = test_dataset.Ts * torch.arange(0, x.shape[0] + y.shape[0]).numpy()
    time2 = test_dataset.Ts * torch.arange(x.shape[0], x.shape[0] + y.shape[0]+1).numpy()

    x = x.to("cpu")
    y = y.to("cpu")
    y_hat = y_hat.to("cpu")
    physics_model_result = physics_model_result.to("cpu")

    plot_estimated_against_real(x, y, y_hat, time, time2, output_dir, y_physics=physics_model_result)

    # Compute all the RMSE metrics for this particular trajectory
    compute_all_rmse(y_hat, y)

    # Compute all the RMSE metrics for the physics model
    print("\nRMSE physics model")
    compute_all_rmse(y[None,...], physics_model_result, expand_y=False, compute_load=True)

if __name__ == "__main__":
    main()
    
