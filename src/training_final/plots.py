#!/usr/bin/env python3

import os
import torch
import matplotlib.pyplot as plt
from mocap_dataset import MocapDatasetLoader

from alpha_model import AlphaModel
from model import SuperModelo

def fetch_best_epoch(output_dir="output/", file="best_epoch.txt"):
    """
        fetch the model from the best epoch
        it corresponds to the last line in the file
    """

    with open(os.path.join(output_dir, file), "rb") as f:
        try:  # catch OSError in case of a one line file 
            f.seek(-2, os.SEEK_END)
            while f.read(1) != b'\n':
                f.seek(-2, os.SEEK_CUR)
        except OSError:
            f.seek(0)
        best_epoch = f.readline().decode()

    return best_epoch


def load_best_model(best_epoch, model, output_dir="./output", device="cpu"):
    
    print(f"epoch_{best_epoch}_best_model.pt")

    checkpoint_path = os.path.join(output_dir, f"epoch_{best_epoch}_best_model.pt")
    print("Loading: {}".format(checkpoint_path))

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])

    return model.to(device)

class Plot:

    def __init__(self, x, y, y_hat, time, time2, name='plot'):
        self.name = name
        self.x = x
        self.y = y
        self.y_hat = y_hat
        self.time = time
        self.time2 = time2

        self.fig = plt.figure(num=self.name, figsize=(4, 2), dpi=300)

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

            x_last, y_last, z_last = self.x[-1, idx[0]] * torch.ones(1), \
                                        self.x[-1, idx[1]] * torch.ones(1), self.x[-1, idx[2]] * torch.ones(1)
            
            plt.plot(self.time2, torch.cat((x_last, self.y_hat[0, :, idx[0]]), dim=0).numpy(), linestyle="--", color="red", label=gx_lbl)
            plt.plot(self.time2, torch.cat((y_last, self.y_hat[0, :, idx[1]]), dim=0).numpy(), linestyle="--", color="green", label=gy_lbl)
            plt.plot(self.time2, torch.cat((z_last, self.y_hat[0, :, idx[2]]), dim=0).numpy(), linestyle="--", color="blue", label=gz_lbl)

            # Plot the real quantity
            plt.plot(self.time, torch.cat((self.x[:, idx[0]], self.y[:, idx[0]]), dim=0).numpy(), color=(1,0,0,0.6))
            plt.plot(self.time, torch.cat((self.x[:, idx[1]], self.y[:, idx[1]]), dim=0).numpy(), color=(0,1,0,0.7))
            plt.plot(self.time, torch.cat((self.x[:, idx[2]], self.y[:, idx[2]]), dim=0).numpy(), color=(0,0,1,0.5))

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


def main():

    # -------------------------------------------------------------------
    # Plots for the regular test were we perform the recursive prediction
    # ------------------------------------------------------------------- 
    # TODO - add plots for the recursive prediction of the physics model
    output_dir = './output'
    best_epoch = fetch_best_epoch(output_dir)
    print(best_epoch)
    
    # Set the device for performing training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device: {}".format(device))

    model = SuperModelo(device)
    model = load_best_model(best_epoch, model, output_dir=output_dir, device=device)

    target_time = 25
    test_dataset = MocapDatasetLoader(input_window=50, output_window=target_time, stride=1, split="test", device=device)
    
    # sampling time
    Ts = 0.03

    x, y = test_dataset[0]
    x = x.to(device)
    y = y.to(device)

    u = y[..., 13:17]
    y_hat = model(x[None, :, :], u[None, :, :])

    time = Ts * torch.arange(0, x.shape[0] + y.shape[0]).numpy()
    time2 = Ts * torch.arange(x.shape[0], x.shape[0] + y.shape[0]+1).numpy()

    x = x.to("cpu")
    y = y.to("cpu")
    y_hat = y_hat.to("cpu")

    pos_plot = Plot(x[..., 0:3], y[..., 0:3], y_hat[..., 0:3], time, time2, name='position')
    vel_plot = Plot(x[..., 3:6], y[..., 3:6], y_hat[..., 3:6], time, time2, name='velocity')
    qtr_plot = Plot(x[..., 6:10], y[..., 6:10], y_hat[..., 6:10], time, time2, name='quaternion')
    ld_plot  = Plot(x[..., 10:13], y[..., 10:13], y_hat[..., 10:13], time, time2, name='load')

    pos_args = {'gx_lbl': "$p_x$", 'gy_lbl': "$p_y$", 'gz_lbl': "$p_z$", 'ylabel': "Position (m)"}
    pos_plot.draw(output_dir, **pos_args)

    vel_args = {'gx_lbl': "$v_x$", 'gy_lbl': "$v_y$", 'gz_lbl': "$v_z$", 'ylabel': "Velocity (m/s)"}
    vel_plot.draw(output_dir, **vel_args)

    qtr_args = {'gw_lbl': "$q_w$", 'gx_lbl': "$q_x$", 'gy_lbl': "$q_y$", 'gz_lbl': "$q_z$", 'ylabel': "Quaternion"}
    qtr_plot.draw(output_dir, **qtr_args)

    ld_args = {'gx_lbl': "$p_x^L$", 'gy_lbl': "$p_y^L$", 'gz_lbl': "$p_z^L$", 'ylabel': "Load Position (m)"}
    ld_plot.draw(output_dir, **ld_args)

    # -------------------------------------------------------------------
    # Plots for the regular test were we perform 1 step prediction
    # ------------------------------------------------------------------- 
    target_time = 25
    input_time = 50
    test_dataset = MocapDatasetLoader(input_window=input_time+target_time, output_window=target_time, stride=1, split="test", device=device)

    x, y = test_dataset[0]
    x = x.to(device)
    y = y.to(device)

    for i in target_time:

        # Get the input state and the input control for this timestep
        input = x[None, i:i+input_time, :]
        u = y[None, i:i+target_time, 13:17]

        # Perform the prediction
        y_hat = model(input, u)

        # Discard all the timesteps predicted except the first one
        y_hat = y_hat[0, 0, :]


    

    # -------------------------------------------------------------------
    # Plots of physics only, Network only and groundtruth
    # -------------------------------------------------------------------



if __name__ == "__main__":
    main()
    
