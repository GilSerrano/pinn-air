import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from data_loaders.mocap_swipe_loader import MocapSwipeLoader

from main import model, load_best_model, train_dataset, target_time

# Load the model
target_time = 25

best_model = torch.load("output2/best_model_idx.pth.tar")
best_model_idx = best_model["best_model_idx"]

load_best_model(best_model_idx)

for i in range(len(train_dataset)):

    x, y = train_dataset[i]

    y_hat = model(x[None, :, :], target_time)
    
    print(y_hat.shape)
    time = torch.arange(0, x.shape[0] + y.shape[0]).numpy(force=True)

    # Plot the resulting prediction
    plt.figure()
    plt.plot(time, torch.cat((x[:, 0], y[:, 0]), dim=0).numpy(force=True), label="x")        # x
    plt.plot(time, torch.cat((x[:, 1], y[:, 1]), dim=0).numpy(force=True), label="y")        # y
    plt.plot(time, torch.cat((x[:, 2], y[:, 2]), dim=0).numpy(force=True), label="z")        # z

    time2 = torch.arange(x.shape[0], x.shape[0] + y_hat.shape[1]).numpy(force=True)
    plt.plot(time2, y_hat[0, :, 0].numpy(force=True), label="x_hat")        # x
    plt.plot(time2, y_hat[0, :, 1].numpy(force=True), label="y_hat")        # y
    plt.plot(time2, y_hat[0, :, 2].numpy(force=True), label="z_hat")        # z

    plt.legend()
    plt.show()