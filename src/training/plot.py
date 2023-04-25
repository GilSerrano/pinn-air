import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from data_loaders.mocap_swipe_loader import MocapSwipeLoader

from main import model, load_best_model, train_dataset, target_time

# Load the model
target_time = 25

best_model = torch.load("output6/best_model_idx.pth.tar")
best_model_idx = best_model["best_model_idx"]

load_best_model(best_model_idx, save_dir="output6/")

test_dataset = MocapSwipeLoader(input_window=50, output_window=target_time, stride=1, split="val", device="cuda")



# Plot the quaternion of the vehicle
# for i in range(len(test_dataset)):

#     x, y = test_dataset[i]

#     y_hat = model(x[None, :, :], target_time)
    
#     print(y_hat.shape)
#     time = torch.arange(0, x.shape[0] + y.shape[0]).numpy(force=True)

#     # Plot the resulting prediction
#     plt.figure()
#     plt.plot(time, torch.cat((x[:, 6], y[:, 6]), dim=0).numpy(force=True), label="qw")        # qx
#     plt.plot(time, torch.cat((x[:, 7], y[:, 7]), dim=0).numpy(force=True), label="qx")        # qy
#     plt.plot(time, torch.cat((x[:, 8], y[:, 8]), dim=0).numpy(force=True), label="qy")        # qz
#     plt.plot(time, torch.cat((x[:, 9], y[:, 9]), dim=0).numpy(force=True), label="qz")        # qw

#     #plt.plot(time, torch.cat((torch.norm(x[:,6:10], p=2, dim=-1), torch.norm(y[:,6:10], p=2, dim=-1)), dim=0).numpy(force=True), label="norm")

#     time2 = torch.arange(x.shape[0], x.shape[0] + y_hat.shape[1]).numpy(force=True)
#     plt.plot(time2, y_hat[0, :, 6].numpy(force=True), label="qw_hat")        # qx
#     plt.plot(time2, y_hat[0, :, 7].numpy(force=True), label="qx_hat")        # qy
#     plt.plot(time2, y_hat[0, :, 8].numpy(force=True), label="qy_hat")        # qz
#     plt.plot(time2, y_hat[0, :, 9].numpy(force=True), label="qz_hat")        # qw

#     plt.legend()
#     plt.show()

# Plot the velocity of the vehicle
# for i in range(len(test_dataset)):

#     x, y = test_dataset[i]

#     y_hat = model(x[None, :, :], target_time)
    
#     print(y_hat.shape)
#     time = torch.arange(0, x.shape[0] + y.shape[0]).numpy(force=True)

#     # Plot the resulting prediction
#     plt.figure()
#     plt.plot(time, torch.cat((x[:, 3], y[:, 3]), dim=0).numpy(force=True), label="x")        # vx
#     plt.plot(time, torch.cat((x[:, 4], y[:, 4]), dim=0).numpy(force=True), label="y")        # vy
#     plt.plot(time, torch.cat((x[:, 5], y[:, 5]), dim=0).numpy(force=True), label="z")        # vz

#     time2 = torch.arange(x.shape[0], x.shape[0] + y_hat.shape[1]).numpy(force=True)
#     plt.plot(time2, y_hat[0, :, 3].numpy(force=True), label="x_hat")        # vx
#     plt.plot(time2, y_hat[0, :, 4].numpy(force=True), label="y_hat")        # vy
#     plt.plot(time2, y_hat[0, :, 5].numpy(force=True), label="z_hat")        # vz

#     plt.legend()
#     plt.show()

# Plot the position of the vehicle
# for i in range(len(test_dataset)):

#     x, y = test_dataset[i]

#     y_hat = model(x[None, :, :], target_time)
    
#     print(y_hat.shape)
#     time = torch.arange(0, x.shape[0] + y.shape[0]).numpy(force=True)

#     # Plot the resulting prediction
#     plt.figure()
#     plt.plot(time, torch.cat((x[:, 0], y[:, 0]), dim=0).numpy(force=True), label="x")        # x
#     plt.plot(time, torch.cat((x[:, 1], y[:, 1]), dim=0).numpy(force=True), label="y")        # y
#     plt.plot(time, torch.cat((x[:, 2], y[:, 2]), dim=0).numpy(force=True), label="z")        # z

#     time2 = torch.arange(x.shape[0], x.shape[0] + y_hat.shape[1]).numpy(force=True)
#     plt.plot(time2, y_hat[0, :, 0].numpy(force=True), label="x_hat")        # x
#     plt.plot(time2, y_hat[0, :, 1].numpy(force=True), label="y_hat")        # y
#     plt.plot(time2, y_hat[0, :, 2].numpy(force=True), label="z_hat")        # z

#     plt.legend()
#     plt.show()

# Plot the position of the load
for i in range(len(test_dataset)):

    x, y = test_dataset[i]

    # Generate the u from the y vector
    u = y[..., 13:17]

    y_hat = model(x[None, :, :], u[None, :, :], target_time)
    
    print(y_hat.shape)
    time = torch.arange(0, x.shape[0] + y.shape[0]).numpy(force=True)

    time2 = torch.arange(x.shape[0], x.shape[0] + y_hat.shape[1]).numpy(force=True)
    plt.plot(time, torch.cat((x[:, 10], y_hat[0, :, 10]), dim=0).numpy(force=True), label="x_hat")        # x
    plt.plot(time, torch.cat((x[:, 11], y_hat[0, :, 11]), dim=0).numpy(force=True), label="y_hat")        # y
    plt.plot(time, torch.cat((x[:, 12], y_hat[0, :, 12]), dim=0).numpy(force=True), label="z_hat")        # z

    # Plot the resulting prediction
    plt.plot(time, torch.cat((x[:, 10], y[:, 10]), dim=0).numpy(force=True), label="x")        # x
    plt.plot(time, torch.cat((x[:, 11], y[:, 11]), dim=0).numpy(force=True), label="y")        # y
    plt.plot(time, torch.cat((x[:, 12], y[:, 12]), dim=0).numpy(force=True), label="z")        # z

    plt.legend()
    plt.show()

# Plot the input of the vehicle
# for i in range(len(test_dataset)):

#     x, y = test_dataset[i]

#     y_hat = model(x[None, :, :], target_time)
    
#     print(y_hat.shape)
#     time = torch.arange(0, x.shape[0] + y.shape[0]).numpy(force=True)

#     # Plot the resulting prediction
#     plt.figure()
#     plt.plot(time, torch.cat((x[:, 13], y[:, 13]), dim=0).numpy(force=True), label="x")        # qx
#     plt.plot(time, torch.cat((x[:, 14], y[:, 14]), dim=0).numpy(force=True), label="y")        # qy
#     plt.plot(time, torch.cat((x[:, 15], y[:, 15]), dim=0).numpy(force=True), label="z")        # qz
#     plt.plot(time, torch.cat((x[:, 16], y[:, 16]), dim=0).numpy(force=True), label="z")        # qw

#     time2 = torch.arange(x.shape[0], x.shape[0] + y_hat.shape[1]).numpy(force=True)
#     plt.plot(time2, y_hat[0, :, 13].numpy(force=True), label="x_hat")        # qx
#     plt.plot(time2, y_hat[0, :, 14].numpy(force=True), label="y_hat")        # qy
#     plt.plot(time2, y_hat[0, :, 15].numpy(force=True), label="z_hat")        # qz
#     plt.plot(time2, y_hat[0, :, 16].numpy(force=True), label="z_hat")        # qw

#     plt.legend()
#     plt.show()