import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from data_loaders.mocap_swipe_loader import MocapSwipeLoader

from main import model, load_best_model, train_dataset, target_time

from paper_rmse import position_rmse, position_rmse_payload, velocity_rmse, quaternion_rmse, rmse_all_state

# Load the model
target_time = 25

best_model = torch.load("output6/best_model_idx.pth.tar")
best_model_idx = best_model["best_model_idx"]

load_best_model(best_model_idx, save_dir="output6/")

test_dataset = MocapSwipeLoader(input_window=50, output_window=target_time, stride=1, split="val", device="cuda")


# Plot the position of the load
i=0

x, y = test_dataset[i]

Ts = 0.03

# Generate the u from the y vector
u = y[..., 13:17]

y_hat = model(x[None, :, :], u[None, :, :], target_time)
time = Ts * torch.arange(0, x.shape[0] + y.shape[0]).numpy(force=True)
time2 = Ts * torch.arange(x.shape[0], x.shape[0] + y.shape[0]+1).numpy(force=True)

fig = plt.figure(figsize=(4, 2), dpi=300)

# Plot the resulting prediction
x_last, y_last, z_last = x[-1, 0] * torch.ones(1), x[-1, 1] * torch.ones(1), x[-1, 2] * torch.ones(1)
plt.plot(time2, torch.cat((x_last, y_hat[0, :, 0]), dim=0).numpy(force=True), linestyle="--", color="red", label="$p_x$")        # x
plt.plot(time2, torch.cat((y_last, y_hat[0, :, 1]), dim=0).numpy(force=True), linestyle="--", color="green", label="$p_y$")        # y
plt.plot(time2, torch.cat((z_last, y_hat[0, :, 2]), dim=0).numpy(force=True), linestyle="--", color="blue", label="$p_z$")        # z

# Plot the real position
plt.plot(time, torch.cat((x[:, 0], y[:, 0]), dim=0).numpy(force=True), color=(1,0,0,0.6))        # x
plt.plot(time, torch.cat((x[:, 1], y[:, 1]), dim=0).numpy(force=True), color=(0,1,0,0.7))        # y
plt.plot(time, torch.cat((x[:, 2], y[:, 2]), dim=0).numpy(force=True), color=(0,0,1,0.5))        # z

plt.xlabel("Time (s)", fontsize=10)
plt.ylabel("Position (m)", fontsize=10)
plt.legend(loc="upper left", fontsize=10)
# adjust tik size for labels
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
fig.subplots_adjust(bottom=0.25, top=0.9, left=0.15, right=0.9)
plt.grid()

plt.savefig("position.pdf")

fig = plt.figure(figsize=(4, 2), dpi=300)
# Plot the resulting prediction
x_last, y_last, z_last = x[-1, 3] * torch.ones(1), x[-1, 4] * torch.ones(1), x[-1, 5] * torch.ones(1)
plt.plot(time2, torch.cat((x_last, y_hat[0, :, 3]), dim=0).numpy(force=True), linestyle="--", color="red", label="$v_x$")        # x
plt.plot(time2, torch.cat((y_last, y_hat[0, :, 4]), dim=0).numpy(force=True), linestyle="--", color="green", label="$v_x$")        # y
plt.plot(time2, torch.cat((z_last, y_hat[0, :, 5]), dim=0).numpy(force=True), linestyle="--", color="blue", label="$v_x$")        # z

# Plot the real position
plt.plot(time, torch.cat((x[:, 3], y[:, 3]), dim=0).numpy(force=True), color=(1,0,0,0.6))        # x
plt.plot(time, torch.cat((x[:, 4], y[:, 4]), dim=0).numpy(force=True), color=(0,1,0,0.7))        # y
plt.plot(time, torch.cat((x[:, 5], y[:, 5]), dim=0).numpy(force=True), color=(0,0,1,0.5))        # z

# adjust tik size for labels
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.xlabel("Time (s)", fontsize=10)
plt.ylabel("Velocity (m/s)", fontsize=10)
plt.legend(loc="upper left", fontsize=10)
fig.subplots_adjust(bottom=0.25, top=0.9, left=0.15, right=0.9)
plt.grid()

plt.savefig("velocity.pdf")

fig = plt.figure(figsize=(4, 2), dpi=300)
# Plot the resulting prediction
w_last, x_last, y_last, z_last = x[-1, 6] * torch.ones(1), x[-1, 7] * torch.ones(1), x[-1, 8] * torch.ones(1), x[-1, 9] * torch.ones(1)
plt.plot(time2, torch.cat((w_last, -y_hat[0, :, 6]), dim=0).numpy(force=True), linestyle="--", color="orange", label="$q_w$")        # w
plt.plot(time2, torch.cat((x_last, -y_hat[0, :, 7]), dim=0).numpy(force=True), linestyle="--", color="red", label="$q_x$")        # x
plt.plot(time2, torch.cat((y_last, -y_hat[0, :, 8]), dim=0).numpy(force=True), linestyle="--", color="green", label="$q_y$")        # y
plt.plot(time2, torch.cat((z_last, -y_hat[0, :, 9]), dim=0).numpy(force=True), linestyle="--", color="blue", label="$q_z$")        # z

# Plot the real position
plt.plot(time, torch.cat((x[:, 6], y[:, 6]), dim=0).numpy(force=True), color=(1, 220/255, 0.0,1.0))        # w
plt.plot(time, torch.cat((x[:, 7], y[:, 7]), dim=0).numpy(force=True), color=(1,0,0,0.6))        # x
plt.plot(time, torch.cat((x[:, 8], y[:, 8]), dim=0).numpy(force=True), color=(0,1,0,0.7))        # y
plt.plot(time, torch.cat((x[:, 9], y[:, 9]), dim=0).numpy(force=True), color=(0,0,1,0.5))        # z
plt.grid()

plt.xlabel("Time (s)", fontsize=10)
plt.ylabel("Quaternion", fontsize=10)
# adjust tik size for labels
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.legend(loc="lower left", fontsize=10)
fig.subplots_adjust(bottom=0.25, top=0.9, left=0.17, right=0.9)

plt.savefig("quaternion.pdf")

fig = plt.figure(figsize=(4, 2), dpi=300)
# Plot the resulting prediction of the load
x_last, y_last, z_last = x[-1, 10] * torch.ones(1), x[-1, 11] * torch.ones(1), x[-1, 12] * torch.ones(1)
plt.plot(time2, torch.cat((x_last, y_hat[0, :, 10]), dim=0).numpy(force=True), linestyle="--", color="red", label="$p^L_x$")        # x
plt.plot(time2, torch.cat((y_last, y_hat[0, :, 11]), dim=0).numpy(force=True), linestyle="--", color="green", label="$p^L_y$")        # y
plt.plot(time2, torch.cat((z_last, y_hat[0, :, 12]), dim=0).numpy(force=True), linestyle="--", color="blue", label="$p^L_z$")        # z

# Plot the real position
plt.plot(time, torch.cat((x[:, 10], y[:, 10]), dim=0).numpy(force=True), color=(1,0,0,0.6))        # x
plt.plot(time, torch.cat((x[:, 11], y[:, 11]), dim=0).numpy(force=True), color=(0,1,0,0.7))        # y
plt.plot(time, torch.cat((x[:, 12], y[:, 12]), dim=0).numpy(force=True), color=(0,0,1,0.5))        # z
plt.grid()

# Ajust x label size
plt.xlabel("Time (s)", fontsize=10)
plt.ylabel("Load Position (m)", fontsize=10)
# adjust tik size for labels
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.legend(loc="upper left", fontsize=10)
fig.subplots_adjust(bottom=0.25, top=0.9, left=0.15, right=0.9)


plt.savefig("load.pdf")

# Compute the RMSE
print(y_hat.shape)
print(y.shape)
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