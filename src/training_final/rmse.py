import torch
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