#!/usr/bin/env python3
import torch
from math_utils import quaternion_multiply, quaternion_invert

def position_error(y_hat, y, target_time):
    
    # Compute the exponential decay
    exp_decay = 10 * torch.exp(-0.1*torch.arange(0, target_time, 1)).to("cuda")
    
    # Compute the discounted position error
    return torch.sum((torch.norm(y_hat[:, :, 0:3] - y[:, :, 0:3], dim=2) ** 2) * exp_decay)

def position_error_payload(y_hat, y, target_time):
    
    # Compute the exponential decay
    exp_decay = 10 * torch.exp(-0.1*torch.arange(0, target_time, 1)).to("cuda")
    
    # Compute the discounted position error
    return torch.sum((torch.norm(y_hat[:, :, 10:13] - y[:, :, 10:13], dim=2) ** 2) * exp_decay)

def velocity_error(y_hat, y, target_time):

    # Compute the exponential decay
    exp_decay = 10 * torch.exp(-0.1*torch.arange(0, target_time, 1)).to("cuda")

    # Compute the discounted velocity error
    return torch.sum((torch.norm(y_hat[:, :, 3:6] - y[:, :, 3:6], dim=2) ** 2) * exp_decay)

def quaternion_error(y_hat, y, target_time):

    # Compute the exponential decay
    exp_decay = 10 * torch.exp(-0.1*torch.arange(0, target_time, 1)).to("cuda")

    # Compute the discounted quaternion error
    identity_quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0]).to("cuda")

    return torch.sum((torch.norm(quaternion_multiply(y[..., 6:10], quaternion_invert(y_hat[..., 6:10])) - identity_quaternion, dim=2) ** 2) * exp_decay)

def quaternion_norm(y_hat):

    quat_norm = y_hat[..., 6:10].norm(p=2, dim=-1)
    one = torch.ones_like(quat_norm)
    
    return torch.sum((quat_norm - one) ** 2)

def physics_error(y_hat, x, y, target_time, physics_model):

    # Create an empty tensor to store the predictions
    physics_pred = torch.zeros(x.shape[0], target_time, 10).to("cuda")

    # Compute the first prediction of the physical model
    physics_pred[:, 0, :] = physics_model.run(x=x[:, -1, 0:10].unsqueeze(1), u=x[:,-1, 13:17].unsqueeze(1)).squeeze(1)

    # Compute the rest of the predictions of the physical model
    physics_pred[:, 1:target_time, :] = physics_model.run(x=y[:, 0:target_time-1, 0:10], u=y[:, 0:target_time-1, 13:17])

    # Compute the error between the predicted and the physical model
    pos_error = position_error(y_hat, physics_pred, target_time)
    vel_error = velocity_error(y_hat, physics_pred, target_time)
    quat_error = quaternion_error(y_hat, physics_pred, target_time)

    return pos_error + vel_error + quat_error

def continuity_last_input_first_output(y_hat, x):
    return torch.sum(torch.norm(y_hat[:, 0, :] - x[:, -1, 0:13], dim=1) **2, dim=0)

def output_continuity(y_hat):

    total_loss = 0
    for t in range(y_hat.shape[1] - 1):
        total_loss += torch.sum(torch.norm(y_hat[:, t, :] - y_hat[:, t+1, :], dim=1) ** 2, dim=0)

    return total_loss