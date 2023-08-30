#!/usr/bin/env python3
"""
| File: loss.py
| Authors: Gil Serrano, Marcelo Jacinto, Jose Gomes and Joao Pinto
| License: BSD-3-Clause. Copyright (c) 2023, Gil Serrano, Marcelo Jacinto, Jose Gomes and Joao Pinto. All rights reserved.
| Description: Definition of the loss functions used in the project.
"""
import torch
from pinn_air.utils.math_utils import quaternion_multiply, quaternion_invert

# [x,y,z | vx,vy,vz | qw,qx,qy,qz | pxl, pyl, pzl]

def position_error(y_hat, y, decay_coefficient):

    # Get the total time from the y_hat
    target_time = y_hat.shape[1]
    
    # Compute the exponential decay
    exp_decay = torch.exp(-decay_coefficient*torch.arange(0, target_time, 1)).to("cuda")
    
    # Compute the discounted position error
    return torch.sum((torch.norm(y_hat[:, :, 0:3] - y[:, :, 0:3], dim=2) ** 2) * exp_decay)

def position_error_payload(y_hat, y, decay_coefficient):

    # Get the total time from the y_hat
    target_time = y_hat.shape[1]
    
    # Compute the exponential decay
    exp_decay = torch.exp(-decay_coefficient*torch.arange(0, target_time, 1)).to("cuda")
    
    # Compute the discounted position error
    return torch.sum((torch.norm(y_hat[:, :, 10:13] - y[:, :, 10:13], dim=2) ** 2) * exp_decay)

def velocity_error(y_hat, y, decay_coefficient):

    # Get the total time from the y_hat
    target_time = y_hat.shape[1]

    # Compute the exponential decay
    exp_decay = torch.exp(-decay_coefficient*torch.arange(0, target_time, 1)).to("cuda")

    # Compute the discounted velocity error
    return torch.sum((torch.norm(y_hat[:, :, 3:6] - y[:, :, 3:6], dim=2) ** 2) * exp_decay)

def quaternion_error(y_hat, y, decay_coefficient):

    # Get the total time from the y_hat
    target_time = y_hat.shape[1]

    # Compute the exponential decay
    exp_decay = torch.exp(-decay_coefficient*torch.arange(0, target_time, 1)).to("cuda")

    # Compute the discounted quaternion error
    identity_quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0]).to("cuda")

    return torch.sum((torch.norm(quaternion_multiply(y[..., 6:10], quaternion_invert(y_hat[..., 6:10])) - identity_quaternion, dim=2) ** 2) * exp_decay)

def quaternion_norm(y_hat):

    quat_norm = y_hat[..., 6:10].norm(p=2, dim=-1)
    one = torch.ones_like(quat_norm)
    
    return torch.sum((quat_norm - one) ** 2)

def physics_error(y_hat, y_physics, decay_coefficient):

    # Compute the error between the predicted and the physical model
    pos_error = position_error(y_hat, y_physics, decay_coefficient)
    vel_error = velocity_error(y_hat, y_physics, decay_coefficient)
    quat_error = quaternion_error(y_hat, y_physics, decay_coefficient)
    payload_error = position_error_payload(y_hat, y_physics, decay_coefficient)

    return pos_error + vel_error + quat_error + payload_error

def iterative_physics_error(y_hat, y, x, physics_model, decay_coefficient):

    batch_size = y_hat.shape[0]
    output_window = y_hat.shape[1]
    system_state_dim = 19

    y_physics = torch.zeros(batch_size, output_window, system_state_dim).to("cuda")
    
    for k in range(output_window): # output window size
        if k == 0:
            y_physics[:, k, :] = physics_model.run_batch(x[:, -1, 0:19], x[:, -1, 19:23])
        else:
            y_hat_aug = torch.cat((y_hat[:, k-1, 0:13], y[:, k-1, 13:19]), dim=1)
            y_physics[:, k, :] = physics_model.run_batch(y_hat_aug, y[:, k-1, 19:23])

    # 0, 1,2     3, 4,5      6,7,8,9
    # [x,y,z] [vx, vy,vz] [qw, qx,qy,qz] [pxl, pyl, pzl]  
    pos_error = y_hat[..., 0:3] - y_physics[...,0:3] + y_hat[..., 13:16]        # adding the slack variables to the error
    vel_error = y_hat[..., 3:6] - y_physics[...,3:6] + y_hat[..., 16:19]        # adding the slack variables to the error
    # Note that the quaternion error cannot be computed by a simple subtraction
    # We must use the quaternion multiplication and inversion
    identity_quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0]).to("cuda")
    quat_error = quaternion_multiply(y_hat[..., 6:10], quaternion_invert(y_physics[..., 6:10])) - identity_quaternion + y_hat[..., 19:23] # adding the slack variables to the error
    payload_error = y_hat[...,10:13] - y_physics[...,10:13] + y_hat[..., 23:26]  # adding the slack variables to the error

    # Concatenate the errors in the last dimension
    error = torch.cat((pos_error, vel_error, quat_error, payload_error), dim=-1)
    
    # Compute the exponential decay
    exp_decay = torch.exp(-decay_coefficient*torch.arange(0, output_window, 1)).to("cuda")

    return torch.sum((torch.norm(error, dim=2) ** 2)  * exp_decay)

def iterative_physics_error_no_slack(y_hat, y, x, physics_model, decay_coefficient):

    batch_size = y_hat.shape[0]
    output_window = y_hat.shape[1]
    system_state_dim = 19

    y_physics = torch.zeros(batch_size, output_window, system_state_dim).to("cuda")
    
    for k in range(output_window): # output window size
        if k == 0:
            y_physics[:, k, :] = physics_model.run_batch(x[:, -1, 0:19], x[:, -1, 19:23])
        else:
            y_hat_aug = torch.cat((y_hat[:, k-1, 0:13], y[:, k-1, 13:19]), dim=1)
            y_physics[:, k, :] = physics_model.run_batch(y_hat_aug, y[:, k-1, 19:23])

    # 0, 1,2     3, 4,5      6,7,8,9
    # [x,y,z] [vx, vy,vz] [qw, qx,qy,qz] [pxl, pyl, pzl]  
    pos_error = y_hat[..., 0:3] - y_physics[...,0:3]
    vel_error = y_hat[..., 3:6] - y_physics[...,3:6]
    # Note that the quaternion error cannot be computed by a simple subtraction
    # We must use the quaternion multiplication and inversion
    identity_quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0]).to("cuda")
    quat_error = quaternion_multiply(y_hat[..., 6:10], quaternion_invert(y_physics[..., 6:10])) - identity_quaternion
    payload_error = y_hat[...,10:13] - y_physics[...,10:13]

    # Concatenate the errors in the last dimension
    error = torch.cat((pos_error, vel_error, quat_error, payload_error), dim=-1)
    
    # Compute the exponential decay
    exp_decay = torch.exp(-decay_coefficient*torch.arange(0, output_window, 1)).to("cuda")

    return torch.sum((torch.norm(error, dim=2) ** 2)  * exp_decay)

def iterative_physics_error_without_payload(y_hat, y, x, physics_model, decay_coefficient):

    batch_size = y_hat.shape[0]
    output_window = y_hat.shape[1]
    system_state_dim = 10

    y_physics = torch.zeros(batch_size, output_window, system_state_dim).to("cuda")
    
    for k in range(output_window): # output window size
        if k == 0:
            y_physics[:, k, :] = physics_model.run_batch(x[:, -1, 0:10], x[:, -1, 10:14])
        else:
            y_physics[:, k, :] = physics_model.run_batch(y_hat[:,k-1,:], y[:, k-1, 10:14])
    
    # Compute the exponential decay
    exp_decay = torch.exp(-decay_coefficient*torch.arange(0, output_window, 1)).to("cuda")

    return torch.sum((torch.norm(y_hat - y_physics[:,:, 0:10], dim=2) ** 2)  * exp_decay)

def physics_error_without_payload(y_hat, y_physics, decay_coefficient):

    # Compute the error between the predicted and the physical model
    pos_error = position_error(y_hat, y_physics, decay_coefficient)
    vel_error = velocity_error(y_hat, y_physics, decay_coefficient)
    quat_error = quaternion_error(y_hat, y_physics, decay_coefficient)

    return pos_error + vel_error + quat_error

def continuity_last_input_first_output(y_hat, x):
    return torch.sum(torch.norm(y_hat[:, 0, 0:13] - x[:, -1, 0:13], dim=1) **2, dim=0)

def continuity_last_input_first_output_without_payload(y_hat, x):
    return torch.sum(torch.norm(y_hat[:, 0, 0:10] - x[:, -1, 0:10], dim=1) **2, dim=0)

def output_continuity(y_hat):

    total_loss = 0
    for t in range(y_hat.shape[1] - 1):
        total_loss += torch.sum(torch.norm(y_hat[:, t, 0:13] - y_hat[:, t+1, 0:13], dim=1) ** 2, dim=0)

    return total_loss

def output_continuity_without_payload(y_hat):

    total_loss = 0
    for t in range(y_hat.shape[1] - 1):
        total_loss += torch.sum(torch.norm(y_hat[:, t, 0:10] - y_hat[:, t+1, 0:10], dim=1) ** 2, dim=0)

    return total_loss

def slack_weight_reduction(y_hat, decay_coefficient):
    # Tries to make the slack weight as small as possible
    #return torch.norm(y_hat[:, :, 13:26], p=2, dim=2).sum()

    output_window = y_hat.shape[1]

    # Compute the exponential decay
    exp_decay = torch.exp(-decay_coefficient*torch.arange(0, output_window, 1)).to("cuda")

    return torch.sum((torch.norm(y_hat[:, :, 13:26], dim=2))  * exp_decay)

def slack_weight_reduction_without_payload(y_hat, decay_coefficient):
    # Tries to make the slack weight as small as possible
    #return torch.norm(y_hat[:, :, 10:20], p=2, dim=2).sum()

    output_window = y_hat.shape[1]

    # Compute the exponential decay
    exp_decay = torch.exp(-decay_coefficient*torch.arange(0, output_window, 1)).to("cuda")

    return torch.sum((torch.norm(y_hat[:, :, 10:20], dim=2))  * exp_decay)