"""
| File: discrete_model.py
| Authors: Gil Serrano, Marcelo Jacinto, Jose Gomes and Joao Pinto
| License: BSD-3-Clause. Copyright (c) 2023, Gil Serrano, Marcelo Jacinto, Jose Gomes and Joao Pinto. All rights reserved.
| Description: Definition of the discretized physics model of a quadrotor with an attached slungload.
"""
import torch
from pinn_air.utils.math_utils import skew_symmetric as skew, quaternion_to_matrix as q2R, matrix_to_quaternion as R2q

class DiscreteModel:

    def __init__(self, Ts: float, mQ: float, mL: float, l: float):
        """
        Constructor of the DiscreteMultirotor

        Args:
            Ts (float): The sampling period (in s)
            mL (float): The mass of the load (in Kg)
            mQ (float): The mass of the quadrotor (in Kg)
            l (float): The length of the cable (in m)
        """

        # Parameters of the model
        self.Ts = Ts        # Sampling period (in s)
        self.mL = mL        # Mass of the load (in Kg)
        self.mQ = mQ        # Mass of the quadrotor (in Kg)
        self.l = l          # Length of the cable (in m)

    def run(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:

        # Set the gravity acceleration
        g = -9.81
        e3 = torch.tensor([0.0, 0.0, 1.0]).to(x.device)

        # Get the state of the system by parts
        p_drone = x[0:3]           # x,y,z
        v_drone = x[3:6]           # vx,vy,vz
        R_drone = q2R(x[6:10])     # qw,qx,qy,qz
        p_load = x[10:13]          # x,y,z
        v_load = x[13:16]          # vx,vy,vz
        wl = x[16:19]              # wx,wy,wz

        # Get the input of the system by parts
        w = u[0:3]      # Target angular velocity (rad/s)
        Tq = u[3]       # Target thrust (N)

        # Compute the direction vector of the cable
        q = (p_load - p_drone) / self.l

        # Compute the tension on the load
        Tl = ((self.mL / (self.mQ + self.mL))*torch.dot(q, torch.matmul(R_drone,-Tq*e3))) + (((self.mL*self.mQ)/(self.mL+self.mQ))*self.l*torch.norm(wl,2)**2)

        # Compute the virtual inputs of the system
        u1 = ((1.0/self.mQ)*Tq*torch.matmul(R_drone, e3)) + ((self.mQ+self.mL)/self.mQ * g * e3)
        u2 = -((1.0/self.mL)*Tl*q) + (g*e3)

        x_t1 = torch.zeros((19,), device=x.device)

        # Compute the linear part of the system of the drone
        x_t1[0:3] = p_drone + self.Ts*v_drone + ((self.Ts**2)/2 * (u1 - ((self.mL/self.mQ) * u2)))
        x_t1[3:6] = v_drone + self.Ts*(u1 - ((self.mL/self.mQ) * u2))

        # Compute the linear part of the system of the load
        x_t1[10:13] = p_load + self.Ts*v_load + ((self.Ts**2)/2 * u2)
        # Normalize the cable vector and multiply by the length and update the position of the load
        x_t1[10:13] = x_t1[0:3] + (x_t1[10:13] - x_t1[0:3]) / torch.norm(x_t1[10:13] - x_t1[0:3]) * self.l
        x_t1[13:16] = v_load + self.Ts*u2

        # Compute the differential equation for the rotation of the vehicle
        x_t1[6:10] = R2q(torch.matmul(R_drone, torch.matrix_exp(skew(w)*self.Ts)))

        # Compute the angular velocity of the load
        friction = 0.025
        x_t1[16:19] = (1 - friction) * wl - (self.Ts*torch.matmul(skew(q),(Tq*torch.matmul(R_drone, e3))) / (self.mQ * self.l))

        return x_t1
    
    def run_batch(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:

        # Set the gravity acceleration
        g = -9.81
        e3 = torch.tensor([0.0, 0.0, 1.0]).to(x.device)

        # Get the state of the system by parts
        p_drone = x[:, 0:3]           # x,y,z
        v_drone = x[:, 3:6]           # vx,vy,vz
        R_drone = q2R(x[:, 6:10])     # qw,qx,qy,qz
        p_load = x[:, 10:13]          # x,y,z
        v_load = x[:, 13:16]          # vx,vy,vz
        wl = x[:, 16:19]              # wx,wy,wz

        # Get the input of the system by parts
        w = u[:, 0:3]      # Target angular velocity (rad/s)
        Tq = u[:, 3]       # Target thrust (N)

        # Compute the direction vector of the cable
        q = (p_load - p_drone) / self.l

        # Compute the tension on the load
        Tl = ( (self.mL / (self.mQ + self.mL))*torch.bmm(q.unsqueeze(1), torch.bmm(R_drone,(-Tq.unsqueeze(-1)*e3).unsqueeze(-1))).squeeze() ) + (((self.mL*self.mQ)/(self.mL+self.mQ))*self.l*torch.norm(wl,2, dim=1)**2)
        
        # Compute the virtual inputs of the system
        u1 = torch.zeros((x.shape[0], 3), device=x.device)
        u2 = torch.zeros((x.shape[0], 3), device=x.device)
        u1 = ((1.0/self.mQ)*Tq.unsqueeze(-1)*torch.matmul(R_drone, e3)) + ((self.mQ+self.mL)/self.mQ * g * e3)
        u2 = -((1.0/self.mL)*Tl.unsqueeze(-1)*q) + (g*e3)
        
        x_t1 = torch.zeros((x.shape[0], x.shape[1]), device=x.device)

        # Compute the linear part of the system of the drone
        x_t1[:, 0:3] = p_drone + self.Ts*v_drone + ((self.Ts**2)/2 * (u1 - ((self.mL/self.mQ) * u2)))
        x_t1[:, 3:6] = v_drone + self.Ts*(u1 - ((self.mL/self.mQ) * u2))

        # Compute the linear part of the system of the load
        x_t1[:, 10:13] = p_load + self.Ts*v_load + ((self.Ts**2)/2 * u2)
        # Normalize the cable vector and multiply by the length and update the position of the load
        x_t1[:, 10:13] = x_t1[:, 0:3] + (x_t1[:, 10:13] - x_t1[:, 0:3]) / torch.norm(x_t1[:, 10:13] - x_t1[:, 0:3], dim=1).unsqueeze(-1) * self.l
        x_t1[:, 13:16] = v_load + self.Ts*u2
        
        # Compute the differential equation for the rotation of the vehicle
        x_t1[:, 6:10] = R2q(torch.matmul(R_drone, torch.matrix_exp(skew(w)*self.Ts)))

        # Compute the angular velocity of the load
        friction = 0.025
        x_t1[:, 16:19] = (1 - friction) * wl - (self.Ts*torch.matmul(skew(q),(Tq.unsqueeze(-1)*torch.matmul(R_drone, e3)).unsqueeze(-1)).squeeze() / (self.mQ * self.l))

        return x_t1