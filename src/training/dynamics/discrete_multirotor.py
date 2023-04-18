import torch
from .utils import quaternion_to_matrix, matrix_to_quaternion, skew_symmetric

class DiscreteMultirotor:
    """
    Class that implements the dynamics of a discrete multirotor model
    """

    def __init__(self, Ts: float, mass: float, device="cpu"):
        """
        Constructor of the DiscreteMultirotor

        Args:
            Ts (float): The sampling period (in s)
            mass (float): The mass of the vehicle (in Kg)
            device (str): The device where the computations will be run
        """

        self.device = device

        self.Ts = Ts
        self.m = mass
        self.g = torch.Tensor([0.0, 0.0, -9.81], device=device)  # [3x1] vector

        # Model for the linear dynamics of the vehicle
        # Note: since we have X \in R^6, i.e. X=[x,y,z,v_x,v_y,v_z]
        # so we need to use the kronecker product to expand this
        A_star = torch.tensor([[1.0,  Ts],
                               [0.0, 1.0]], device=device)
        
        B_star = torch.tensor([(Ts ** 2.0) / 2.0, Ts], device=device).unsqueeze(-1)

        # ---------------------------------
        # X[k+1] = Ax[k] + Bu[k] 
        # (where at this abstraction level, 
        # u[k] = [acc_x, acc_y, acc_z])
        # ---------------------------------
        
        # The "A" matrix of the dynamic model for the linear dynamics
        self.A = torch.kron(A_star, torch.eye(3, 3).to(device))

        # The "B" matrix of the dynamic model for the linear dynamics        
        self.B = torch.kron(B_star, torch.eye(3,3).to(device))

    def run(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Equation that describes the discrete state-space equations of a multirotor
        operating in 3D space. We expect the inputs to be in the form
             x=(batch, time_series_length, len(state))
             u=(batch, time_series_length, len(input))

        Args:
            x (torch.Tensor): The state of the system at the previous time-step, i.e. x[k-1]
                x[k-1] = [x,y,z | v_x, v_y,v_z | q_x, q_y, q_z, q_w]
            u (torch.Tensor): The input of the system at the current time-step, i.e. u[k]
                u[k] = [w_x, w_y, w_z | T]  
        Returns:
            torch.Tensor: The state of the system at the current time-step, i.e. x[k]
        """
        
        # -----------------------------------------------------
        # Compute the input of the system for the linear motion
        # -----------------------------------------------------
        
        # 1) Convert the quaternion to a rotation matrix (this can be simplified) 
        # Note that we receive the quaternion in the [qx, qx, qy, qw] convention
        rot = quaternion_to_matrix(x[..., 6:10])

        # Generate the thrust vector (in order to handle batches and sequences)
        u_shape = list(u.shape[:-1]) + [3]                  # (batch_size, sequence_size, 3)
        total_thrust = torch.zeros(size=u_shape)
        total_thrust[..., 2] = u[..., 3]                    # Vectors of the type (batch_size, time_sequence, [0.0, 0.0, total_thrust])

        # 2) Compute the reference linear acceleration that the linear system is supposed to track
        u_k = (1.0 / self.m) * torch.matmul(rot, total_thrust[..., None]).squeeze(dim=-1) + self.g        # (batch_size, time_sequence, 3)

        # dimensions of self.A =(6=i,6=j) | X=(batch_size=b, time_sequence=t, 6=j) -> (b, t, i)
        # dimensions of self.B =(6=i, 3=j) | U=(batch_size=b, time_sequence=t, 3=j)

        # 3) Compute the reference state x[k+1] = A x[k] + B u[k]
        new_x = torch.einsum('ij,btj->bti', self.A, x[..., 0:6]) + torch.einsum('ij,btj->bti', self.B, u_k)

        # -----------------------------------------------------
        # Compute the rotational motion update 
        # -----------------------------------------------------

        # 1) Compute the skew-symmetric matrix
        skew = skew_symmetric(u[..., 0:3])

        # 2) Compute the new rotation matrix R[k+1]
        new_R = rot * torch.matrix_exp(self.Ts * skew)

        # 3) Generate the new quaternion from the rotation matrix
        # Note that we produce a quaternion in the [qx, qy, qz, qw] convention
        new_q = matrix_to_quaternion(new_R)

        # -----------------------------------------------------
        # Concatenate both into a new state vector
        # -----------------------------------------------------
        return torch.cat([new_x, new_q], -1)


if __name__ == "__main__":

    multi = DiscreteMultirotor(3, mass=1.5)
    