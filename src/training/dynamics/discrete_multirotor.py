import torch


class DiscreteMultirotor:
    """
    Class that implements the dynamics of a discrete multirotor system
    with rotations
    """

    def __init__(self, Ts: float, mass: float):
        """
        Constructor of the DiscreteMultirotor

        Args:
            Ts (float): The sampling period (in s)
            mass (float): The mass of the vehicle (in Kg)
        """
        self.Ts = Ts
        self.m = mass
        self.g = torch.Tensor([0.0, 0.0, -9.81]).unsqueeze(-1)  # [3x1] vector

        # Model for the linear dynamics of the vehicle
        # Note: since we have X \in R^6, i.e. X=[x,y,z,v_x,v_y,v_z]
        # so we need to use the kronecker product to expand this
        A_star = torch.tensor([[1.0,  Ts],
                               [0.0, 1.0]])
        
        B_star = torch.tensor([(Ts ** 2.0) / 2.0, Ts]).unsqueeze(-1)

        # ---------------------------------
        # X[k+1] = Ax[k] + Bu[k] 
        # (where at this abstraction level, 
        # u[k] = [acc_x, acc_y, acc_z])
        # ---------------------------------
        
        # The "A" matrix of the dynamic model for the linear dynamics
        self.A = torch.kron(A_star, torch.eye(3, 3))

        # The "B" matrix of the dynamic model for the linear dynamics        
        self.B = torch.kron(B_star, torch.eye(3,3))

        # Compute the u[k] of our model from Fref[k] (the total thrust in N at time step k)
        self.
    

    def run(self, x: torch.Tensor, u: torch.Tensor):
        """
        Equation that describes the discrete state-space equations of a multirotor
        operating in 3D space.

        Args:
            x (torch.Tensor): The state of the system at the previous time-step, i.e. x[k-1]
            u (torch.Tensor): The input of the system at the current time-step, i.e. u[k]

        Returns:
            torch.Tensor: The state of the system at the current time-step, i.e. x[k]
        """
        
        # -----------------------------------------------------
        # Compute the input of the system for the linear motion
        # -----------------------------------------------------
        u_k = self.g + ((1.0 / self.m) * )

        # -----------------------------------------------------
        # Compute the rotational motion update 
        # -----------------------------------------------------

    


if __name__ == "__main__":

    multi = DiscreteMultirotor(3, mass=1.5)
    