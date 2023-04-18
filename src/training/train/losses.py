import torch

def system_loss(x, y_hat, system_model):
    """
    A method that implements
    ||y_hat - f(x) ||^2

    Args:
        x (nn.Tensor): The input of the network
        y_hat (nn.Tensor): The expected output
        system_model (fn): A function that encodes a model of the system
    """
    
    return torch.mean((y_hat - system_model(x)) ** 2)
