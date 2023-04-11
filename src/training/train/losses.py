import torch

def mse_loss(x, y):
    """Mean square error loss

    Args:
        x (nn.Tensor): The output of the network
        y (nn.Tensor): The expected output
    """
    torch.mean((x - y) ** 2)


def system_loss(x, y, system_model):
    