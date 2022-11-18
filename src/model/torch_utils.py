"""Define utils function for torch models"""
from typing import Any

from torch import nn


def tanh_initializer(layer: Any) -> None:
    """
    Initialize weights of linear layer with tanh activation

    Parameters
    ----------
    layer: Any
        a layer from nn.modules
    """
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain("tanh"))


def relu_initializer(layer: Any) -> None:
    """
    Initialize weights of linear layer with relu activation

    Parameters
    ----------
    layer: Any
        a layer from nn.modules
    """
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")


def leaky_relu_initializer(layer: Any) -> None:
    """
    Initialize weights of linear layer with leaky_relu activation

    Parameters
    ----------
    layer: Any
        a layer from nn.modules
    """
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_uniform_(layer.weight, nonlinearity="leaky_relu")


def get_last_out_features(network: nn.Sequential) -> int:
    """Get out_features of last linear layer of the network"""
    for layer in network[::-1]:
        if isinstance(layer, nn.Linear):
            return layer.out_features
    return 0
