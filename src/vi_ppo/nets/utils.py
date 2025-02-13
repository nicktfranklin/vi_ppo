from typing import Tuple

import torch


def assert_correct_end_shape(
    x: torch.tensor, shape: Tuple[int, int] | Tuple[int, int, int]
) -> bool:
    if len(shape) == 2:
        assert (
            x.shape[-2:] == shape
        ), f"Tensor Shape {x.shape} does not match target {shape}"
    elif len(shape) == 3:
        assert (
            x.shape[-3:] == shape
        ), f"Tensor Shape {x.shape} does not match target {shape}"
    else:
        raise Exception(f"Shape {shape} is unsupported")


def maybe_expand_batch(
    x: torch.tensor, target_shape: Tuple[int, int] | Tuple[int, int, int]
) -> torch.tensor:
    # expand if unbatched
    if check_shape_match(x, target_shape):
        return x[None, ...]
    return x


def check_shape_match(x: torch.FloatTensor, shape: Tuple[int]):
    return (x.ndim == len(shape)) and (
        x.view(-1).shape[0] == torch.prod(torch.tensor(shape))
    )


def get_activation(activation: str) -> torch.nn.Module:
    """
    Returns a PyTorch activation layer based on the input string.

    Supported activations: 'relu', 'elu', 'silu', 'leaky_relu', 'sigmoid',
                           'tanh', 'gelu', 'selu', 'softplus', 'prelu'

    Args:
        activation (str): The name of the activation function.

    Returns:
        torch.nn.Module: The corresponding activation layer.

    Raises:
        ValueError: If the activation name is not supported.
    """
    activation = activation.lower()
    activation_mapping = {
        "relu": torch.nn.ReLU,
        "elu": torch.nn.ELU,
        "silu": torch.nn.SiLU,
        "leaky_relu": torch.nn.LeakyReLU,
        "sigmoid": torch.nn.Sigmoid,
        "tanh": torch.nn.Tanh,
        "gelu": torch.nn.GELU,
        "selu": torch.nn.SELU,
        "softplus": torch.nn.Softplus,
        "prelu": torch.nn.PReLU,
    }
    if activation not in activation_mapping:
        raise ValueError(
            f"Unsupported activation: '{activation}'. "
            f"Supported activations are: {list(activation_mapping.keys())}"
        )
    return activation_mapping[activation]()
