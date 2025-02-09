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
