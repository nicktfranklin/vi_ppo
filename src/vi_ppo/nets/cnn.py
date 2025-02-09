from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from vi_ppo.nets.utils import assert_correct_end_shape, maybe_expand_batch


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.act = nn.ELU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.act(x)
        return x


@dataclass
class CnnBlockConfig:
    input_dims: int
    output_dims: int
    hidden_dims: list[int] = [32, 64, 64]
    kernel_sizes: list[int] = [8, 4, 3]
    strides: list[int] = [4, 2, 1]
    activation: str = "elu"


class CnnBlock(nn.Module):
    def __init__(self, config: CnnBlockConfig):
        super().__init__()

        if kernel_sizes is None:
            kernel_sizes = [8, 4, 3]
        if strides is None:
            strides = [4, 2, 1]

        modules = []
        h_in = in_channels
        for h_dim, kernel_size, stride in zip(channels, kernel_sizes, strides):
            modules.append(ConvBlock(h_in, h_dim, kernel_size, stride, padding=1))
            h_in = h_dim

        self.cnn = nn.Sequential(*modules)
        self.embedding_dim = embedding_dim

    def calculate_output_shape(self, input_shape):
        with torch.no_grad():
            # Create a dummy input tensor
            dummy_input = torch.zeros(1, *input_shape)
            # Pass the dummy input through the CNN layers
            output = self.cnn(dummy_input)
        return output_shape.shape

    def forward(self, x):
        # Assume NxCxHxW input or CxHxW input
        assert x.ndim == 4 or x.ndim == 3
        assert_correct_end_shape(x, self.input_shape)
        x = maybe_expand_batch(x, self.input_shape)
        x = self.cnn(x)
        x = self.mlp(torch.flatten(x, start_dim=1))
        return x


class ConvTransposeBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=1,
        output_padding=0,
    ):
        super().__init__()
        self.conv_t = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding
        )
        self.act = nn.ELU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_t(x)
        x = self.act(x)
        return x


class CnnDecoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 1024,
        output_channels: int = 1,
        channels: Optional[List[int]] = None,
        output_shape=None,  # not used
    ):
        super().__init__()

        if channels is None:
            channels = [32, 64, 128, 256, 512][::-1]

        self.fc = nn.Linear(embedding_dim, 4 * channels[0])
        self.first_channel_size = channels[0]

        modules = []
        for ii in range(len(channels) - 1):
            modules.append(
                ConvTransposeBlock(
                    channels[ii],
                    channels[ii + 1],
                    3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                )
            )
        self.deconv = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                channels[-1],
                channels[-1],
                3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm2d(channels[-1]),
            nn.GELU(),
            nn.Conv2d(
                channels[-1], out_channels=output_channels, kernel_size=3, padding=1
            ),
            nn.Sigmoid(),
        )

    def forward(self, z):
        hidden = self.fc(z)
        hidden = hidden.view(-1, self.first_channel_size, 2, 2)
        hidden = self.deconv(hidden)
        observation = self.final_layer(hidden)
        if observation.shape[0] == 1:
            return observation.squeeze(0)
        return observation
