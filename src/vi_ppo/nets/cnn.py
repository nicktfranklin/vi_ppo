from dataclasses import dataclass, field
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
class CnnConfig:
    input_channels: int
    channels: list[int] = field(default_factory=lambda: [32, 64, 64])
    kernel_sizes: list[int] = field(default_factory=lambda: [8, 4, 3])
    strides: list[int] = field(default_factory=lambda: [4, 2, 1])
    activation: str = "elu"
    flatten_output: bool = False


class Cnn(nn.Module):
    config_cls = CnnConfig

    def __init__(self, config: CnnConfig):
        super().__init__()
        self.config = config

        modules = []
        h_in = config.input_channels
        for h_dim, kernel_size, stride in zip(
            config.channels, config.kernel_sizes, config.strides
        ):
            modules.append(ConvBlock(h_in, h_dim, kernel_size, stride, padding=1))
            h_in = h_dim

        self.cnn = nn.Sequential(*modules)

    def calculate_output_shape(self, input_shape):
        with torch.no_grad():
            # Create a dummy input tensor
            dummy_input = torch.zeros(1, *input_shape)
            # Pass the dummy input through the CNN layers
            output = self(dummy_input)
        return output.shape

    def forward(self, x):

        # Assume NxCxHxW input or CxHxW input
        assert x.ndim == 4 or x.ndim == 3
        if x.ndim == 3:
            x = x.unsqueeze(0)
        # assert_correct_end_shape(x, self.input_shape)
        # x = maybe_expand_batch(x, self.input_shape)
        x = self.cnn(x)
        if self.config.flatten_output:
            x = x.view(x.size(0), -1)
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


@dataclass
class TransposedCnnConfig:
    embedding_dim: int = 1024
    output_channels: int = 1
    channels: list[int] = field(default_factory=lambda: [32, 64, 128, 256, 512][::-1])


class TransposedCnn(nn.Module):
    def __init__(
        self,
        config: TransposedCnnConfig,
    ):
        super().__init__()

        self.fc = nn.Linear(config.embedding_dim, 4 * config.channels[0])
        self.first_channel_size = config.channels[0]

        modules = []
        for ii in range(len(config.channels) - 1):
            modules.append(
                ConvTransposeBlock(
                    config.channels[ii],
                    config.channels[ii + 1],
                    3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                )
            )
        self.deconv = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                config.channels[-1],
                config.channels[-1],
                3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm2d(config.channels[-1]),
            nn.GELU(),
            nn.Conv2d(
                config.channels[-1],
                out_channels=config.output_channels,
                kernel_size=3,
                padding=1,
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
