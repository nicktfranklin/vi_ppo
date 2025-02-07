from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class MlpConfig:
    input_dims: int
    output_dims: int
    hidden_dims: int
    n_layers: int
    activation: str = "silu"


class Mlp(nn.Module):
    config_cls = MlpConfig

    def __init__(self, config: MlpConfig):
        super().__init__()
        self.input_projection = (
            nn.Linear(config.input_dims, config.hidden_dims)
            if (config.input_dims != config.hidden_dims)
            else nn.Identity()
        )
        self.output_projection = (
            nn.Linear(config.hidden_dims, config.output_dims)
            if (config.hidden_dims != config.output_dims)
            else nn.Identity()
        )

        self.hidden_layers = nn.ModuleList(
            [
                nn.Linear(config.hidden_dims, config.hidden_dims)
                for _ in range(config.n_layers)
            ]
        )

        if config.activation == "silu":
            self.act_fn = nn.SiLU()
        elif config.activation == "relu":
            self.act_fn = nn.ReLU()
        elif config.activation == "tanh":
            self.act_fn = nn.Tanh()
        else:
            raise ValueError("only silu is supported")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        for layer in self.hidden_layers:
            x = self.act_fn(layer(x)) + x
        x = self.output_projection(x)
        return x
