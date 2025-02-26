from dataclasses import dataclass, field
from typing import Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from vi_ppo.nets.quantizers import FSQ, LFQ, VectorQuantizeEMA


@dataclass
class VQVAEArgs:
    z_dim: int = 8
    z_layers: int = 8
    embed_dim: int = 256
    quantizer: str = "fsq"
    n_embed: int = 1024
    lfq_dim: int = None
    entropy_loss_weight: float = None
    codebook_loss_weight: float = None
    levels: list = field(default_factory=lambda: [8, 6, 5])


class VQVAE(nn.Module):
    config_cls = VQVAEArgs

    def __init__(self, encoder: nn.Module, decoder: nn.Module, config: VQVAEArgs):
        """
        Initializes the VQVAE module.

        Args:
            args: A namespace containing the arguments for the VQVAE.
        """
        super().__init__()
        self.config = config

        if config.quantizer == "ema" or config.quantizer == "origin":
            self.quantize_t = VectorQuantizeEMA(
                config, config.embed_dim, config.n_embed
            )
        elif config.quantizer == "lfq":
            self.quantize_t = LFQ(
                codebook_size=2**config.lfq_dim,
                dim=config.lfq_dim,
                entropy_loss_weight=config.entropy_loss_weight,
                commitment_loss_weight=config.codebook_loss_weight,
            )
        elif config.quantizer == "fsq":
            self.quantize_t = FSQ(levels=config.levels)
        else:
            print("quantizer error!")
            exit()

        self.enc = encoder
        self.dec = decoder

    def forward(self, input: torch.Tensor, return_id: bool = True) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
    ]:
        """
        Forward pass through the VQVAE.

        Args:
            input (torch.Tensor): Input tensor.
            return_id (bool): Whether to return the quantized indices.

        Returns:
            Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]: Decoded tensor, quantization loss, and optionally the quantized indices.
        """
        quant_t, diff, id_t = self.encode(input)
        dec = self.dec(quant_t.view(-1, self.args.z_layers * self.args.z_dim))
        if return_id:
            return dec, diff, id_t
        return dec, diff

    def loss(
        self, input: torch.Tensor, target: torch.Tensor | None = None
    ) -> torch.Tensor:
        reconstruction, codebook_loss = self(input, return_id=False)
        target = input if target is None else target
        loss = F.mse_loss(reconstruction, target)
        return loss + codebook_loss

    def encode(
        self, input: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encodes the input tensor.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Quantized tensor, quantization loss, and quantized indices.
        """
        logits = self.enc(input).view(-1, self.config.z_layers, self.config.z_dim)
        if self.config.quantizer == "ema" or self.config.quantizer == "origin":
            quant_t, diff_t, id_t = self.quantize_t(logits)
            diff_t = diff_t.unsqueeze(0)
        elif self.config.quantizer == "fsq":
            quant_t, id_t = self.quantize_t(logits)
            diff_t = torch.tensor(0.0).to(quant_t.device).float()
        elif self.config.quantizer == "lfq":
            quant_t, id_t, diff_t = self.quantize_t(logits)
        return quant_t, diff_t, id_t

    def decode(self, code: torch.Tensor) -> torch.Tensor:
        """
        Decodes the quantized tensor.

        Args:
            code (torch.Tensor): Quantized tensor.

        Returns:
            torch.Tensor: Decoded tensor.
        """
        return self.dec(code)

    def decode_code(self, code_t: torch.Tensor) -> torch.Tensor:
        """
        Decodes the quantized indices.

        Args:
            code_t (torch.Tensor): Quantized indices.

        Returns:
            torch.Tensor: Decoded tensor.
        """
        quant_t = self.quantize_t.embed_code(code_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        dec = self.dec(quant_t)
        return dec
