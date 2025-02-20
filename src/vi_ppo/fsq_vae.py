from dataclasses import dataclass, field
from typing import Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from vi_ppo.nets.quantizers import FSQ, LFQ, VectorQuantizeEMA


@dataclass
class VQVAEArgs:
    embed_dim: int = 256
    quantizer: str = "fsq"
    n_embed: int = 1024
    lfq_dim: int = None
    entropy_loss_weight: float = None
    codebook_loss_weight: float = None
    levels: list = field(default_factory=lambda: [8, 5, 5, 5])


class VQVAE(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, args: VQVAEArgs):
        """
        Initializes the VQVAE module.

        Args:
            args: A namespace containing the arguments for the VQVAE.
        """
        super().__init__()
        self.args = args

        if args.quantizer == "ema" or args.quantizer == "origin":
            self.quantize_t = VectorQuantizeEMA(args, args.embed_dim, args.n_embed)
        elif args.quantizer == "lfq":
            self.quantize_t = LFQ(
                codebook_size=2**args.lfq_dim,
                dim=args.lfq_dim,
                entropy_loss_weight=args.entropy_loss_weight,
                commitment_loss_weight=args.codebook_loss_weight,
            )
        elif args.quantizer == "fsq":
            self.quantize_t = FSQ(levels=args.levels)
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
        dec = self.dec(quant_t)
        if return_id:
            return dec, diff, id_t
        return dec, diff

    def loss(self, input: torch.Tensor, target: torch.Tensor):
        reconstruction, codebook_loss = self(input)
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
        logits = self.enc(input)
        if self.args.quantizer == "ema" or self.args.quantizer == "origin":
            quant_t, diff_t, id_t = self.quantize_t(logits)
            diff_t = diff_t.unsqueeze(0)
        elif self.args.quantizer == "fsq":
            quant_t, id_t = self.quantize_t(logits)
            diff_t = torch.tensor(0.0).cuda().float()
        elif self.args.quantizer == "lfq":
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
