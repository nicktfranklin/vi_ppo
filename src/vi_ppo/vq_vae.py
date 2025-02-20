from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from vi_ppo.nets.quantizers import FSQ, LFQ, VectorQuantizeEMA


@dataclass
class VQVAEArgs:
    in_channel: int
    channel: int
    embed_dim: int
    quantizer: str
    n_embed: int = None
    lfq_dim: int = None
    entropy_loss_weight: float = None
    codebook_loss_weight: float = None
    levels: int = None


class Encoder(nn.Module):
    def __init__(self, args: VQVAEArgs):
        """
        Initializes the Encoder module.

        Args:
            args: A namespace containing the arguments for the encoder.
        """
        super().__init__()
        in_channel = args.in_channel
        channel = args.channel
        embed_dim = args.embed_dim

        blocks = [
            nn.Conv2d(in_channel, channel, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 4, stride=2, padding=1),
        ]

        blocks.append(nn.ReLU(inplace=True))
        blocks.append(nn.Conv2d(channel, embed_dim, 1))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Encoded tensor.
        """
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(self, args: VQVAEArgs):
        """
        Initializes the Decoder module.

        Args:
            args: A namespace containing the arguments for the decoder.
        """
        super().__init__()

        in_channel = args.embed_dim
        out_channel = args.in_channel
        channel = args.channel

        blocks = [
            nn.ConvTranspose2d(in_channel, channel, 4, stride=2, padding=1),
        ]
        blocks.append(nn.ReLU(inplace=True))
        blocks.extend(
            [
                nn.ConvTranspose2d(channel, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(channel, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, out_channel, 1),
            ]
        )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the decoder.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Decoded tensor.
        """
        return self.blocks(input)


class VQVAE(nn.Module):
    def __init__(self, args: VQVAEArgs):
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

        self.enc = Encoder(args)
        self.dec = Decoder(args)

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
