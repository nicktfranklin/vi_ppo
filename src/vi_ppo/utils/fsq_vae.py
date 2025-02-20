import numpy as np
import torch
from quantizers import FSQ, LFQ, VectorQuantizeEMA
from torch import nn
from torch.nn import functional as F

"""
Finite Scalar Quantization: VQ-VAE Made Simple - https://arxiv.org/abs/2309.15505
Code adapted from Jax version in Appendix A.1
"""

from typing import List, Optional

import torch
import torch.nn as nn
from einops import pack, rearrange, unpack
from torch import Tensor, int32
from torch.nn import Module

# helper functions

"""
Lookup Free Quantization
Proposed in https://arxiv.org/abs/2310.05737

basically a 2-level FSQ (Finite Scalar Quantization) with entropy loss
https://arxiv.org/abs/2309.15505
"""

from collections import namedtuple
from math import ceil, log2

import torch
import torch.nn.functional as F
from einops import pack, rearrange, reduce, unpack
from torch import Tensor
from torch import distributed as dist
from torch import einsum, nn
from torch.nn import Module
from torch.nn import functional as F


def all_reduce(tensor, op=dist.ReduceOp.SUM):
    world_size = dist.get_world_size()
    if world_size == 1:
        return tensor
    dist.all_reduce(tensor, op=op)
    return tensor


class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        torch.nn.init.xavier_uniform_(embed, gain=torch.nn.init.calculate_gain("tanh"))
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input, continuous_relax=False, temperature=1.0, hard=False):
        input = input.permute(0, 2, 3, 1)
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )  # dist map, shape=[*, n_embed]

        if not continuous_relax:
            # argmax + lookup
            _, embed_ind = (-dist).max(1)
            embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
            # print(embed_ind.shape)
            # print(input.shape)
            embed_ind = embed_ind.view(*input.shape[:-1])
            quantize = self.embed_code(embed_ind)
        elif not hard:
            # gumbel softmax weighted sum
            embed_soft, embed_ind = gumbel_softmax(-dist, tau=temperature, hard=False)
            embed_ind = embed_ind.view(*input.shape[:-1])
            embed_soft = embed_soft.view(*input.shape[:-1], self.n_embed)
            quantize = embed_soft @ self.embed.transpose(0, 1)
        else:
            # gumbel softmax hard lookup
            embed_onehot, embed_ind = gumbel_softmax(-dist, tau=temperature, hard=True)
            embed_ind = embed_ind.view(*input.shape[:-1])
            quantize = self.embed_code(embed_ind)

        if self.training and ((continuous_relax and hard) or (not continuous_relax)):
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            # dist_fn.all_reduce(embed_onehot_sum)
            # dist_fn.all_reduce(embed_sum)

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)
        if not continuous_relax:
            diff = (quantize.detach() - input).pow(2).mean()
            quantize = input + (quantize - input).detach()
        else:
            # maybe need replace a KL term here
            qy = (-dist).softmax(-1)
            diff = torch.sum(
                qy * torch.log(qy * self.n_embed + 1e-20), dim=-1
            ).mean()  # KL
            # diff = (quantize - input).pow(2).mean().detach() # gumbel softmax do not need diff
            quantize = quantize.to(memory_format=torch.channels_last)
        quantize = quantize.permute(0, 3, 1, 2)
        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class VectorQuantizeEMA(nn.Module):
    def __init__(
        self, args, embedding_dim, n_embed, commitment_cost=1, decay=0.99, eps=1e-5
    ):
        super().__init__()
        self.args = args
        self.ema = True if args.quantizer == "ema" else False

        self.embedding_dim = embedding_dim
        self.n_embed = n_embed
        self.commitment_cost = commitment_cost

        self.embed = nn.Embedding(n_embed, embedding_dim)
        self.embed.weight.data.uniform_(-1.0 / n_embed, 1.0 / n_embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", self.embed.weight.data.clone())

        self.decay = decay
        self.eps = eps

    def forward(self, z_e):
        B, C, H, W = z_e.shape

        # z_e = z
        z_e = z_e.permute(0, 2, 3, 1)  # (B, H, W, C)
        flatten = z_e.reshape(-1, self.embedding_dim)  # (B*H*W, C)

        dist = (
            flatten.pow(2).sum(1, keepdim=True)  # (B*H*W, 1)
            - 2 * flatten @ self.embed.weight.t()  # (B*H*W, n_embed)
            + self.embed.weight.pow(2).sum(1, keepdim=True).t()  # (1, n_embed)
        )  # (B*H*W, n_embed)
        _, embed_ind = (-dist).max(1)  # choose the nearest neighboor
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(
            flatten.dtype
        )  # (BHW, n_embed)
        embed_ind = embed_ind.view(B, H, W)  #

        z_q = self.embed_code(embed_ind)  # B, H, W, C

        if self.training and self.ema:
            embed_onehot_sum = embed_onehot.sum(0)  #
            embed_sum = (flatten.transpose(0, 1) @ embed_onehot).transpose(0, 1)  #

            all_reduce(embed_onehot_sum.contiguous())
            all_reduce(embed_sum.contiguous())

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.embed.weight.data.copy_(embed_normalized)

        if self.ema:
            diff = self.commitment_cost * (z_q.detach() - z_e).pow(2).mean()
        else:
            diff = (
                self.commitment_cost * (z_q.detach() - z_e).pow(2).mean()
                + (z_q - z_e.detach()).pow(2).mean()
            )

        z_q = z_e + (z_q - z_e).detach()
        z_q = z_q.permute(0, 3, 1, 2)  # B,H,W,C -> B,C,H,W
        return z_q, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.weight)


# constants

Return = namedtuple("Return", ["quantized", "indices", "entropy_aux_loss"])

LossBreakdown = namedtuple(
    "LossBreakdown", ["per_sample_entropy", "batch_entropy", "commitment"]
)

# helper functions


def exists(v):
    return v is not None


def default(*args):
    for arg in args:
        if exists(arg):
            return arg() if callable(arg) else arg
    return None


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


# distance


def euclidean_distance_squared(x, y):
    x2 = reduce(x**2, "... n d -> ... n", "sum")
    y2 = reduce(y**2, "n d -> n", "sum")
    xy = einsum("... i d, j d -> ... i j", x, y) * -2
    return rearrange(x2, "... i -> ... i 1") + y2 + xy


# entropy


def log(t, eps=1e-5):
    return t.clamp(min=eps).log()


def entropy(prob):
    return -prob * log(prob)


# class


class LFQ(Module):
    def __init__(
        self,
        *,
        dim=None,
        codebook_size=None,
        entropy_loss_weight=0.1,
        commitment_loss_weight=1.0,
        diversity_gamma=2.5,
        straight_through_activation=nn.Identity(),
        num_codebooks=1,
        keep_num_codebooks_dim=None,
        codebook_scale=1.0,  # for residual LFQ, codebook scaled down by 2x at each layer
    ):
        super().__init__()

        # some assert validations

        assert exists(dim) or exists(
            codebook_size
        ), "either dim or codebook_size must be specified for LFQ"
        assert (
            not exists(codebook_size) or log2(codebook_size).is_integer()
        ), f"your codebook size must be a power of 2 for lookup free quantization (suggested {2 ** ceil(log2(codebook_size))})"

        codebook_size = default(codebook_size, lambda: 2**dim)
        codebook_dim = int(log2(codebook_size))

        codebook_dims = codebook_dim * num_codebooks
        dim = default(dim, codebook_dims)

        has_projections = dim != codebook_dims
        self.project_in = (
            nn.Linear(dim, codebook_dims) if has_projections else nn.Identity()
        )
        self.project_out = (
            nn.Linear(codebook_dims, dim) if has_projections else nn.Identity()
        )
        self.has_projections = has_projections

        self.dim = dim
        self.codebook_dim = codebook_dim
        self.num_codebooks = num_codebooks

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        # straight through activation

        self.activation = straight_through_activation

        # entropy aux loss related weights

        self.diversity_gamma = diversity_gamma
        self.entropy_loss_weight = entropy_loss_weight

        # codebook scale

        self.codebook_scale = codebook_scale

        # commitment loss

        self.commitment_loss_weight = commitment_loss_weight

        # for no auxiliary loss, during inference

        self.register_buffer("mask", 2 ** torch.arange(codebook_dim - 1, -1, -1))
        self.register_buffer("zero", torch.tensor(0.0), persistent=False)

        # codes

        all_codes = torch.arange(codebook_size)
        bits = ((all_codes[..., None].int() & self.mask) != 0).float()
        codebook = self.bits_to_codes(bits)

        self.register_buffer("codebook", codebook, persistent=False)

    def bits_to_codes(self, bits):
        return bits * self.codebook_scale * 2 - self.codebook_scale

    @property
    def dtype(self):
        return self.codebook.dtype

    def indices_to_codes(self, indices, project_out=True):
        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))

        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, "... -> ... 1")

        # indices to codes, which are bits of either -1 or 1

        bits = ((indices[..., None].int() & self.mask) != 0).to(self.dtype)

        codes = self.bits_to_codes(bits)

        codes = rearrange(codes, "... c d -> ... (c d)")

        # whether to project codes out to original dimensions
        # if the input feature dimensions were not log2(codebook size)

        if project_out:
            codes = self.project_out(codes)

        # rearrange codes back to original shape

        if is_img_or_video:
            codes = rearrange(codes, "b ... d -> b d ...")

        return codes

    def forward(self, x, inv_temperature=1.0, return_loss_breakdown=False):
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension, which is also log2(codebook size)
        c - number of codebook dim
        """

        is_img_or_video = x.ndim >= 4

        # standardize image or video into (batch, seq, dimension)

        if is_img_or_video:
            x = rearrange(x, "b d ... -> b ... d")
            x, ps = pack_one(x, "b * d")

        assert (
            x.shape[-1] == self.dim
        ), f"expected dimension of {self.dim} but received {x.shape[-1]}"

        x = self.project_in(x)

        # split out number of codebooks

        x = rearrange(x, "b n (c d) -> b n c d", c=self.num_codebooks)

        # quantize by eq 3.

        original_input = x

        codebook_value = torch.ones_like(x) * self.codebook_scale
        quantized = torch.where(x > 0, codebook_value, -codebook_value)

        # use straight-through gradients with tanh (or custom activation fn) if training

        if self.training:
            x = self.activation(x)
            x = x - x.detach() + quantized
        else:
            x = quantized

        # calculate indices

        indices = reduce((x > 0).int() * self.mask.int(), "b n c d -> b n c", "sum")

        # entropy aux loss

        if self.training and self.entropy_loss_weight != 0:
            print("entropy Not Zero")
            exit()
            distance = euclidean_distance_squared(original_input, self.codebook)

            prob = (-distance * inv_temperature).softmax(dim=-1)

            per_sample_entropy = entropy(prob).mean()

            avg_prob = reduce(prob, "b n c d -> b c d", "mean")
            codebook_entropy = entropy(avg_prob).mean()

            # 1. entropy will be nudged to be low for each code, to encourage the network to output confident predictions
            # 2. codebook entropy will be nudged to be high, to encourage all codes to be uniformly used within the batch

            entropy_aux_loss = (
                per_sample_entropy - self.diversity_gamma * codebook_entropy
            )
        else:
            # if not training, just return dummy 0
            entropy_aux_loss = per_sample_entropy = codebook_entropy = self.zero

        # commit loss

        if self.training:
            commit_loss = F.mse_loss(original_input, quantized.detach())
        else:
            commit_loss = self.zero

        # merge back codebook dim

        x = rearrange(x, "b n c d -> b n (c d)")

        # project out to feature dimension if needed

        x = self.project_out(x)

        # reconstitute image or video dimensions

        if is_img_or_video:
            x = unpack_one(x, ps, "b * d")
            x = rearrange(x, "b ... d -> b d ...")

            indices = unpack_one(indices, ps, "b * c")

        # whether to remove single codebook dim

        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, "... 1 -> ...")

        # complete aux loss

        aux_loss = (
            entropy_aux_loss * self.entropy_loss_weight
            + commit_loss * self.commitment_loss_weight
        )

        ret = Return(x, indices, aux_loss)

        if not return_loss_breakdown:
            return ret

        return ret, LossBreakdown(per_sample_entropy, codebook_entropy, commit_loss)


if __name__ == "__main__":

    quantizer = LFQ(
        codebook_size=65536,  # codebook size, must be a power of 2
        dim=16,  # this is the input feature dimension, defaults to log2(codebook_size) if not defined
        entropy_loss_weight=0.0,  # how much weight to place on entropy loss
        diversity_gamma=1.0,  # within entropy loss, how much weight to give to diversity of codes, taken from https://arxiv.org/abs/1911.05894
    )

    image_feats = torch.randn(1, 16, 32, 32)

    quantized, indices, entropy_aux_loss = quantizer(image_feats)
    print(quantized)
    print(entropy_aux_loss)


def exists(v):
    return v is not None


def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


# tensor helpers


def round_ste(z: Tensor) -> Tensor:
    """Round with straight through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()


# main class


class FSQ(Module):
    def __init__(
        self,
        levels: List[int],
        dim: Optional[int] = None,
        num_codebooks=1,
        keep_num_codebooks_dim: Optional[bool] = None,
        scale: Optional[float] = None,
    ):
        super().__init__()
        _levels = torch.tensor(levels, dtype=int32)
        self.register_buffer("_levels", _levels, persistent=False)

        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=int32)
        self.register_buffer("_basis", _basis, persistent=False)

        self.scale = scale

        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim

        effective_codebook_dim = codebook_dim * num_codebooks
        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        self.dim = default(dim, len(_levels) * num_codebooks)

        has_projections = self.dim != effective_codebook_dim
        self.project_in = (
            nn.Linear(self.dim, effective_codebook_dim)
            if has_projections
            else nn.Identity()
        )
        self.project_out = (
            nn.Linear(effective_codebook_dim, self.dim)
            if has_projections
            else nn.Identity()
        )
        self.has_projections = has_projections

        self.codebook_size = self._levels.prod().item()

        implicit_codebook = self.indices_to_codes(
            torch.arange(self.codebook_size), project_out=False
        )
        self.register_buffer("implicit_codebook", implicit_codebook, persistent=False)

    def bound(self, z: Tensor, eps: float = 1e-3) -> Tensor:
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self._levels - 1) * (1 - eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).tan()
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z: Tensor) -> Tensor:
        """Quantizes z, returns quantized zhat, same shape as z."""
        quantized = round_ste(self.bound(z))
        half_width = self._levels // 2  # Renormalize to [-1, 1].
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def codes_to_indices(self, zhat: Tensor) -> Tensor:
        """Converts a `code` to an index in the codebook."""
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(int32)

    def indices_to_codes(self, indices: Tensor, project_out=True) -> Tensor:
        """Inverse of `codes_to_indices`."""

        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))

        indices = rearrange(indices, "... -> ... 1")
        codes_non_centered = (indices // self._basis) % self._levels
        codes = self._scale_and_shift_inverse(codes_non_centered)

        if self.keep_num_codebooks_dim:
            codes = rearrange(codes, "... c d -> ... (c d)")

        if project_out:
            codes = self.project_out(codes)

        if is_img_or_video:
            codes = rearrange(codes, "b ... d -> b d ...")

        return codes

    def forward(self, z: Tensor) -> Tensor:
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension, which is also log2(codebook size)
        c - number of codebook dim
        """

        is_img_or_video = z.ndim >= 4

        # standardize image or video into (batch, seq, dimension)

        if is_img_or_video:
            z = rearrange(z, "b d ... -> b ... d")
            z, ps = pack_one(z, "b * d")

        assert (
            z.shape[-1] == self.dim
        ), f"expected dimension of {self.dim} but found dimension of {z.shape[-1]}"

        z = self.project_in(z)

        z = rearrange(z, "b n (c d) -> b n c d", c=self.num_codebooks)

        codes = self.quantize(z)
        indices = self.codes_to_indices(codes)

        codes = rearrange(codes, "b n c d -> b n (c d)")

        out = self.project_out(codes)

        # reconstitute image or video dimensions

        if is_img_or_video:
            out = unpack_one(out, ps, "b * d")
            out = rearrange(out, "b ... d -> b d ...")

            indices = unpack_one(indices, ps, "b * c")

        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, "... 1 -> ...")

        return out, indices


if __name__ == "__main__":
    levels = [8, 5, 5, 5]  # see 4.1 and A.4.1 in the paper
    quantizer = FSQ(levels)

    x = torch.randn(1, 4, 16, 16)  # 4 since there are 4 levels
    xhat, indices = quantizer(x)

    print(xhat.shape)  # (1, 1024, 4) - (batch, seq, dim)
    # print(indices.shape) # (1, 1024)    - (batch, seq)


class Encoder(nn.Module):
    def __init__(self, args):
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

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(self, args):
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

    def forward(self, input):
        return self.blocks(input)


class VQVAE(nn.Module):
    def __init__(self, args):
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
            # args.embed_dim = args.lfq_dim
        elif args.quantizer == "fsq":
            self.quantize_t = FSQ(levels=args.levels)
            # args.embed_dim = len(args.levels)
        else:
            print("quantizer error!")
            exit()

        self.enc = Encoder(args)
        self.dec = Decoder(args)

    def forward(self, input, return_id=True):
        (
            quant_t,
            diff,
            id_t,
        ) = self.encode(input)
        dec = self.dec(quant_t)
        if return_id:
            return dec, diff, id_t
        return dec, diff

    def encode(self, input):
        logits = self.enc(input)
        if self.args.quantizer == "ema" or self.args.quantizer == "origin":
            quant_t, diff_t, id_t = self.quantize_t(logits)
            # quant_t = quant_t.permute(0, 3, 1, 2) have change the dimension in quantizer
            diff_t = diff_t.unsqueeze(0)

        elif self.args.quantizer == "fsq":
            quant_t, id_t = self.quantize_t(logits)
            diff_t = torch.tensor(0.0).cuda().float()

        elif self.args.quantizer == "lfq":
            # quantized, indices, entropy_aux_loss = quantizer(image_feats)
            quant_t, id_t, diff_t = self.quantize_t(logits)
        return quant_t, diff_t, id_t

    def decode(self, code):
        return self.dec(code)

    def decode_code(self, code_t):
        quant_t = self.quantize_t.embed_code(code_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        dec = self.dec(quant_t)

        return dec
