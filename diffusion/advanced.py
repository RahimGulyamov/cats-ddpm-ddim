"""Larger diffusion U-Net: scale/shift conditioning, dense skips and attention."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .network import TimeEmbedding


def zero_module(module):
    for parameter in module.parameters():
        nn.init.zeros_(parameter)
    return module


class ConditionedResidual(nn.Module):
    def __init__(self, incoming, outgoing, time_width, dropout):
        super().__init__()
        self.first = nn.Sequential(nn.GroupNorm(8, incoming), nn.SiLU(),
                                   nn.Conv2d(incoming, outgoing, 3, padding=1))
        self.condition = nn.Sequential(nn.SiLU(), nn.Linear(time_width, outgoing * 2))
        self.norm = nn.GroupNorm(8, outgoing)
        self.second = nn.Sequential(nn.SiLU(), nn.Dropout(dropout),
                                    zero_module(nn.Conv2d(outgoing, outgoing, 3, padding=1)))
        self.skip = nn.Conv2d(incoming, outgoing, 1) if incoming != outgoing else nn.Identity()

    def forward(self, x, time):
        h = self.first(x)
        scale, shift = self.condition(time).to(h.dtype).chunk(2, dim=1)
        h = self.norm(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        return self.skip(x) + self.second(h)


class Attention(nn.Module):
    def __init__(self, channels, heads):
        super().__init__()
        if channels % heads:
            raise ValueError("Attention channels must be divisible by heads")
        self.heads = heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.projection = zero_module(nn.Conv2d(channels, channels, 1))

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, self.heads, c // self.heads, h * w).unbind(1)
        attended = F.scaled_dot_product_attention(q.transpose(-1, -2),
                                                 k.transpose(-1, -2), v.transpose(-1, -2))
        return x + self.projection(attended.transpose(-1, -2).reshape(b, c, h, w))


class QualityUNet(nn.Module):
    def __init__(self, base_channels=96, channel_multipliers=(1, 1, 2, 3, 4),
                 blocks_per_level=2, attention_levels=(3, 4), attention_heads=8,
                 dropout=0.1, gradient_checkpointing=True):
        super().__init__()
        if base_channels < 8 or base_channels % 8 or blocks_per_level < 1:
            raise ValueError("Invalid base_channels or blocks_per_level")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        if any(i < 0 or i >= len(channel_multipliers) for i in attention_levels):
            raise ValueError("attention_levels must identify existing U-Net levels")
        self.gradient_checkpointing = gradient_checkpointing
        self.time = TimeEmbedding(base_channels)
        time_width = base_channels * 4
        channels = [base_channels * m for m in channel_multipliers]
        self.input = nn.Conv2d(3, channels[0], 3, padding=1)
        incoming = channels[0]
        self.encoder, self.downsample = nn.ModuleList(), nn.ModuleList()
        for index, outgoing in enumerate(channels):
            blocks = nn.ModuleList()
            for _ in range(blocks_per_level):
                blocks.append(nn.ModuleList([
                    ConditionedResidual(incoming, outgoing, time_width, dropout),
                    Attention(outgoing, attention_heads) if index in attention_levels else nn.Identity()]))
                incoming = outgoing
            self.encoder.append(blocks)
            if index < len(channels) - 1:
                self.downsample.append(nn.Conv2d(outgoing, outgoing, 3, stride=2, padding=1))
        self.middle1 = ConditionedResidual(incoming, incoming, time_width, dropout)
        self.middle_attention = Attention(incoming, attention_heads)
        self.middle2 = ConditionedResidual(incoming, incoming, time_width, dropout)
        self.decoder, self.upsample = nn.ModuleList(), nn.ModuleList()
        for index in reversed(range(len(channels))):
            outgoing = channels[index]
            blocks = nn.ModuleList()
            for _ in range(blocks_per_level):
                blocks.append(nn.ModuleList([
                    ConditionedResidual(incoming + outgoing, outgoing, time_width, dropout),
                    Attention(outgoing, attention_heads) if index in attention_levels else nn.Identity()]))
                incoming = outgoing
            self.decoder.append(blocks)
            if index > 0:
                self.upsample.append(nn.Conv2d(outgoing, outgoing, 3, padding=1))
        self.output = nn.Sequential(nn.GroupNorm(8, channels[0]), nn.SiLU(),
                                    zero_module(nn.Conv2d(channels[0], 3, 3, padding=1)))

    def residual(self, module, x, time):
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(module, x, time, use_reentrant=False)
        return module(x, time)

    def forward(self, x, timesteps):
        time, skips = self.time(timesteps), []
        h = self.input(x)
        for index, blocks in enumerate(self.encoder):
            for residual, attention in blocks:
                h = attention(self.residual(residual, h, time))
                skips.append(h)
            if index < len(self.downsample):
                h = self.downsample[index](h)
        h = self.residual(self.middle2,
                          self.middle_attention(self.residual(self.middle1, h, time)), time)
        for index, blocks in enumerate(self.decoder):
            for residual, attention in blocks:
                skip = skips.pop()
                if h.shape[-2:] != skip.shape[-2:]:
                    raise ValueError("Image dimensions must be divisible by the downsampling factor")
                h = attention(self.residual(residual, torch.cat((h, skip), dim=1), time))
            if index < len(self.upsample):
                h = self.upsample[index](F.interpolate(h, scale_factor=2, mode="nearest"))
        return self.output(h)
