"""U-Net with residual blocks, timestep embeddings and bottleneck attention."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class TimeEmbedding(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width = width
        self.mlp = nn.Sequential(nn.Linear(width, width * 4), nn.SiLU(),
                                 nn.Linear(width * 4, width * 4))

    def forward(self, t):
        frequencies = torch.exp(torch.arange(self.width // 2, device=t.device)
                                * (-math.log(10000) / (self.width // 2 - 1)))
        phase = t.float()[:, None] * frequencies[None]
        return self.mlp(torch.cat((phase.sin(), phase.cos()), dim=1))


class Residual(nn.Module):
    def __init__(self, incoming, outgoing, time_width):
        super().__init__()
        self.first = nn.Sequential(nn.GroupNorm(8, incoming), nn.SiLU(),
                                   nn.Conv2d(incoming, outgoing, 3, padding=1))
        self.time = nn.Sequential(nn.SiLU(), nn.Linear(time_width, outgoing))
        self.second = nn.Sequential(nn.GroupNorm(8, outgoing), nn.SiLU(),
                                    nn.Conv2d(outgoing, outgoing, 3, padding=1))
        self.skip = nn.Conv2d(incoming, outgoing, 1) if incoming != outgoing else nn.Identity()

    def forward(self, x, time):
        h = self.first(x) + self.time(time)[:, :, None, None]
        return self.second(h) + self.skip(x)


class SpatialAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.output = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, 4, c // 4, h * w).unbind(1)
        attended = F.scaled_dot_product_attention(q.transpose(-1, -2),
                                                 k.transpose(-1, -2), v.transpose(-1, -2))
        return x + self.output(attended.transpose(-1, -2).reshape(b, c, h, w))


class NoiseUNet(nn.Module):
    def __init__(self, base_channels=16, channel_multipliers=(1, 2, 4, 4)):
        super().__init__()
        if base_channels < 8 or base_channels % 8:
            raise ValueError("base_channels must be a positive multiple of eight")
        self.time = TimeEmbedding(base_channels)
        channels = [base_channels * m for m in channel_multipliers]
        self.input = nn.Conv2d(3, channels[0], 3, padding=1)
        self.encoder = nn.ModuleList()
        self.downsample = nn.ModuleList()
        incoming = channels[0]
        for index, outgoing in enumerate(channels):
            self.encoder.append(nn.ModuleList([Residual(incoming, outgoing, base_channels * 4),
                                               Residual(outgoing, outgoing, base_channels * 4)]))
            if index < len(channels) - 1:
                self.downsample.append(nn.Conv2d(outgoing, outgoing, 3, stride=2, padding=1))
            incoming = outgoing
        self.middle1 = Residual(channels[-1], channels[-1], base_channels * 4)
        self.attention = SpatialAttention(channels[-1])
        self.middle2 = Residual(channels[-1], channels[-1], base_channels * 4)
        self.decoder = nn.ModuleList()
        for outgoing in reversed(channels):
            self.decoder.append(nn.ModuleList([Residual(incoming + outgoing, outgoing, base_channels * 4),
                                               Residual(outgoing, outgoing, base_channels * 4)]))
            incoming = outgoing
        self.output = nn.Sequential(nn.GroupNorm(8, channels[0]), nn.SiLU(),
                                    nn.Conv2d(channels[0], 3, 3, padding=1))

    def forward(self, x, timesteps):
        time = self.time(timesteps)
        h, skips = self.input(x), []
        for index, blocks in enumerate(self.encoder):
            for block in blocks:
                h = block(h, time)
            skips.append(h)
            if index < len(self.downsample):
                h = self.downsample[index](h)
        h = self.middle2(self.attention(self.middle1(h, time)), time)
        for blocks, skip in zip(self.decoder, reversed(skips)):
            h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = blocks[0](torch.cat((h, skip), dim=1), time)
            h = blocks[1](h, time)
        return self.output(h)
