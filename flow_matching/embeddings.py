"""Embedding del tiempo t y modulación FiLM."""

import math

import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):
    """t ∈ [0, 1] -> vector de `dim`: senoidal (como en transformers/difusión) + MLP."""

    def __init__(self, dim=128, max_period=10_000.0, scale=1000.0):
        super().__init__()
        assert dim % 2 == 0
        self.dim, self.max_period, self.scale = dim, max_period, scale
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def sinusoidal(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(self.max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        args = (t.float() * self.scale)[:, None] * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, t):
        """t: (B,) -> (B, dim)"""
        return self.mlp(self.sinusoidal(t))


class FiLM(nn.Module):
    """h -> LayerNorm(h) · (1 + escala(t)) + desplazamiento(t).

    La proyección empieza en cero: al inicio la modulación es la identidad
    (sobre la característica normalizada).
    """

    def __init__(self, time_dim, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels, elementwise_affine=False)
        self.proj = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, 2 * channels))
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, h, temb):
        """h: (P, C); temb: (P, time_dim), ya repartido a cada punto."""
        scale, shift = self.proj(temb).chunk(2, dim=-1)
        return self.norm(h) * (1.0 + scale) + shift
