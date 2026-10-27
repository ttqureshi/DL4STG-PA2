"""Axis-specific operations with no hidden global readout."""
from dataclasses import asdict, dataclass
import torch
from torch import nn
import torch.nn.functional as F

STACKS = {
    "T": ["T"] * 4,
    "S-first": ["S", "T"] * 3,
    "T-first": ["T", "S"] * 3,
    "ST": ["ST"] * 3,
    "A": ["A"] * 3,
}


@dataclass
class Config:
    architecture: str
    width: int = 32
    pairs: int = 3
    heads: int = 4

    def dict(self):
        return asdict(self)


class LocalBlock(nn.Module):
    def __init__(self, kind, width):
        super().__init__()
        self.kind = kind
        self.conv = nn.Conv2d(width, width, (3, 3)) if kind == "ST" else nn.Conv1d(width, width, 3)
        self.norm = nn.LayerNorm(width)  # Per token: never across time or sensors.

    def forward(self, x):
        b, h, n, d = x.shape
        if self.kind == "T":
            z = x.permute(0, 2, 3, 1).reshape(b*n, d, h)
            z = self.conv(F.pad(z, (2, 0))).reshape(b, n, d, h).permute(0, 3, 1, 2)
        elif self.kind == "S":
            z = x.permute(0, 1, 3, 2).reshape(b*h, d, n)
            z = self.conv(F.pad(z, (1, 1))).reshape(b, h, d, n).permute(0, 1, 3, 2)
        else:
            z = self.conv(F.pad(x.permute(0, 3, 1, 2), (1, 1, 2, 0))).permute(0, 2, 3, 1)
        return x + F.gelu(self.norm(z))


class AttentionBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.ff_norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width))

    def forward(self, x, return_attention=False):
        b, h, n, d = x.shape
        z = self.norm(x.reshape(b, h * n, d))
        mixed, weights = self.attention(z, z, z, need_weights=return_attention, average_attn_weights=False)
        z = x.reshape(b, h * n, d) + mixed
        z = z + self.ff(self.ff_norm(z))
        result = z.reshape(b, h, n, d)
        return (result, weights) if return_attention else result


class TeachingModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.architecture in ("S-first", "T-first"):
            start = ["S", "T"] if config.architecture == "S-first" else ["T", "S"]
            self.kinds = start * config.pairs
        else:
            self.kinds = STACKS[config.architecture]
        self.stem = nn.Linear(4, config.width)
        self.blocks = nn.ModuleList([
            AttentionBlock(config.width, config.heads) if k == "A" else LocalBlock(k, config.width)
            for k in self.kinds
        ])
        self.readout = nn.Linear(config.width, 1)

    def forward(self, x, trace=False, attention=False):
        z = self.stem(x)
        states, weights = [z], []
        for block, kind in zip(self.blocks, self.kinds):
            if kind == "A" and attention:
                z, w = block(z, return_attention=True)
                weights.append(w)
            else:
                z = block(z)
            if trace:
                states.append(z)
        # Every prediction reads only its own last-time representation.
        y = self.readout(z[:, -1]).squeeze(-1)
        if trace or attention:
            return y, states, weights
        return y


def key(architecture, pairs=3):
    return f"{architecture}-p{pairs}" if architecture in ("S-first", "T-first") else architecture


def receptive_field(kinds, time, sensor, history=8, sensors=16):
    """Exact structural support, including boundaries and residual paths."""
    reachable = {(time, sensor)}
    for kind in reversed(kinds):
        if kind == "A":
            return torch.ones(history, sensors, dtype=torch.bool).numpy()
        dt = (0, -1, -2) if kind in ("T", "ST") else (0,)
        ds = (-1, 0, 1) if kind in ("S", "ST") else (0,)
        reachable = {(t + a, s + b) for t, s in reachable for a in dt for b in ds
                     if 0 <= t + a < history and 0 <= s + b < sensors}
    mask = torch.zeros(history, sensors, dtype=torch.bool)
    for t, s in reachable:
        mask[t, s] = True
    return mask.numpy()
