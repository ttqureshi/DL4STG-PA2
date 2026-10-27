"""ConvLSTM cell: one convolution, four gates, no peepholes."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ConvLSTMCell(nn.Module):
    """Single ConvLSTM step.

    One ``Conv2d`` reads ``concat[u_tau, h_{tau-1}]`` along channels and emits
    the four gates in order ``[i, f, o, g]``:

        c_tau = sigmoid(f) * c_{tau-1} + sigmoid(i) * tanh(g)
        h_tau = sigmoid(o) * tanh(c_tau)

    Kernel 3x3, stride 1, bias on. No peephole paths. The convolution keeps
    PyTorch's default initialization.

    ``dilation`` defaults to 1. Padding is then 1, and the cell is the Model A
    cell. A larger dilation uses the same 3x3 weights and a wider
    neighbourhood. Padding is ``dilation * (3 - 1) // 2``, so a 16x16 map
    stays 16x16.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        if dilation < 1:
            raise ValueError(f"dilation must be >= 1, got {dilation}.")
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.dilation = dilation
        kernel_size = 3
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=True,
        )

    def forward(self, u: Tensor, h: Tensor, c: Tensor) -> tuple[Tensor, Tensor]:
        i, f, o, g = self.conv(torch.cat([u, h], dim=1)).chunk(4, dim=1)
        c_tau = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_tau = torch.sigmoid(o) * torch.tanh(c_tau)
        return h_tau, c_tau
