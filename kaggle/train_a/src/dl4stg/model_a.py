"""Model A: two-layer ConvLSTM forecaster on pixel-unshuffled frames.

The module is the architecture only. It is not trained here.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .convlstm import ConvLSTMCell

_INPUT_STEPS = 5
_TOTAL_STEPS = 9
_HIDDEN = 64
_FOLD = 4


class ModelA(nn.Module):
    """Forecast 5 frames from 5 input frames with a 9-step rollout.

    Parameters
    ----------
    x : Tensor
        ``(B, 5, 1, 64, 64)`` or ``(5, 1, 64, 64)``.

    Returns
    -------
    Tensor
        Five frames in ``[0, 1]``, same rank as ``x``:
        ``(B, 5, 1, 64, 64)`` or ``(5, 1, 64, 64)``.

    Each frame is ``pixel_unshuffle``'d by 4 to ``(16, 16, 16)``. Layer 1 reads
    that folded frame. Layer 2 reads layer 1's new hidden state at the same
    step, not the previous one. A 1x1 convolution (64 -> 16), ``pixel_shuffle``
    by 4, and a sigmoid produce the frame.

    Steps 1..5 consume the five real frames. Steps 6..9 consume the network's
    own previous prediction. There is no teacher forcing and the prediction is
    not detached, so a later loss can flow back through the feedback frames.
    Hidden and cell states are zero at the start of every forward. Decoder
    outputs of steps 1..4 are not returned. The forecast is the outputs of
    steps 5..9.
    """

    def __init__(self) -> None:
        super().__init__()
        self.unshuffle = nn.PixelUnshuffle(_FOLD)
        self.layer1 = ConvLSTMCell(in_channels=16, hidden_channels=_HIDDEN)
        self.layer2 = ConvLSTMCell(in_channels=_HIDDEN, hidden_channels=_HIDDEN)
        self.head = nn.Conv2d(_HIDDEN, 16, kernel_size=1, bias=True)
        self.shuffle = nn.PixelShuffle(_FOLD)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 4:
            return self._rollout(x.unsqueeze(0)).squeeze(0)
        if x.dim() != 5:
            raise ValueError(
                "Expected (5, 1, 64, 64) or (B, 5, 1, 64, 64), "
                f"got shape {tuple(x.shape)}."
            )
        return self._rollout(x)

    def _rollout(self, x: Tensor) -> Tensor:
        if x.shape[1] != _INPUT_STEPS or x.shape[2] != 1:
            raise ValueError(
                f"Expected input (B, 5, 1, H, W), got shape {tuple(x.shape)}."
            )
        batch, _, _, height, width = x.shape
        if height % _FOLD or width % _FOLD:
            raise ValueError(
                f"Spatial size must be divisible by {_FOLD}, got {height}x{width}."
            )
        folded_h = height // _FOLD
        folded_w = width // _FOLD
        h1 = x.new_zeros(batch, _HIDDEN, folded_h, folded_w)
        c1 = x.new_zeros(batch, _HIDDEN, folded_h, folded_w)
        h2 = x.new_zeros(batch, _HIDDEN, folded_h, folded_w)
        c2 = x.new_zeros(batch, _HIDDEN, folded_h, folded_w)

        previous: Tensor | None = None
        forecasts: list[Tensor] = []
        for tau in range(1, _TOTAL_STEPS + 1):
            if tau <= _INPUT_STEPS:
                frame = x[:, tau - 1]
            else:
                if previous is None:
                    raise RuntimeError("Missing feedback frame.")
                # Same tensor the decoder just produced. Not detached.
                frame = previous
            h1, c1 = self.layer1(self.unshuffle(frame), h1, c1)
            h2, c2 = self.layer2(h1, h2, c2)
            if tau >= _INPUT_STEPS:
                previous = torch.sigmoid(self.shuffle(self.head(h2)))
                forecasts.append(previous)
        return torch.stack(forecasts, dim=1)
