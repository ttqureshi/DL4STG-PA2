"""Model D: Model A's ConvLSTM with dilation 2 on both layers.

The module is the architecture only. It is not trained here.
"""

from __future__ import annotations

from .convlstm import ConvLSTMCell
from .model_a import ModelA

_HIDDEN = 64
_DILATION = 2


class ModelD(ModelA):
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

    The rollout is Model A's: ``pixel_unshuffle`` by 4, layer 2 reads layer 1's
    new hidden state, a 1x1 head, ``pixel_shuffle`` by 4, and a sigmoid. Steps
    1..5 consume the five real frames. Steps 6..9 consume the network's own
    previous prediction, not detached. Hidden and cell states start at zero.
    The forecast is the outputs of steps 5..9.

    Both ConvLSTM layers keep the 3x3 kernel and 64 hidden channels, and use
    dilation 2 (padding 2). Dilation does not add weights.
    """

    def __init__(self) -> None:
        super().__init__()
        hidden = self.layer1.hidden_channels
        self.layer1 = ConvLSTMCell(
            in_channels=16,
            hidden_channels=hidden,
            dilation=_DILATION,
        )
        self.layer2 = ConvLSTMCell(
            in_channels=hidden,
            hidden_channels=hidden,
            dilation=_DILATION,
        )
        if hidden != _HIDDEN:
            raise RuntimeError(f"Expected hidden width {_HIDDEN}, got {hidden}.")
