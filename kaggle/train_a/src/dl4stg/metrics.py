"""Baseline forecasts and horizon-wise MSE metrics.

Shapes
------
Frame stacks are either:
  * unbatched : ``(T, 1, 64, 64)``
  * batched   : ``(B, T, 1, 64, 64)``

Inputs have ``T = 5``. Models always emit 5 forecast frames. Targets may have
fewer than 5 frames (when the clip runs out), so metrics only score the
available horizons.

The pure helpers here (blank / copy-last / per-horizon MSE) are reused across
later phases.
"""

from __future__ import annotations

import torch
from torch import Tensor

N_OUT = 5  # models always emit 5 frames


def _is_batched(x: Tensor) -> bool:
    """A 5-D tensor ``(B, T, 1, H, W)`` is batched; a 4-D ``(T, 1, H, W)`` is not."""
    if x.dim() == 5:
        return True
    if x.dim() == 4:
        return False
    raise ValueError(
        f"Expected 4-D (T,1,H,W) or 5-D (B,T,1,H,W) tensor, got shape {tuple(x.shape)}."
    )


def blank_forecast(x: Tensor) -> Tensor:
    """Return an all-zeros forecast of 5 frames matching ``x``'s spatial shape.

    ``x`` is the input stack ``(T, 1, H, W)`` or ``(B, T, 1, H, W)``. The output
    has the time dimension replaced by ``N_OUT = 5``.
    """
    if _is_batched(x):
        b, _, c, h, w = x.shape
        return torch.zeros((b, N_OUT, c, h, w), dtype=x.dtype, device=x.device)
    _, c, h, w = x.shape
    return torch.zeros((N_OUT, c, h, w), dtype=x.dtype, device=x.device)


def copy_last_forecast(x: Tensor) -> Tensor:
    """Repeat the last input frame 5 times as the forecast.

    ``x`` is the input stack ``(T, 1, H, W)`` or ``(B, T, 1, H, W)``.
    """
    if _is_batched(x):
        last = x[:, -1:].clone()  # (B, 1, C, H, W)
        return last.repeat(1, N_OUT, 1, 1, 1)
    last = x[-1:].clone()  # (1, C, H, W)
    return last.repeat(N_OUT, 1, 1, 1)


def mse_per_horizon(pred: Tensor, target: Tensor) -> Tensor:
    """Mean squared error per forecast horizon, over available target frames.

    Parameters
    ----------
    pred : Tensor
        Forecast of 5 frames: ``(5, 1, H, W)`` or ``(B, 5, 1, H, W)``.
    target : Tensor
        Ground-truth of ``n <= 5`` frames: ``(n, 1, H, W)`` or
        ``(B, n, 1, H, W)``. Only the first ``n`` horizons of ``pred`` are
        scored.

    Returns
    -------
    Tensor
        Shape ``(n,)`` -- the MSE for each available horizon, averaged over
        pixels (and batch, if present). ``n`` is the number of target frames.
    """
    batched = _is_batched(pred)
    if batched:
        n = target.shape[1]
        p = pred[:, :n]  # (B, n, 1, H, W)
        sq = (p - target) ** 2
        # Average over batch and pixels, keep horizon axis.
        return sq.permute(1, 0, 2, 3, 4).reshape(n, -1).mean(dim=1)
    n = target.shape[0]
    p = pred[:n]  # (n, 1, H, W)
    sq = (p - target) ** 2
    return sq.reshape(n, -1).mean(dim=1)
