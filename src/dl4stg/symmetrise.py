"""Symmetrised forecaster over the dihedral group G.

``f_bar(X) = (1/8) sum_{g in G} g^{-1}(f(g(X)))``.

``G`` has eight elements: rotate by ``a`` in ``{0, 1, 2, 3}``, then optionally
mirror. The inverse of ``R^a`` is ``R^{-a}``. Each ``M R^a`` is its own
inverse. ``g`` is applied to every frame, the underlying model predicts, the
inverse is applied to every predicted frame, and the eight predictions are
averaged. The underlying network and its weights are not modified.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .transforms import apply_g, invert_g

# Index order matches training: R^0, R^1, R^2, R^3, then M R^a for a = 0..3.
GROUP = tuple((a, mirrored) for mirrored in (False, True) for a in range(4))


class SymmetrisedForecaster(nn.Module):
    """Wrap a forecaster so the forecast is the group average of its predictions.

    Parameters
    ----------
    x : Tensor
        ``(B, 5, 1, 64, 64)`` or ``(5, 1, 64, 64)``.

    Returns
    -------
    Tensor
        Average of the eight restored forecasts, same shape as ``x``.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        if len(GROUP) != 8:
            raise RuntimeError(f"G must have 8 elements, got {len(GROUP)}.")
        self.model = model

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() not in (4, 5):
            raise ValueError(
                "Expected (5, 1, 64, 64) or (B, 5, 1, 64, 64), "
                f"got shape {tuple(x.shape)}."
            )
        total: Tensor | None = None
        for a, mirrored in GROUP:
            transformed = apply_g(x, a, mirrored)
            predicted = self.model(transformed)
            restored = invert_g(predicted, a, mirrored)
            total = restored if total is None else total + restored
        if total is None:
            raise RuntimeError("The group average received no predictions.")
        return total / float(len(GROUP))
