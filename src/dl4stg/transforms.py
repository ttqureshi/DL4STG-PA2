"""Dihedral-group (D4) transforms for MovingMNIST video tensors.

Frame numbering
---------------
The writeup numbers frames 1..20. This code uses 0-based tensor indices.
The mapping is: writeup frame ``f`` is tensor index ``f - 1``.
So writeup frames 1..20 correspond to tensor indices 0..19.

Marker checks in Table 1 are stated in 0-based index space: the marker uses
0-based frame index 7 (writeup frame 8), and the expected bright pixels are
``(7, 10, 43)`` and ``(7, 43, 10)`` in ``(frame_index, row, col)`` form. These
0-based indices are NOT renumbered to 1-based frame numbers.

Shapes
------
A video is ``(20, 1, 64, 64)`` float. Transforms apply to every frame and act
on the last two dimensions ``(H, W)`` only. They therefore accept:
  * a single frame ``(1, 64, 64)`` or ``(64, 64)``
  * a stack ``(T, 1, 64, 64)`` (or any leading batch/time dims)

Group conventions
-----------------
* Mirror ``M``: ``(M x)[i, j] = x[i, 63 - j]`` -- a flip along the last
  dimension (width), NOT height.
* Rotation ``R``: ``torch.rot90(x, k=1, dims=(-2, -1))`` -- 90 degrees
  anticlockwise. ``R^a`` is ``rot90`` with ``k=a``; ``R^{-1}`` is ``k=3``
  (equivalently ``k=-1``); ``R^4`` is the identity.
* A group element ``g = (a, mirror_flag)`` means: rotate by ``a`` (in 0..3)
  first, then mirror if ``mirror_flag`` is True. ``G`` (the dihedral group D4)
  has 8 elements.
* Each ``M R^a`` is its own inverse. The inverse of ``R^a`` is ``R^{-a}``.
"""

from __future__ import annotations

import torch
from torch import Tensor


def mirror(x: Tensor) -> Tensor:
    """Mirror along width (last dim): ``(M x)[..., i, j] = x[..., i, 63 - j]``."""
    return torch.flip(x, dims=[-1])


def rotate(x: Tensor, k: int = 1) -> Tensor:
    """Rotate anticlockwise by ``k`` right-angle turns on the last two dims.

    ``k=1`` is 90 degrees anticlockwise. ``k`` may be negative; ``k`` is taken
    modulo 4 by ``torch.rot90`` semantics (``k=4`` is identity).
    """
    return torch.rot90(x, k=k, dims=(-2, -1))


def apply_g(x: Tensor, a: int, mirrored: bool) -> Tensor:
    """Apply group element ``g = (a, mirrored)``: rotate by ``a`` then mirror.

    Parameters
    ----------
    x : Tensor
        Frame ``(1, 64, 64)``/``(64, 64)`` or stack ``(T, 1, 64, 64)``.
    a : int
        Number of 90-degree anticlockwise turns (0..3).
    mirrored : bool
        If True, mirror (flip width) after rotating.
    """
    y = rotate(x, k=a % 4)
    if mirrored:
        y = mirror(y)
    return y


def invert_g(x: Tensor, a: int, mirrored: bool) -> Tensor:
    """Apply the inverse of group element ``g = (a, mirrored)`` to ``x``.

    * If not mirrored: ``g = R^a``, so ``g^{-1} = R^{-a}``.
    * If mirrored: ``g = M R^a`` is its own inverse, so ``g^{-1} = g``.
    """
    if mirrored:
        # M R^a is its own inverse.
        return apply_g(x, a, True)
    return rotate(x, k=(-a) % 4)
