"""Sliding-window extraction for multi-step video forecasting.

Frame numbering
---------------
The writeup numbers frames 1..20 (1-based). This module speaks the writeup's
1-based frame numbers at its interface (start ``t`` and returned frame-number
lists), and converts to 0-based tensor indices internally
(tensor index = frame_number - 1).

Window definition
-----------------
At speed ``k``, starting at writeup frame ``t`` (1-based):
  * inputs  : frames ``t, t+k, t+2k, t+3k, t+4k`` -- all must exist
              (frame number <= 20).
  * targets : up to five frames ``t+5k`` .. ``t+9k``, stopping early when a
              frame number would exceed 20.

Models always emit 5 frames. Missing targets are simply absent from the
returned target tensor (the model still predicts 5, but we only score the
available horizons).

Legal starts
------------
* Models A, B (1x only): need all 5 inputs AND all 5 targets to exist.
  ``t + 9k <= 20`` at ``k = 1`` gives ``t in {1..11}``.
* Models C, D: need all 5 inputs AND at least 2 targets.
  ``t + 6k <= 20`` gives ``t in {1..14}`` at 1x, ``{1..8}`` at 2x,
  ``{1, 2}`` at 3x.

Default eval: ``t = 1``, ``k = 1`` -> inputs 1-5, targets 6-10.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch import Tensor

MAX_FRAME = 20  # number of frames per video (writeup frames 1..20)


def make_window(
    video: Tensor, t: int, k: int
) -> Tuple[Tensor, Tensor, List[int], List[int]]:
    """Extract one (inputs, targets) window from ``video`` at start ``t``, speed ``k``.

    Parameters
    ----------
    video : Tensor
        Shape ``(20, 1, 64, 64)``. Indexed by 0-based tensor index.
    t : int
        1-based writeup start frame number.
    k : int
        Speed (stride) in frames.

    Returns
    -------
    inputs : Tensor
        Shape ``(5, 1, 64, 64)``.
    targets : Tensor
        Shape ``(n_targets, 1, 64, 64)`` where ``n_targets`` may be < 5
        (or 0) when targets run off the end of the clip.
    input_frames : list[int]
        The 5 input frame numbers (1-based).
    target_frames : list[int]
        The target frame numbers (1-based), length == ``n_targets``.
    """
    input_frames = [t + i * k for i in range(5)]
    if input_frames[-1] > MAX_FRAME:
        raise ValueError(
            f"Illegal window: inputs {input_frames} exceed frame {MAX_FRAME} "
            f"(t={t}, k={k})."
        )

    # At most five targets: t+5k .. t+9k. Drop any that fall past frame 20.
    target_frames: List[int] = []
    for j in range(5, 10):
        frame = t + j * k
        if frame > MAX_FRAME:
            break
        target_frames.append(frame)

    # Convert 1-based frame numbers to 0-based tensor indices.
    in_idx = [f - 1 for f in input_frames]
    tg_idx = [f - 1 for f in target_frames]

    inputs = video[in_idx]
    if tg_idx:
        targets = video[tg_idx]
    else:
        targets = video[0:0]  # empty (0, 1, 64, 64)

    return inputs, targets, input_frames, target_frames


def legal_starts(model: str, k: int) -> List[int]:
    """Return the sorted list of legal 1-based start frames for a model at speed ``k``.

    * Models A/B: 1x only; require all 5 inputs and all 5 targets.
    * Models C/D: require all 5 inputs and at least 2 targets.
    """
    m = model.strip().upper()
    starts: List[int] = []
    for t in range(1, MAX_FRAME + 1):
        last_input = t + 4 * k
        if last_input > MAX_FRAME:
            continue
        if m in ("A", "B"):
            if k != 1:
                continue  # Models A/B operate at 1x only.
            # Need all 5 targets: t+5k .. t+9k, so t + 9k <= 20.
            if t + 9 * k <= MAX_FRAME:
                starts.append(t)
        elif m in ("C", "D"):
            # Need at least 2 targets: t+5k and t+6k, so t + 6k <= 20.
            if t + 6 * k <= MAX_FRAME:
                starts.append(t)
        else:
            raise ValueError(f"Unknown model {model!r}; expected one of A,B,C,D.")
    return starts
