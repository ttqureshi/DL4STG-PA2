"""MovingMNIST loading and train/val/test splitting.

NOTE: This phase (Phase 1 / Table 1) does NOT download or load any data. The
function below documents the intended loading behaviour and is exercised only
in a later phase.

Dataset
-------
``torchvision.datasets.MovingMNIST`` with ``split=None`` returns the FULL set of
10000 sequences, each ``(20, 1, 64, 64)`` -> full tensor ``(10000, 20, 1, 64, 64)``
of ``uint8``. We deliberately use ``split=None`` (NOT ``split='train'`` or
``split='test'``) so that we control the split indices ourselves:

  * train : sequences ``0:8000``
  * val   : sequences ``8000:9000``
  * test  : sequences ``9000:10000``

Values are converted to ``float32`` in ``[0, 1]`` by dividing by 255.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor


def load_moving_mnist(
    root: str, download: bool = True
) -> Tuple[Tensor, Tensor, Tensor]:
    """Load MovingMNIST and return ``(train, val, test)`` float tensors.

    Uses ``split=None`` so the dataset exposes all 10000 sequences; we slice
    them into train ``0:8000`` / val ``8000:9000`` / test ``9000:10000``.
    Each returned tensor is ``float32`` in ``[0, 1]`` with per-sequence shape
    ``(20, 1, 64, 64)``.

    This function intentionally performs the download/load lazily and is NOT
    called during Phase 1 (Table 1).
    """
    from torchvision.datasets import MovingMNIST

    dataset = MovingMNIST(root=root, split=None, download=download)
    data = dataset.data.to(torch.float32) / 255.0

    train = data[0:8000]
    val = data[8000:9000]
    test = data[9000:10000]
    return train, val, test
