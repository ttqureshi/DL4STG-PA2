"""Count Model D parameters and check that its one-step reach exceeds Model A.

Does not train. Does not download or read MovingMNIST. Does not read the
validation split or the test split. The reported total is
``sum(p.numel() for p in model.parameters() if p.requires_grad)``.

The run aborts when Model D exceeds 721,176 trainable parameters, when its
one-step reach is not strictly greater than 8 original pixels, or when Model A
no longer counts 480,784.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


CAP = 721176
MODEL_A_PARAMS = 480784
FOLD = 4


def _ensure_src_on_path() -> None:
    here = Path(__file__).resolve()
    for root in (here.parent, *here.parents):
        src = root / "src"
        if (src / "dl4stg" / "model_d.py").is_file():
            text = str(src)
            if text not in sys.path:
                sys.path.insert(0, text)
            return
    raise RuntimeError("Could not find src/dl4stg/model_d.py starting from " + str(here))


def _write_report(text: str) -> None:
    targets = []
    working = Path("/kaggle/working")
    if working.is_dir():
        targets.append(working / "model_d_params.json")
    here = Path(__file__).resolve()
    for root in (here.parent, *here.parents):
        results = root / "results"
        if (root / "src" / "dl4stg").is_dir() and results.is_dir():
            targets.append(results / "model_d_params.json")
            break
    targets.append(Path.cwd() / "model_d_params.json")
    seen: set[Path] = set()
    for path in targets:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(text, encoding="utf-8")


def _count(model) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _cells_moved(conv) -> int:
    """Folded cells one spatial convolution can move, centre to centre.

    A 1x1 kernel moves nothing. A 3x3 moves ``dilation`` cells, because the
    taps sit at offsets ``-dilation, 0, +dilation``. A 5x5 with dilation 1
    moves 2. This is the same arithmetic as the reach note, not a looked-up
    receptive-field formula applied to a stack.
    """
    k_h = int(conv.kernel_size[0])
    k_w = int(conv.kernel_size[1])
    if k_h != k_w:
        raise RuntimeError(f"Expected a square kernel, got {conv.kernel_size}.")
    if k_h <= 1:
        return 0
    dilation = int(conv.dilation[0])
    if int(conv.dilation[1]) != dilation:
        raise RuntimeError(f"Expected square dilation, got {conv.dilation}.")
    return dilation * (k_h - 1) // 2


def _one_step_reach(model) -> dict:
    """Longest path inside one step, in original pixels.

    Layer 2 reads layer 1's new hidden state, so the new input and layer 1's
    previous hidden state pass through both convolutions. Layer 2's previous
    hidden state passes through only the second. The 1x1 head adds its own
    move, which is 0.
    """
    layer1 = _cells_moved(model.layer1.conv)
    layer2 = _cells_moved(model.layer2.conv)
    head = _cells_moved(model.head)
    longest_cells = layer1 + layer2 + head
    return {
        "layer1_folded_cells": layer1,
        "layer2_folded_cells": layer2,
        "head_folded_cells": head,
        "longest_folded_cells": longest_cells,
        "fold_pixels": FOLD,
        "reach_original_pixels": longest_cells * FOLD,
        "layer1_dilation": int(model.layer1.conv.dilation[0]),
        "layer2_dilation": int(model.layer2.conv.dilation[0]),
        "layer1_padding": [int(p) for p in model.layer1.conv.padding],
        "layer2_padding": [int(p) for p in model.layer2.conv.padding],
        "layer1_kernel": [int(k) for k in model.layer1.conv.kernel_size],
        "layer2_kernel": [int(k) for k in model.layer2.conv.kernel_size],
    }


def _conv_params(conv) -> int:
    bias = 0 if conv.bias is None else int(conv.bias.numel())
    return int(conv.weight.numel()) + bias


def main() -> None:
    _ensure_src_on_path()
    import torch

    from dl4stg.model_a import ModelA
    from dl4stg.model_d import ModelD

    torch.manual_seed(0)
    model_a = ModelA()
    model_d = ModelD()

    count_a = _count(model_a)
    count_d = _count(model_d)
    reach_a = _one_step_reach(model_a)
    reach_d = _one_step_reach(model_d)

    x = torch.zeros(2, 5, 1, 64, 64)
    y_d = model_d(x)
    with torch.no_grad():
        unbatched = model_d(torch.zeros(5, 1, 64, 64))
    loss = torch.nn.functional.mse_loss(y_d, torch.zeros_like(y_d))
    loss.backward()
    missing_grads = [
        name
        for name, param in model_d.named_parameters()
        if param.requires_grad and param.grad is None
    ]

    per_module = {
        "layer1": _conv_params(model_d.layer1.conv),
        "layer2": _conv_params(model_d.layer2.conv),
        "head": _conv_params(model_d.head),
    }
    report = {
        "model_d_parameters": count_d,
        "model_a_parameters": count_a,
        "parameter_cap": CAP,
        "count_method": "sum(p.numel() for p in model.parameters() if p.requires_grad)",
        "per_module": per_module,
        "reach_d": reach_d,
        "reach_a": reach_a,
        "output_shape": list(y_d.shape),
        "unbatched_output_shape": list(unbatched.shape),
        "output_min": float(y_d.detach().min()),
        "output_max": float(y_d.detach().max()),
        "missing_grads": missing_grads,
        "trained": False,
        "validation_comparison_run": False,
        "test_split_used": False,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    text = json.dumps(report, indent=2) + "\n"
    print(text, end="")
    _write_report(text)

    problems: list[str] = []
    if count_a != MODEL_A_PARAMS:
        problems.append(f"Model A {count_a} != {MODEL_A_PARAMS}")
    if reach_a["reach_original_pixels"] != 8:
        problems.append(f"Model A reach {reach_a['reach_original_pixels']} != 8")
    if count_d > CAP:
        problems.append(f"Model D {count_d} exceeds {CAP}")
    if reach_d["reach_original_pixels"] <= 8:
        problems.append(
            f"Model D reach {reach_d['reach_original_pixels']} is not strictly greater than 8"
        )
    if count_d != sum(per_module.values()):
        problems.append("module sum does not match numel")
    if reach_d["layer1_dilation"] != 2 or reach_d["layer2_dilation"] != 2:
        problems.append("Model D dilation is not 2 on both layers")
    if reach_d["layer1_padding"] != [2, 2] or reach_d["layer2_padding"] != [2, 2]:
        problems.append("Model D padding is not 2")
    if reach_a["layer1_dilation"] != 1 or reach_a["layer2_dilation"] != 1:
        problems.append("Model A dilation is not 1")
    if reach_a["layer1_padding"] != [1, 1] or reach_a["layer2_padding"] != [1, 1]:
        problems.append("Model A padding is not 1")
    if list(y_d.shape) != [2, 5, 1, 64, 64]:
        problems.append(f"output_shape {tuple(y_d.shape)}")
    if list(unbatched.shape) != [5, 1, 64, 64]:
        problems.append(f"unbatched_output_shape {tuple(unbatched.shape)}")
    if not (0.0 <= report["output_min"] and report["output_max"] <= 1.0):
        problems.append("output left [0, 1]")
    if missing_grads:
        problems.append("missing grads: " + ", ".join(missing_grads))
    if problems:
        print("FAIL: " + "; ".join(problems), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
