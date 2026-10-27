"""Count Model A parameters and run one dummy forward plus backward.

Does not train. Does not download or read MovingMNIST. The reported total is
``sum(p.numel() for p in model.parameters() if p.requires_grad)``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    here = Path(__file__).resolve()
    for root in (here.parent, *here.parents):
        src = root / "src"
        if (src / "dl4stg" / "model_a.py").is_file():
            text = str(src)
            if text not in sys.path:
                sys.path.insert(0, text)
            return
    raise RuntimeError("Could not find src/dl4stg/model_a.py starting from " + str(here))


def _write_report(text: str) -> None:
    targets = []
    working = Path("/kaggle/working")
    if working.is_dir():
        targets.append(working / "model_a_params.json")
    here = Path(__file__).resolve()
    for root in (here.parent, *here.parents):
        results = root / "results"
        if (root / "src" / "dl4stg").is_dir() and results.is_dir():
            targets.append(results / "model_a_params.json")
            break
    targets.append(Path.cwd() / "model_a_params.json")
    seen: set[Path] = set()
    for path in targets:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(text, encoding="utf-8")


def _conv_report(conv) -> dict:
    k_h, k_w = (int(k) for k in conv.kernel_size)
    weight = int(conv.weight.numel())
    bias = 0 if conv.bias is None else int(conv.bias.numel())
    formula = int(conv.in_channels) * int(conv.out_channels) * k_h * k_w + bias
    return {
        "in_channels": int(conv.in_channels),
        "out_channels": int(conv.out_channels),
        "kernel_size": [k_h, k_w],
        "stride": [int(s) for s in conv.stride],
        "padding": [int(p) for p in conv.padding],
        "bias": conv.bias is not None,
        "weight_numel": weight,
        "bias_numel": bias,
        "params": weight + bias,
        "formula_params": formula,
    }


def main() -> None:
    _ensure_src_on_path()
    import torch
    from torch import nn

    from dl4stg.model_a import ModelA

    torch.manual_seed(0)
    model = ModelA()

    per_parameter = {
        name: int(param.numel())
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    total = int(sum(per_parameter.values()))
    frozen = int(sum(p.numel() for p in model.parameters() if not p.requires_grad))

    def module_params(module: nn.Module) -> int:
        return int(sum(p.numel() for p in module.parameters() if p.requires_grad))

    per_module = {
        "layer1": module_params(model.layer1),
        "layer2": module_params(model.layer2),
        "head": module_params(model.head),
        "unshuffle": module_params(model.unshuffle),
        "shuffle": module_params(model.shuffle),
    }
    breakdown = {
        "layer1": _conv_report(model.layer1.conv),
        "layer2": _conv_report(model.layer2.conv),
        "head": _conv_report(model.head),
    }

    step_frames: list = []
    layer1_h: list = []
    layer2_u: list = []

    def remember_frame(_module, inputs, _output) -> None:
        step_frames.append(inputs[0])

    def remember_h1(_module, _inputs, output) -> None:
        layer1_h.append(output[0])

    def remember_u2(_module, inputs, _output) -> None:
        layer2_u.append(inputs[0])

    hooks = [
        model.unshuffle.register_forward_hook(remember_frame),
        model.layer1.register_forward_hook(remember_h1),
        model.layer2.register_forward_hook(remember_u2),
    ]
    x = torch.zeros(2, 5, 1, 64, 64)
    y = model(x)
    for hook in hooks:
        hook.remove()

    real_inputs_ok = len(step_frames) == 9 and all(
        torch.equal(step_frames[i], x[:, i]) for i in range(5)
    )
    feedback_values_ok = len(step_frames) == 9 and all(
        torch.equal(step_frames[5 + i], y[:, i]) for i in range(4)
    )
    feedback_not_detached = len(step_frames) == 9 and all(
        step_frames[5 + i].grad_fn is not None for i in range(4)
    )
    layer2_reads_new_h1 = (
        len(layer1_h) == 9
        and len(layer2_u) == 9
        and all(torch.equal(layer2_u[i], layer1_h[i]) for i in range(9))
    )

    feedback_reaches_later = False
    if len(step_frames) == 9 and feedback_not_detached:
        feedback_grad = torch.autograd.grad(
            y[:, 1:].sum(),
            step_frames[5],
            retain_graph=True,
            allow_unused=True,
        )[0]
        last_feedback_grad = torch.autograd.grad(
            y[:, 4].sum(),
            step_frames[8],
            retain_graph=True,
            allow_unused=True,
        )[0]
        feedback_reaches_later = (
            feedback_grad is not None
            and float(feedback_grad.abs().sum()) > 0.0
            and last_feedback_grad is not None
            and float(last_feedback_grad.abs().sum()) > 0.0
        )

    loss = torch.nn.functional.mse_loss(y, torch.zeros_like(y))
    loss.backward()
    missing_grads = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad and param.grad is None
    ]
    backward_ok = (
        len(missing_grads) == 0
        and real_inputs_ok
        and feedback_values_ok
        and feedback_not_detached
        and feedback_reaches_later
        and layer2_reads_new_h1
    )

    with torch.no_grad():
        unbatched = model(torch.zeros(5, 1, 64, 64))

    output_min = float(y.detach().min())
    output_max = float(y.detach().max())
    report = {
        "total": total,
        "frozen": frozen,
        "per_module": per_module,
        "per_parameter": per_parameter,
        "breakdown": breakdown,
        "output_shape": list(y.shape),
        "unbatched_output_shape": list(unbatched.shape),
        "output_min": output_min,
        "output_max": output_max,
        "backward_ok": backward_ok,
        "missing_grads": missing_grads,
        "rollout_steps": len(step_frames),
        "real_inputs_ok": real_inputs_ok,
        "feedback_values_ok": feedback_values_ok,
        "feedback_not_detached": feedback_not_detached,
        "feedback_reaches_later": feedback_reaches_later,
        "layer2_reads_new_h1": layer2_reads_new_h1,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": str(next(model.parameters()).device),
        "torch_version": torch.__version__,
        "trained": False,
    }
    text = json.dumps(report, indent=2) + "\n"
    print(text, end="")
    _write_report(text)

    problems: list[str] = []
    formula_total = sum(item["formula_params"] for item in breakdown.values())
    if total != formula_total:
        problems.append(f"numel {total} != formula {formula_total}")
    if total != 480784:
        problems.append(f"total {total} != 480784")
    expected_modules = {"layer1": 184576, "layer2": 295168, "head": 1040}
    for name, count in expected_modules.items():
        if per_module[name] != count or breakdown[name]["params"] != count:
            problems.append(f"{name} measured {per_module[name]}")
    if breakdown["layer1"]["in_channels"] != 80 or breakdown["layer1"]["out_channels"] != 256:
        problems.append("layer1 conv shape")
    if breakdown["layer2"]["in_channels"] != 128 or breakdown["layer2"]["out_channels"] != 256:
        problems.append("layer2 conv shape")
    if breakdown["head"]["in_channels"] != 64 or breakdown["head"]["out_channels"] != 16:
        problems.append("head conv shape")
    if list(y.shape) != [2, 5, 1, 64, 64]:
        problems.append(f"output_shape {tuple(y.shape)}")
    if list(unbatched.shape) != [5, 1, 64, 64]:
        problems.append(f"unbatched_output_shape {tuple(unbatched.shape)}")
    if not (0.0 <= output_min and output_max <= 1.0):
        problems.append(f"output range [{output_min}, {output_max}]")
    if frozen != 0:
        problems.append(f"frozen parameters {frozen}")
    if not backward_ok:
        problems.append("backward_ok is false")
    if torch.cuda.is_available():
        problems.append("cuda is available; this run was supposed to stay on CPU")
    if problems:
        print("FAIL: " + "; ".join(problems), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
