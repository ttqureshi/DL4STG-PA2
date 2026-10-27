"""Table 1 checks for mirror, rotation, and sliding windows.

Prints one JSON object with every measured value and ``passed``.
Also writes ``table1.json`` for ``kaggle kernels output``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _ensure_src_on_path() -> Path:
    """Put the repo ``src`` directory on ``sys.path`` so ``dl4stg`` imports."""
    here = Path(__file__).resolve()
    roots = [here.parent, *here.parents]
    for root in roots:
        src = root / "src"
        if (src / "dl4stg" / "transforms.py").is_file():
            src_text = str(src)
            if src_text not in sys.path:
                sys.path.insert(0, src_text)
            return src
    raise RuntimeError(
        "Could not find src/dl4stg/transforms.py starting from " + str(here)
    )


def _write_report(text: str) -> None:
    targets = [Path.cwd() / "table1.json"]
    kaggle_working = Path("/kaggle/working")
    if kaggle_working.is_dir():
        targets.append(kaggle_working / "table1.json")
    here = Path(__file__).resolve()
    for root in [here.parent, *here.parents]:
        results = root / "results"
        if (root / "src" / "dl4stg").is_dir() and results.is_dir():
            targets.append(results / "table1.json")
            break
    seen = set()
    for path in targets:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(text, encoding="utf-8")


def _bright_pixels(tensor):
    """Return (frame, row, col) for every non-zero pixel. Channel is dropped."""
    coords = (tensor != 0).nonzero(as_tuple=False)
    pixels = []
    for frame, _channel, row, col in coords.tolist():
        pixels.append([int(frame), int(row), int(col)])
    return pixels


def _fills(stack):
    """Mean of each frame. A constant frame of value v has mean v."""
    return [float(frame.mean().item()) for frame in stack]


def _max_abs(delta) -> float:
    return float(delta.abs().max().item())


def run_checks() -> dict:
    import torch

    from dl4stg.transforms import mirror, rotate
    from dl4stg.windows import legal_starts, make_window

    # Import the other Phase 1 modules so a syntax error fails this check.
    # data.py does not download unless load_moving_mnist is called.
    import dl4stg.data  # noqa: F401
    import dl4stg.metrics  # noqa: F401

    torch.manual_seed(0)
    x = torch.rand(20, 1, 64, 64)

    p = torch.zeros(20, 1, 64, 64)
    p[7, 0, 10, 20] = 1

    mm = mirror(mirror(x))
    a_max = _max_abs(mm - x)
    a_equal = bool(torch.equal(mm, x))

    r4 = rotate(x, k=4)
    b_max = _max_abs(r4 - x)
    b_equal = bool(torch.equal(r4, x))

    # R^{-1} is rot90 k=3. Left side is M(R(M(x))).
    mrm = mirror(rotate(mirror(x), k=1))
    r_inv = rotate(x, k=3)
    c_max = _max_abs(mrm - r_inv)
    c_equal = bool(torch.equal(mrm, r_inv))

    mp = mirror(p)
    rp = rotate(p, k=1)
    d_pixels = _bright_pixels(mp)
    e_pixels = _bright_pixels(rp)
    d_pixel = d_pixels[0] if len(d_pixels) == 1 else d_pixels
    e_pixel = e_pixels[0] if len(e_pixels) == 1 else e_pixels

    video = torch.empty(20, 1, 64, 64)
    for frame_number in range(1, 21):
        video[frame_number - 1].fill_(float(frame_number))

    windows = {}
    expected_windows = {
        1: ([1, 2, 3, 4, 5], [6, 7, 8, 9, 10]),
        2: ([1, 3, 5, 7, 9], [11, 13, 15, 17, 19]),
        3: ([1, 4, 7, 10, 13], [16, 19]),
    }
    for speed, (exp_in, exp_tg) in expected_windows.items():
        inputs, targets, input_frames, target_frames = make_window(video, t=1, k=speed)
        windows[speed] = {
            "t": 1,
            "k": speed,
            "input_frames": [int(v) for v in input_frames],
            "target_frames": [int(v) for v in target_frames],
            "input_fills": _fills(inputs),
            "target_fills": _fills(targets),
            "input_shape": list(inputs.shape),
            "target_shape": list(targets.shape),
            "expected_input_frames": exp_in,
            "expected_target_frames": exp_tg,
        }

    expected_starts = {
        ("A", 1): list(range(1, 12)),
        ("B", 1): list(range(1, 12)),
        ("C", 1): list(range(1, 15)),
        ("D", 1): list(range(1, 15)),
        ("C", 2): list(range(1, 9)),
        ("D", 2): list(range(1, 9)),
        ("C", 3): [1, 2],
        ("D", 3): [1, 2],
    }
    measured_starts = {}
    for model, speed in expected_starts:
        key = f"{model}_{speed}x"
        measured_starts[key] = [int(v) for v in legal_starts(model, speed)]

    failures = []
    if a_max != 0.0 or not a_equal:
        failures.append("a")
    if b_max != 0.0 or not b_equal:
        failures.append("b")
    if c_max != 0.0 or not c_equal:
        failures.append("c")
    if d_pixel != [7, 10, 43]:
        failures.append("d")
    if e_pixel != [7, 43, 10]:
        failures.append("e")
    if float(p[7, 0, 10, 20].item()) != 1.0:
        failures.append("marker")

    labels = {1: "f", 2: "g", 3: "h"}
    for speed, label in labels.items():
        window = windows[speed]
        fills_ok = window["input_fills"] == [float(v) for v in window["expected_input_frames"]]
        fills_ok = fills_ok and window["target_fills"] == [
            float(v) for v in window["expected_target_frames"]
        ]
        frames_ok = (
            window["input_frames"] == window["expected_input_frames"]
            and window["target_frames"] == window["expected_target_frames"]
        )
        if not frames_ok or not fills_ok:
            failures.append(label)

    for (model, speed), expected in expected_starts.items():
        key = f"{model}_{speed}x"
        if measured_starts[key] != expected:
            failures.append(f"legal_starts_{key}")

    report = {
        "seed": 0,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "x_shape": list(x.shape),
        "marker_index": [7, 0, 10, 20],
        "marker_value": float(p[7, 0, 10, 20].item()),
        "a_max_abs": a_max,
        "a_torch_equal": a_equal,
        "b_max_abs": b_max,
        "b_torch_equal": b_equal,
        "c_max_abs": c_max,
        "c_torch_equal": c_equal,
        "c_r_inverse": "rot90 k=3",
        "d_bright_pixel": d_pixel,
        "d_bright_count": len(d_pixels),
        "e_bright_pixel": e_pixel,
        "e_bright_count": len(e_pixels),
        "f_input_frames": windows[1]["input_frames"],
        "f_target_frames": windows[1]["target_frames"],
        "f_input_fills": windows[1]["input_fills"],
        "f_target_fills": windows[1]["target_fills"],
        "g_input_frames": windows[2]["input_frames"],
        "g_target_frames": windows[2]["target_frames"],
        "g_input_fills": windows[2]["input_fills"],
        "g_target_fills": windows[2]["target_fills"],
        "h_input_frames": windows[3]["input_frames"],
        "h_target_frames": windows[3]["target_frames"],
        "h_input_fills": windows[3]["input_fills"],
        "h_target_fills": windows[3]["target_fills"],
        "f_input_shape": windows[1]["input_shape"],
        "f_target_shape": windows[1]["target_shape"],
        "g_input_shape": windows[2]["input_shape"],
        "g_target_shape": windows[2]["target_shape"],
        "h_input_shape": windows[3]["input_shape"],
        "h_target_shape": windows[3]["target_shape"],
        "legal_starts": measured_starts,
        "failures": failures,
        "passed": len(failures) == 0,
    }
    return report


def main() -> None:
    _ensure_src_on_path()
    try:
        report = run_checks()
    except Exception as exc:
        report = {"passed": False, "error": f"{type(exc).__name__}: {exc}", "failures": ["exception"]}
        text = json.dumps(report, indent=2) + "\n"
        print(text, end="")
        _write_report(text)
        raise
    text = json.dumps(report, indent=2) + "\n"
    print(text, end="")
    _write_report(text)
    if not report["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
