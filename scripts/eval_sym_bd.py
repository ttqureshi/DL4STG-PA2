"""Evaluate symmetrised Models B and D. Does not train and does not write weights.

Model B is Model A's architecture, loaded from weights/model_b_seed*.pt.
Model D is loaded from weights/model_d_seed*.pt. Each checkpoint is wrapped as

    f_bar(X) = (1/8) sum_{g in G} g^{-1}(f(g(X))).

Test indices are 9000-9999 from MovingMNIST with split=None. Training time per
epoch is copied from results/table5_b.json and results/table5_d.json.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

EXPECTED_PARAMS = 480784
BATCH_SIZE = 32
SEEDS = (0, 1)
N_TEST = 1000
# A consistency gap this large is not a floating-point residual.
GAP_LIMIT = 1e-5


def _ensure_src_on_path() -> None:
    here = Path(__file__).resolve().parent
    candidates = [here, here.parent, here.parent.parent, Path("/kaggle/working"), Path.cwd()]
    for root in candidates:
        src = root / "src"
        if (src / "dl4stg" / "symmetrise.py").is_file():
            text = str(src)
            if text not in sys.path:
                sys.path.insert(0, text)
            return
    raise RuntimeError("Could not find src/dl4stg/symmetrise.py")


def _output_root() -> Path:
    working = Path("/kaggle/working")
    if working.is_dir() and (working / "src" / "dl4stg" / "symmetrise.py").is_file():
        return working
    here = Path(__file__).resolve().parent
    for root in (here, here.parent, here.parent.parent):
        if (root / "src" / "dl4stg" / "symmetrise.py").is_file() and (root / "results").is_dir():
            return root
    return Path.cwd()


def _data_root() -> str:
    if Path("/kaggle/working").is_dir():
        return "/tmp/moving_mnist"
    return str(_output_root() / "data")


def three_sig(value: float) -> str:
    value = float(value)
    if value == 0.0:
        return "0"
    if abs(value) < 1e-3:
        return f"{value:.2e}"
    return f"{value:.3g}"


def sci(value: float) -> str:
    """Scientific notation, including a floating-point zero."""
    return f"{float(value):.6e}"


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _count_params(model) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _load_state(model, path: Path, device) -> str:
    import torch

    before = _sha256(path)
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    after = _sha256(path)
    if after != before:
        raise SystemExit(f"Loading {path.name} changed the file on disk.")
    model.eval()
    return before


def _read_epoch_seconds(path: Path) -> dict:
    table = json.loads(path.read_text(encoding="utf-8"))
    copied = {
        "0": float(table["per_seed"]["0"]["mean_epoch_seconds"]),
        "1": float(table["per_seed"]["1"]["mean_epoch_seconds"]),
        "mean": float(table["mean_epoch_seconds"]),
        "source": f"results/{path.name}",
    }
    pair = (copied["0"] + copied["1"]) / 2.0
    if abs(pair - copied["mean"]) > 1e-9:
        raise SystemExit(
            f"{path.name} mean_epoch_seconds {copied['mean']} is not the mean of the two seeds."
        )
    return copied


def _gpu_name() -> str:
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        check=False,
        capture_output=True,
        text=True,
    )
    names = [line.strip() for line in (query.stdout or "").splitlines() if line.strip()]
    if not names:
        raise SystemExit("nvidia-smi reported no GPU.")
    if len(names) == 1:
        return names[0]
    return f"{len(names)} x {names[0]}"


def _horizon_means(pred, target):
    import torch

    from dl4stg.metrics import mse_per_horizon

    got = mse_per_horizon(pred, target).detach().to(dtype=torch.float64, device="cpu")
    n = int(target.shape[1])
    manual = (pred[:, :n] - target).pow(2).mean(dim=(0, 2, 3, 4))
    manual = manual.detach().to(dtype=torch.float64, device="cpu")
    if torch.max(torch.abs(got - manual)) > 1e-5:
        raise SystemExit("mse_per_horizon disagrees with the pixel mean.")
    return got


def _display(metrics: dict) -> dict:
    shown = {}
    for key, value in metrics.items():
        shown[key] = sci(value) if key in ("C_M", "C_R") else three_sig(value)
    return shown


def _metric_block(metrics: dict) -> dict:
    block = {key: float(value) for key, value in metrics.items()}
    block["display"] = _display(metrics)
    block["C_M_sci"] = sci(metrics["C_M"])
    block["C_R_sci"] = sci(metrics["C_R"])
    return block


def _mean_metrics(a: dict, b: dict) -> dict:
    keys = [key for key in a if key not in ("display", "C_M_sci", "C_R_sci")]
    return {key: (float(a[key]) + float(b[key])) / 2.0 for key in keys}


def _check_gaps(where: str, metrics: dict) -> None:
    for key in ("C_M", "C_R"):
        gap = abs(float(metrics[key]))
        print(f"{where} {key}={sci(metrics[key])}", flush=True)
        if gap > GAP_LIMIT:
            raise SystemExit(
                f"{where} {key}={sci(metrics[key])} is larger than a floating-point zero."
            )


def _prepare_runtime() -> None:
    import torch

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _load_test():
    from dl4stg.data import load_moving_mnist

    root = _data_root()
    print(f"loading MovingMNIST split=None from {root}", flush=True)
    started = time.perf_counter()
    train, val, test = load_moving_mnist(root, download=True)
    load_seconds = time.perf_counter() - started
    print(
        f"loaded shapes train={tuple(train.shape)} val={tuple(val.shape)} "
        f"test={tuple(test.shape)} seconds={load_seconds:.1f}",
        flush=True,
    )
    if tuple(train.shape) != (8000, 20, 1, 64, 64):
        raise SystemExit(f"unexpected train shape {tuple(train.shape)}")
    if tuple(val.shape) != (1000, 20, 1, 64, 64):
        raise SystemExit(f"unexpected val shape {tuple(val.shape)}")
    if tuple(test.shape) != (N_TEST, 20, 1, 64, 64):
        raise SystemExit(f"unexpected test shape {tuple(test.shape)}")
    # load_moving_mnist slices test as data[9000:10000], indices 9000-9999.
    test = test.clone()
    del train, val
    return test, load_seconds


def _stack_window(videos, t: int, k: int):
    import torch

    from dl4stg.windows import make_window

    inputs = []
    targets = []
    in_frames = None
    tg_frames = None
    for i in range(videos.shape[0]):
        inp, tg, inf, tgf = make_window(videos[i], t, k)
        inputs.append(inp)
        targets.append(tg)
        in_frames, tg_frames = inf, tgf
    return torch.stack(inputs, 0), torch.stack(targets, 0), in_frames, tg_frames


def _score_batched(forecaster, inputs, targets, device, time_forward: bool):
    import torch

    n = int(targets.shape[1])
    total = torch.zeros(n, dtype=torch.float64)
    seen = 0
    seconds = 0.0
    with torch.no_grad():
        for start in range(0, inputs.shape[0], BATCH_SIZE):
            batch_in = inputs[start : start + BATCH_SIZE].to(device, non_blocking=True)
            batch_tg = targets[start : start + BATCH_SIZE].to(device, non_blocking=True)
            if time_forward:
                torch.cuda.synchronize()
                started = time.perf_counter()
            pred = forecaster(batch_in)
            if time_forward:
                torch.cuda.synchronize()
                seconds += time.perf_counter() - started
            total += _horizon_means(pred, batch_tg) * int(batch_in.shape[0])
            seen += int(batch_in.shape[0])
    if seen != inputs.shape[0]:
        raise SystemExit(f"scored {seen} videos, expected {inputs.shape[0]}.")
    return (total / seen).tolist(), seconds


def _score_original_and_consistency(forecaster, test, device):
    """Time only f_bar on the original 1x t=1 clips. Consistency is extra."""
    import torch

    from dl4stg.transforms import mirror, rotate
    from dl4stg.windows import make_window

    reference_in, reference_tg, in_frames, tg_frames = make_window(test[0], 1, 1)
    if in_frames != [1, 2, 3, 4, 5] or tg_frames != [6, 7, 8, 9, 10]:
        raise SystemExit(f"1x t=1 frames {in_frames} {tg_frames}")
    if not torch.equal(reference_in, test[0, 0:5]) or not torch.equal(reference_tg, test[0, 5:10]):
        raise SystemExit("1x t=1 slice disagrees with make_window.")
    sum_h = torch.zeros(5, dtype=torch.float64)
    totals = {key: torch.zeros(5, dtype=torch.float64) for key in ("C_M", "E_M", "C_R", "E_R")}
    seen = 0
    seconds = 0.0
    with torch.no_grad():
        for start in range(0, test.shape[0], BATCH_SIZE):
            videos = test[start : start + BATCH_SIZE].to(device, non_blocking=True)
            inputs = videos[:, 0:5]
            targets = videos[:, 5:10]
            torch.cuda.synchronize()
            started = time.perf_counter()
            pred = forecaster(inputs)
            torch.cuda.synchronize()
            seconds += time.perf_counter() - started
            horizons = _horizon_means(pred, targets)
            batch = int(inputs.shape[0])
            sum_h += horizons * batch
            pred_m = forecaster(mirror(inputs))
            pred_r = forecaster(rotate(inputs, k=1))
            pairs = {
                "C_M": (pred_m, mirror(pred)),
                "E_M": (pred_m, mirror(targets)),
                "C_R": (pred_r, rotate(pred, k=1)),
                "E_R": (pred_r, rotate(targets, k=1)),
            }
            for key, (left, right) in pairs.items():
                totals[key] += _horizon_means(left, right) * batch
            seen += batch
    if seen != N_TEST:
        raise SystemExit(f"inference saw {seen} test videos, expected {N_TEST}.")
    horizons_1x = (sum_h / seen).tolist()
    consistency = {key: (value / seen).tolist() for key, value in totals.items()}
    return horizons_1x, consistency, seconds


def _evaluate_seed(name, builder, weight_path: Path, test, device) -> dict:
    import torch

    from dl4stg.symmetrise import SymmetrisedForecaster

    underlying = builder().to(device)
    params = _count_params(underlying)
    if params != EXPECTED_PARAMS:
        raise SystemExit(f"{name} parameter count {params} != {EXPECTED_PARAMS}.")
    digest = _load_state(underlying, weight_path, device)
    forecaster = SymmetrisedForecaster(underlying).to(device)
    forecaster.eval()
    if _count_params(forecaster) != params:
        raise SystemExit(
            f"{name} wrapper parameter count {_count_params(forecaster)} != {params}."
        )
    if forecaster.training or underlying.training:
        raise SystemExit(f"{name} is not in eval mode.")

    horizons_1x, consistency, infer_seconds = _score_original_and_consistency(
        forecaster, test, device
    )
    if len(horizons_1x) != 5:
        raise SystemExit(f"{name} 1x t=1 produced {len(horizons_1x)} horizons.")
    speed = {"1x": horizons_1x}
    expected_targets = {2: [11, 13, 15, 17, 19], 3: [16, 19]}
    for k, expected in expected_targets.items():
        inputs, targets, in_frames, tg_frames = _stack_window(test, 1, k)
        if in_frames != [1 + i * k for i in range(5)] or tg_frames != expected:
            raise SystemExit(f"{name} {k}x t=1 frames inputs {in_frames} targets {tg_frames}.")
        horizons, _unused = _score_batched(
            forecaster, inputs, targets, device, time_forward=False
        )
        if horizons is None or len(horizons) != len(expected):
            raise SystemExit(f"{name} {k}x scored {len(horizons)} horizons, frames {tg_frames}.")
        speed[f"{k}x"] = horizons
    if len(speed["3x"]) != 2:
        raise SystemExit("3x must score only the first two horizons.")

    def e5(horizons: list[float]) -> float:
        if len(horizons) != 5:
            raise SystemExit("E(5) needs five horizons.")
        return float(sum(horizons) / 5.0)

    def e2(horizons: list[float]) -> float:
        if len(horizons) < 2:
            raise SystemExit("E(2) needs two horizons.")
        return float(sum(horizons[:2]) / 2.0)

    metrics = {
        "E5": e5(horizons_1x),
        "C_M": e5(consistency["C_M"]),
        "E_M": e5(consistency["E_M"]),
        "C_R": e5(consistency["C_R"]),
        "E_R": e5(consistency["E_R"]),
        "E2_1x": e2(speed["1x"]),
        "E2_2x": e2(speed["2x"]),
        "E2_3x": e2(speed["3x"]),
    }
    _check_gaps(f"{name} {weight_path.name}", metrics)
    print(
        f"{name} {weight_path.name} E5={three_sig(metrics['E5'])} "
        f"E_M={three_sig(metrics['E_M'])} E_R={three_sig(metrics['E_R'])} "
        f"E2_1x={three_sig(metrics['E2_1x'])} E2_2x={three_sig(metrics['E2_2x'])} "
        f"E2_3x={three_sig(metrics['E2_3x'])} inference_seconds={infer_seconds:.6f}",
        flush=True,
    )
    del forecaster, underlying
    torch.cuda.empty_cache()
    return {
        "weights": f"weights/{weight_path.name}",
        "weights_sha256": digest,
        "params": params,
        "metrics": _metric_block(metrics),
        "horizons": {
            "1x": horizons_1x,
            "2x": speed["2x"],
            "3x": speed["3x"],
            "C_M": consistency["C_M"],
            "E_M": consistency["E_M"],
            "C_R": consistency["C_R"],
            "E_R": consistency["E_R"],
        },
        "inference_seconds": infer_seconds,
    }


def _evaluate_model(name, builder, weight_names, epoch_seconds, test, device, out: Path) -> dict:
    per_seed = {}
    for seed in SEEDS:
        report = _evaluate_seed(
            name, builder, out / "weights" / weight_names[seed], test, device
        )
        if _sha256(out / "weights" / weight_names[seed]) != report["weights_sha256"]:
            raise SystemExit(f"{weight_names[seed]} changed during evaluation.")
        report["mean_epoch_seconds"] = epoch_seconds[str(seed)]
        per_seed[str(seed)] = report
    mean_metrics = _mean_metrics(per_seed["0"]["metrics"], per_seed["1"]["metrics"])
    _check_gaps(f"{name} mean", mean_metrics)
    inference = [per_seed[str(seed)]["inference_seconds"] for seed in SEEDS]
    return {
        "underlying": "ModelA" if name == "B" else "ModelD",
        "wrapped": "SymmetrisedForecaster",
        "params": EXPECTED_PARAMS,
        "training_time_source": epoch_seconds["source"],
        "training_time_retimed": False,
        "mean_epoch_seconds": epoch_seconds["mean"],
        "per_seed": per_seed,
        "mean": {
            "metrics": _metric_block(mean_metrics),
            "mean_epoch_seconds": epoch_seconds["mean"],
            "inference_seconds": float(sum(inference) / len(inference)),
        },
        "inference_seconds": {
            "seed_0": per_seed["0"]["inference_seconds"],
            "seed_1": per_seed["1"]["inference_seconds"],
            "mean": float(sum(inference) / len(inference)),
        },
    }


def _preflight() -> None:
    """CPU checks. No dataset download, no training, no weight writes."""
    _ensure_src_on_path()
    import torch
    from torch import nn

    from dl4stg.model_a import ModelA
    from dl4stg.model_d import ModelD
    from dl4stg.symmetrise import GROUP, SymmetrisedForecaster
    from dl4stg.transforms import apply_g, invert_g, mirror, rotate

    if len(GROUP) != 8 or len(set(GROUP)) != 8:
        raise SystemExit(f"G is not the eight distinct elements: {GROUP}")
    marker = torch.arange(2 * 5 * 64 * 64, dtype=torch.float32).reshape(2, 5, 1, 64, 64)
    for a, mirrored in GROUP:
        restored = invert_g(apply_g(marker, a, mirrored), a, mirrored)
        if not torch.equal(restored, marker):
            raise SystemExit(f"g^{-1} g is not the identity for a={a} mirrored={mirrored}.")

    class _ColumnBias(nn.Module):
        def forward(self, x):
            cols = torch.linspace(0, 1, x.shape[-1], dtype=x.dtype)
            scale = cols.view(1, 1, 1, 1, -1)
            frame = torch.sigmoid(3.0 * x[:, :1] * scale)
            return frame.repeat(1, 5, 1, 1, 1)

    probe = torch.rand(2, 5, 1, 64, 64)
    raw = _ColumnBias()
    wrapped = SymmetrisedForecaster(raw)

    def gap(model, kind: str) -> float:
        transform = mirror if kind == "M" else (lambda z: rotate(z, 1))
        with torch.no_grad():
            left = model(transform(probe))
            right = transform(model(probe))
        return float((left - right).pow(2).mean())

    for kind in ("M", "R"):
        raw_gap = gap(raw, kind)
        wrapped_gap = gap(wrapped, kind)
        print(f"preflight biased raw {kind}={sci(raw_gap)} wrapped {kind}={sci(wrapped_gap)}", flush=True)
        if raw_gap < 1e-3:
            raise SystemExit(f"The bias probe is already consistent under {kind}.")
        if wrapped_gap > 1e-10:
            raise SystemExit(f"Wrapped bias probe {kind} gap {sci(wrapped_gap)} is not a float zero.")

    torch.manual_seed(0)
    sample = torch.rand(1, 5, 1, 64, 64)
    untrained = ModelA()
    untrained.eval()
    covered = SymmetrisedForecaster(untrained)
    covered.eval()
    with torch.no_grad():
        plain = (mirror(untrained(sample)) - untrained(mirror(sample))).pow(2).mean()
        closed = (mirror(covered(sample)) - covered(mirror(sample))).pow(2).mean()
    print(
        f"preflight untrained ModelA C_M raw={sci(float(plain))} wrapped={sci(float(closed))}",
        flush=True,
    )
    if float(plain) < 1e-6:
        raise SystemExit("Untrained Model A was already mirror-consistent on the probe.")
    if float(closed) > 1e-8:
        raise SystemExit(f"Wrapped untrained Model A C_M {sci(float(closed))} is not a float zero.")
    if _count_params(untrained) != EXPECTED_PARAMS:
        raise SystemExit("Untrained Model A parameter count is not 480784.")
    if untrained.layer1.dilation != 1:
        raise SystemExit("Model A dilation is not 1.")

    out = _output_root()
    specs = (
        ("B", ModelA, "model_b_seed0.pt", 1),
        ("B", ModelA, "model_b_seed1.pt", 1),
        ("D", ModelD, "model_d_seed0.pt", 2),
        ("D", ModelD, "model_d_seed1.pt", 2),
    )
    for name, builder, filename, dilation in specs:
        path = out / "weights" / filename
        before = _sha256(path)
        model = builder()
        _load_state(model, path, torch.device("cpu"))
        if _count_params(model) != EXPECTED_PARAMS:
            raise SystemExit(f"{filename} parameter count is not {EXPECTED_PARAMS}.")
        if model.layer1.dilation != dilation or model.layer2.dilation != dilation:
            raise SystemExit(f"{filename} dilation is not {dilation}.")
        if _sha256(path) != before:
            raise SystemExit(f"{filename} changed during the load check.")
        print(f"preflight loaded {filename} sha256={before} params={EXPECTED_PARAMS}", flush=True)
    for table_name in ("table5_b.json", "table5_d.json"):
        copied = _read_epoch_seconds(out / "results" / table_name)
        print(
            f"preflight {table_name} epoch seconds "
            f"seed0={copied['0']} seed1={copied['1']} mean={copied['mean']}",
            flush=True,
        )
    calls = {"n": 0}

    class _Counting(nn.Module):
        def forward(self, x):
            calls["n"] += 1
            return x[:, :1].repeat(1, 5, 1, 1, 1)

    counter = SymmetrisedForecaster(_Counting())
    with torch.no_grad():
        counter(torch.zeros(1, 5, 1, 64, 64))
    if calls["n"] != 8:
        raise SystemExit(f"The wrapper ran the model {calls['n']} times, expected 8.")
    print("preflight ok", flush=True)


def main() -> None:
    _ensure_src_on_path()
    import torch

    from dl4stg.model_a import ModelA
    from dl4stg.model_d import ModelD

    _prepare_runtime()
    cuda_ok = bool(torch.cuda.is_available())
    print(f"torch.cuda.is_available()={cuda_ok}", flush=True)
    print(f"torch_version={torch.__version__}", flush=True)
    if not cuda_ok:
        raise SystemExit("CUDA is not available. Stopping.")
    device = torch.device("cuda")
    cuda_device = torch.cuda.get_device_name(0)
    gpu_name = _gpu_name()
    print(f"cuda_device={cuda_device} gpu_name={gpu_name}", flush=True)

    out = _output_root()
    print(f"output_root={out}", flush=True)
    epoch_b = _read_epoch_seconds(out / "results" / "table5_b.json")
    epoch_d = _read_epoch_seconds(out / "results" / "table5_d.json")
    weight_paths = [
        out / "weights" / name
        for name in (
            "model_b_seed0.pt",
            "model_b_seed1.pt",
            "model_d_seed0.pt",
            "model_d_seed1.pt",
        )
    ]
    hashes_before = {path.name: _sha256(path) for path in weight_paths}

    test, load_seconds = _load_test()
    models = {
        "B": _evaluate_model(
            "B",
            ModelA,
            {0: "model_b_seed0.pt", 1: "model_b_seed1.pt"},
            epoch_b,
            test,
            device,
            out,
        ),
        "D": _evaluate_model(
            "D",
            ModelD,
            {0: "model_d_seed0.pt", 1: "model_d_seed1.pt"},
            epoch_d,
            test,
            device,
            out,
        ),
    }
    hashes_after = {path.name: _sha256(path) for path in weight_paths}
    if hashes_after != hashes_before:
        raise SystemExit(f"Weight files changed: before={hashes_before} after={hashes_after}")

    table = {
        "table": 6,
        "forecaster": "symmetrised",
        "definition": "f_bar(X) = (1/8) sum_{g in G} g^{-1}(f(g(X)))",
        "group": "G = {R^a, M R^a : a = 0,1,2,3}. Inverse of R^a is R^{-a}. Each M R^a is its own inverse.",
        "final_model": "symmetrised D",
        "d_plus_b_trained": False,
        "trained_this_run": False,
        "weights_modified": False,
        "test_indices": "9000-9999",
        "split": None,
        "n_test": N_TEST,
        "batch_size": BATCH_SIZE,
        "t": 1,
        "reduction": (
            "E(5), C_T and E_T average squared error over the 64x64 pixels and the five "
            "forecast horizons of each video, then over the 1000 test videos, at 1x with t=1. "
            "C_T = MSE(f_bar(T(X)), T(f_bar(X))). E_T = MSE(f_bar(T(X)), T(Y)). "
            "E(2) averages the first two horizon MSEs. At 3x only those two horizons exist."
        ),
        "inference": (
            "Symmetrised forecaster. The clock covers all 8 transforms, underlying forwards, "
            "inverse transforms, and the average, for 1000 original 1x t=1 test videos, "
            "batch 32, eval, no_grad. The clock starts after each batch is on device."
        ),
        "cudnn_deterministic": True,
        "gpu_name": gpu_name,
        "cuda_device": cuda_device,
        "cuda": True,
        "torch_version": torch.__version__,
        "kaggle_kernel": "mtayyabtahir/dl4stg-pa2-sym-bd",
        "dataset_load_seconds_excluded": load_seconds,
        "weights_sha256": hashes_after,
        "models": models,
    }
    destination = out / "results" / "table6_bd.json"
    _write_text(destination, json.dumps(table, indent=2) + "\n")
    if hashes_after != {path.name: _sha256(path) for path in weight_paths}:
        raise SystemExit("Weight files changed while writing the results.")
    print(f"wrote {destination}", flush=True)
    summary = {
        "gpu_name": gpu_name,
        "cuda_device": cuda_device,
        "B": {
            "C_M_sci": models["B"]["mean"]["metrics"]["C_M_sci"],
            "C_R_sci": models["B"]["mean"]["metrics"]["C_R_sci"],
            "seed_0": models["B"]["per_seed"]["0"]["metrics"]["display"],
            "seed_1": models["B"]["per_seed"]["1"]["metrics"]["display"],
            "mean": models["B"]["mean"]["metrics"]["display"],
            "mean_epoch_seconds": models["B"]["mean_epoch_seconds"],
            "inference_seconds": models["B"]["inference_seconds"],
        },
        "D": {
            "C_M_sci": models["D"]["mean"]["metrics"]["C_M_sci"],
            "C_R_sci": models["D"]["mean"]["metrics"]["C_R_sci"],
            "seed_0": models["D"]["per_seed"]["0"]["metrics"]["display"],
            "seed_1": models["D"]["per_seed"]["1"]["metrics"]["display"],
            "mean": models["D"]["mean"]["metrics"]["display"],
            "mean_epoch_seconds": models["D"]["mean_epoch_seconds"],
            "inference_seconds": models["D"]["inference_seconds"],
        },
    }
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    _ensure_src_on_path()
    if "--check" in sys.argv:
        _preflight()
    else:
        main()
