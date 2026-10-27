"""Train Model C (two seeds) and write Table 5.

Model C is Model A: the same ConvLSTM, 480,784 parameters, no architecture
change. Training does not use symmetry augmentation. For each video in a
shuffled epoch, k is drawn uniformly from {1, 2, 3} and t is then drawn
uniformly from the legal starts for model C at that k (windows.legal_starts).
The network always emits 5 frames. The loss averages squared error over the
pixels and the available target frames of that video, then over the batch, so
a shorter target does not change that video's weight.

The test split is read only after both seeds have saved their final-epoch
weights. It is not used to choose a checkpoint, a hyperparameter, or a seed.
"""

from __future__ import annotations

import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


EXPECTED_PARAMS = 480784
EPOCHS = 30
BATCH_SIZE = 32
LR = 1e-3
SEEDS = (0, 1)
SPEEDS = (1, 2, 3)


def _ensure_src_on_path() -> None:
    here = Path(__file__).resolve().parent
    candidates = [here, here.parent, here.parent.parent, Path("/kaggle/working"), Path.cwd()]
    for root in candidates:
        src = root / "src"
        if (src / "dl4stg" / "model_a.py").is_file() and (src / "dl4stg" / "windows.py").is_file():
            text = str(src)
            if text not in sys.path:
                sys.path.insert(0, text)
            return
    raise RuntimeError("Could not find src/dl4stg/model_a.py and windows.py")


def _output_root() -> Path:
    working = Path("/kaggle/working")
    if working.is_dir():
        return working
    here = Path(__file__).resolve().parent
    for root in (here, here.parent, here.parent.parent):
        if (root / "src" / "dl4stg" / "model_a.py").is_file() and (root / "results").is_dir():
            return root
    return Path.cwd()


def _data_root() -> str:
    if Path("/kaggle/working").is_dir():
        return "/tmp/moving_mnist"
    here = Path(__file__).resolve().parent
    for root in (here, here.parent, here.parent.parent):
        if (root / "src" / "dl4stg" / "model_a.py").is_file():
            return str(root / "data")
    return str(Path.cwd() / "data")


def three_sig(value: float) -> str:
    value = float(value)
    if value == 0.0:
        return "0"
    if abs(value) < 1e-3:
        return f"{value:.2e}"
    return f"{value:.3g}"


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _print_nvidia_smi() -> tuple[str, str]:
    try:
        full = subprocess.run(["nvidia-smi"], check=False, capture_output=True, text=True)
    except OSError as exc:
        raise SystemExit(f"nvidia-smi unavailable ({exc}); refusing to train.") from exc
    full_text = (full.stdout or "") + (full.stderr or "")
    print(full_text, flush=True)
    if full.returncode != 0:
        raise SystemExit("nvidia-smi failed; refusing to train.")
    try:
        query = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise SystemExit(f"nvidia-smi query unavailable ({exc}); refusing to train.") from exc
    if query.returncode != 0:
        raise SystemExit("nvidia-smi GPU-name query failed; refusing to train.")
    names = [line.strip() for line in (query.stdout or "").splitlines() if line.strip()]
    if not names:
        raise SystemExit("nvidia-smi reported no GPU; refusing to train.")
    if len(names) == 1:
        gpu_name = names[0]
    else:
        gpu_name = f"{len(names)} x {names[0]}"
    print(f"gpu_name={gpu_name}", flush=True)
    return gpu_name, full_text


def _count_params(model) -> tuple[int, int]:
    total = int(sum(p.numel() for p in model.parameters()))
    trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    return total, trainable


def _require_param_count(model, where: str) -> int:
    total, trainable = _count_params(model)
    print(f"{where} total_parameters={total} trainable_parameters={trainable}", flush=True)
    if total != EXPECTED_PARAMS or trainable != EXPECTED_PARAMS:
        raise SystemExit(
            f"{where} parameter count total={total} trainable={trainable} "
            f"!= {EXPECTED_PARAMS}; aborting."
        )
    return trainable


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _expected_legal() -> dict[int, list[int]]:
    return {1: list(range(1, 15)), 2: list(range(1, 9)), 3: [1, 2]}


def _legal_starts() -> dict[int, list[int]]:
    from dl4stg.windows import legal_starts

    expected = _expected_legal()
    legal = {}
    for k, want in expected.items():
        got = legal_starts("C", k)
        if got != want:
            raise SystemExit(f"Model C legal starts at {k}x are {got}, expected {want}.")
        legal[k] = list(got)
    return legal


def _draw_kt(batch: int, generator, legal: dict[int, list[int]]) -> tuple[list[int], list[int]]:
    """Draw k uniformly from {1, 2, 3}, then t uniformly from legal starts at that k."""
    import torch

    ks: list[int] = []
    ts: list[int] = []
    for _ in range(batch):
        k = int(torch.randint(1, 4, (), generator=generator).item())
        starts = legal[k]
        index = int(torch.randint(0, len(starts), (), generator=generator).item())
        ks.append(k)
        ts.append(starts[index])
    return ks, ts


def _windows_for_batch(videos, ks: list[int], ts: list[int]):
    from dl4stg.windows import make_window

    if len(ks) != int(videos.shape[0]) or len(ts) != len(ks):
        raise SystemExit(
            f"window batch {int(videos.shape[0])} does not match {len(ks)} draws"
        )
    inputs = []
    targets = []
    counts = []
    for i, (k, t) in enumerate(zip(ks, ts)):
        inp, tg, in_frames, tg_frames = make_window(videos[i], t, k)
        if in_frames != [t + step * k for step in range(5)]:
            raise SystemExit(f"input frames {in_frames} at t={t} k={k}")
        if len(tg_frames) != int(tg.shape[0]) or len(tg_frames) < 2:
            raise SystemExit(f"t={t} k={k} has targets {tg_frames}")
        if tg_frames != [t + step * k for step in range(5, 10) if t + step * k <= 20]:
            raise SystemExit(f"target frames {tg_frames} at t={t} k={k}")
        inputs.append(inp)
        targets.append(tg)
        counts.append(len(tg_frames))
    return __import__("torch").stack(inputs, 0), targets, counts


def equal_video_loss(pred, targets: list) :
    """Mean over videos of the mean squared error on that video's available frames."""
    import torch

    losses = []
    for i, target in enumerate(targets):
        n = int(target.shape[0])
        if n < 2 or n > int(pred.shape[1]):
            raise RuntimeError(f"expected 2..5 targets, got {n}")
        losses.append((pred[i, :n] - target).pow(2).mean())
    return torch.stack(losses).mean()


def _self_check() -> None:
    import torch

    from dl4stg.metrics import mse_per_horizon
    from dl4stg.windows import make_window

    legal = _legal_starts()
    video = torch.arange(20 * 4 * 4, dtype=torch.float32).reshape(20, 1, 4, 4)
    expected_targets = {
        (1, 1): [6, 7, 8, 9, 10],
        (1, 12): [17, 18, 19, 20],
        (1, 14): [19, 20],
        (2, 1): [11, 13, 15, 17, 19],
        (2, 8): [18, 20],
        (3, 1): [16, 19],
        (3, 2): [17, 20],
    }
    for k, starts in legal.items():
        for t in starts:
            inp, tg, in_frames, tg_frames = make_window(video, t, k)
            if in_frames != [t + step * k for step in range(5)]:
                raise SystemExit(f"self-check inputs {in_frames}")
            want = [t + step * k for step in range(5, 10) if t + step * k <= 20]
            if tg_frames != want or int(tg.shape[0]) != len(want) or len(want) < 2:
                raise SystemExit(f"self-check targets k={k} t={t} {tg_frames}")
            if not torch.equal(inp, video[[f - 1 for f in in_frames]]):
                raise SystemExit("self-check input pixels")
            if not torch.equal(tg, video[[f - 1 for f in tg_frames]]):
                raise SystemExit("self-check target pixels")
    for key, frames in expected_targets.items():
        _inp, _tg, _inf, got = make_window(video, key[1], key[0])
        if got != frames:
            raise SystemExit(f"self-check {key} targets {got} != {frames}")

    pred = torch.zeros(2, 5, 1, 2, 2)
    long_target = torch.ones(5, 1, 2, 2)
    short_target = torch.zeros(2, 1, 2, 2)
    loss = equal_video_loss(pred, [long_target, short_target])
    if abs(float(loss) - 0.5) > 1e-6:
        raise SystemExit(f"equal-video loss is {float(loss)}, expected 0.5")
    pooled = (pred[0, :5] - long_target).pow(2).sum() + (pred[1, :2] - short_target).pow(2).sum()
    pooled = pooled / ((5 + 2) * 1 * 2 * 2)
    if abs(float(pooled) - 5.0 / 7.0) > 1e-6:
        raise SystemExit("pooled reference for the unequal-target check is wrong")
    if abs(float(loss) - float(pooled)) < 1e-3:
        raise SystemExit("equal-video loss collapsed to a frame-weighted pool")

    full_pred = torch.rand(4, 5, 1, 8, 8)
    full_target = torch.rand(4, 5, 1, 8, 8)
    got = equal_video_loss(full_pred, [full_target[i] for i in range(4)])
    ref = mse_per_horizon(full_pred, full_target).mean()
    flat = (full_pred - full_target).pow(2).mean()
    if abs(float(got) - float(ref)) > 1e-6 or abs(float(got) - float(flat)) > 1e-6:
        raise SystemExit("equal-length loss disagrees with mse_per_horizon")

    graph = torch.randn(1, 5, 1, 2, 2, requires_grad=True)
    equal_video_loss(graph, [torch.randn(2, 1, 2, 2)]).backward()
    if graph.grad is None or float(graph.grad[0, :2].abs().sum()) <= 0.0:
        raise SystemExit("loss did not reach the available frames")
    if float(graph.grad[0, 2:].abs().sum()) != 0.0:
        raise SystemExit("loss reached frames past the available targets")

    generator = torch.Generator().manual_seed(0)
    ks, ts = _draw_kt(300, generator, legal)
    if set(ks) != {1, 2, 3}:
        raise SystemExit(f"k draw missed a speed: {sorted(set(ks))}")
    for k, t in zip(ks, ts):
        if t not in legal[k]:
            raise SystemExit(f"illegal draw k={k} t={t}")
    print(
        "self-check ok: legal starts 1x 1..14, 2x 1..8, 3x {1,2}; "
        "equal-video loss; unused forecast frames get no gradient",
        flush=True,
    )


def _check_training_video(train, legal: dict[int, list[int]]) -> None:
    import torch

    from dl4stg.windows import make_window

    video = train[0]
    for k, starts in legal.items():
        for t in starts:
            _inp, tg, _inf, tg_frames = make_window(video, t, k)
            if int(tg.shape[0]) != len(tg_frames) or len(tg_frames) < 2:
                raise SystemExit(f"train video window k={k} t={t} targets {tg_frames}")
    batch = video.unsqueeze(0).repeat(4, 1, 1, 1, 1)
    inputs, targets, counts = _windows_for_batch(
        batch,
        [1, 2, 3, 1],
        [1, 1, 1, 14],
    )
    if counts != [5, 5, 2, 2]:
        raise SystemExit(f"batched target counts {counts}, expected [5, 5, 2, 2]")
    if tuple(inputs.shape) != (4, 5, 1, 64, 64):
        raise SystemExit(f"batched inputs {tuple(inputs.shape)}")
    pred = torch.zeros_like(inputs)
    loss = equal_video_loss(pred, targets)
    manual = []
    for target in targets:
        manual.append(float(target.pow(2).mean()))
    if abs(float(loss) - float(sum(manual) / len(manual))) > 1e-6:
        raise SystemExit("training-video loss does not match the per-video mean")
    print("window check ok on training video 0", flush=True)


def _train_seed(seed: int, train, device, out: Path, legal: dict[int, list[int]]) -> dict:
    import torch
    from torch.utils.data import DataLoader, Dataset

    from dl4stg.model_a import ModelA

    _set_seed(seed)
    model = ModelA().to(device)
    params = _require_param_count(model, f"seed={seed}")
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0)

    class _Videos(Dataset):
        def __init__(self, videos):
            self.videos = videos

        def __len__(self):
            return int(self.videos.shape[0])

        def __getitem__(self, idx):
            return self.videos[idx]

    g_loader = torch.Generator()
    g_loader.manual_seed(seed)
    g_windows = torch.Generator()
    g_windows.manual_seed(seed)
    loader = DataLoader(
        _Videos(train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=g_loader,
        num_workers=0,
        drop_last=False,
    )

    history = []
    k_counts = {1: 0, 2: 0, 3: 0}
    for epoch in range(1, EPOCHS + 1):
        model.train()
        started = time.perf_counter()
        running = 0.0
        seen = 0
        for videos in loader:
            videos = videos.to(device, non_blocking=True)
            ks, ts = _draw_kt(int(videos.shape[0]), g_windows, legal)
            inputs, targets, counts = _windows_for_batch(videos, ks, ts)
            if int(inputs.shape[0]) != len(counts):
                raise SystemExit("window batch size mismatch")
            pred = model(inputs)
            if tuple(pred.shape) != (inputs.shape[0], 5, 1, 64, 64):
                raise SystemExit(f"model emitted {tuple(pred.shape)}, expected 5 frames")
            loss = equal_video_loss(pred, targets)
            if not math.isfinite(float(loss.detach())):
                raise SystemExit(f"seed {seed} epoch {epoch} non-finite loss")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            batch = int(inputs.shape[0])
            running += float(loss.detach()) * batch
            seen += batch
            for k, n in zip(ks, counts):
                k_counts[k] += 1
                if n < 2:
                    raise SystemExit(f"scored {n} targets")
        seconds = time.perf_counter() - started
        if seen != int(train.shape[0]):
            raise SystemExit(
                f"seed {seed} epoch {epoch} saw {seen} videos, expected {train.shape[0]}."
            )
        train_loss = running / seen
        history.append({"epoch": epoch, "seconds": seconds, "train_loss": train_loss})
        print(
            f"seed={seed} epoch={epoch}/{EPOCHS} train_loss={train_loss:.6f} seconds={seconds:.3f}",
            flush=True,
        )

    weights = out / "weights" / f"model_c_seed{seed}.pt"
    weights.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weights)
    report = {
        "seed": seed,
        "model": "C",
        "architecture": "ModelA",
        "params": params,
        "symmetry_augmentation": False,
        "k_counts": {str(k): k_counts[k] for k in SPEEDS},
        "epochs": history,
    }
    _write_text(out / "results" / f"train_c_seed{seed}.json", json.dumps(report, indent=2) + "\n")
    print(f"saved {weights}", flush=True)
    del model
    torch.cuda.empty_cache()
    return report


def _stack_windows(videos, t: int, k: int):
    from dl4stg.windows import make_window

    inputs = []
    targets = []
    in_frames = None
    tg_frames = None
    for i in range(int(videos.shape[0])):
        inp, tg, inf, tgf = make_window(videos[i], t, k)
        inputs.append(inp)
        targets.append(tg)
        in_frames, tg_frames = inf, tgf
    stacked_targets = __import__("torch").stack(targets, 0)
    return __import__("torch").stack(inputs, 0), stacked_targets, in_frames, tg_frames


def _build_eval_windows(test) -> dict:
    import torch

    expected = {
        1: ([1, 2, 3, 4, 5], [6, 7, 8, 9, 10]),
        2: ([1, 3, 5, 7, 9], [11, 13, 15, 17, 19]),
        3: ([1, 4, 7, 10, 13], [16, 19]),
    }
    windows = {}
    for k, (want_in, want_tg) in expected.items():
        inputs, targets, in_frames, tg_frames = _stack_windows(test, 1, k)
        if in_frames != want_in or tg_frames != want_tg:
            raise SystemExit(f"k={k} frames {in_frames} {tg_frames} != {want_in} {want_tg}")
        if k == 3 and int(targets.shape[1]) != 2:
            raise SystemExit("3x t=1 should expose exactly two targets")
        if k != 3 and int(targets.shape[1]) != 5:
            raise SystemExit(f"{k}x t=1 should expose five targets")
        windows[k] = {
            "inputs": inputs,
            "targets": targets,
            "input_frames": in_frames,
            "target_frames": tg_frames,
        }
        print(
            f"eval window k={k} inputs={in_frames} targets={tg_frames} "
            f"input_shape={tuple(inputs.shape)} target_shape={tuple(targets.shape)}",
            flush=True,
        )
    if not torch.equal(windows[1]["inputs"], test[:, 0:5]):
        raise SystemExit("1x t=1 inputs are not frames 1-5")
    if not torch.equal(windows[1]["targets"], test[:, 5:10]):
        raise SystemExit("1x t=1 targets are not frames 6-10")
    return windows


def _predict(model, inputs, device, time_it: bool) -> tuple:
    import torch

    model.eval()
    pieces = []
    seconds = 0.0
    seen = 0
    with torch.no_grad():
        for start in range(0, int(inputs.shape[0]), BATCH_SIZE):
            batch = inputs[start : start + BATCH_SIZE].to(device, non_blocking=True)
            if time_it:
                torch.cuda.synchronize()
                started = time.perf_counter()
            pred = model(batch)
            if time_it:
                torch.cuda.synchronize()
                seconds += time.perf_counter() - started
            if tuple(pred.shape) != (batch.shape[0], 5, 1, 64, 64):
                raise SystemExit(f"eval emitted {tuple(pred.shape)}")
            pieces.append(pred.detach().cpu())
            seen += int(batch.shape[0])
    if seen != int(inputs.shape[0]):
        raise SystemExit(f"prediction saw {seen} videos, expected {inputs.shape[0]}")
    return torch.cat(pieces, 0), seconds


def _reduced_mse(pred, target) -> tuple[float, list[float]]:
    import torch

    from dl4stg.metrics import mse_per_horizon

    n = int(target.shape[1])
    if n < 1 or n > int(pred.shape[1]):
        raise SystemExit(f"cannot score {n} targets")
    ref = mse_per_horizon(pred, target)
    err = (pred[:, :n] - target).pow(2)
    per_horizon_video = err.mean(dim=(2, 3, 4))
    horizons = per_horizon_video.mean(dim=0)
    if torch.max(torch.abs(ref - horizons)).item() > 1e-5:
        raise SystemExit("mse_per_horizon disagrees with the pixel mean")
    per_video = per_horizon_video.mean(dim=1)
    scalar = float(horizons.mean().item())
    if abs(scalar - float(per_video.mean().item())) > 1e-5:
        raise SystemExit("horizon mean and per-video mean disagree")
    return scalar, [float(v) for v in horizons.tolist()]


def _score_transform(fx, ftx, y, kind: str) -> dict[str, float]:
    from dl4stg.transforms import mirror, rotate

    if kind == "M":
        transform = mirror
    elif kind == "R":
        transform = rotate
    else:
        raise SystemExit(f"unknown transform {kind}")

    def apply(tensor):
        if kind == "R":
            return transform(tensor, k=1)
        return transform(tensor)

    tfx = apply(fx)
    ty = apply(y)
    c_scalar, c_horizons = _reduced_mse(ftx, tfx)
    e_scalar, e_horizons = _reduced_mse(ftx, ty)
    return {
        "C": c_scalar,
        "E": e_scalar,
        "C_horizons": c_horizons,
        "E_horizons": e_horizons,
    }


def _evaluate_seed(seed: int, windows: dict, device, out: Path) -> dict:
    import torch

    from dl4stg.model_a import ModelA
    from dl4stg.transforms import mirror, rotate

    model = ModelA().to(device)
    _require_param_count(model, f"eval seed={seed}")
    weights_path = out / "weights" / f"model_c_seed{seed}.pt"
    try:
        state = torch.load(weights_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Original forecaster only. This clock is the 1x t=1 forward, after the
    # batch is on device. Mirror, rotation, 2x, and 3x are not included.
    fx, seconds = _predict(model, windows[1]["inputs"], device, time_it=True)
    _e5, horizons_1x = _reduced_mse(fx, windows[1]["targets"])
    if len(horizons_1x) != 5:
        raise SystemExit("1x eval did not score five horizons")
    e5 = float(sum(horizons_1x) / 5.0)
    if abs(e5 - _e5) > 1e-5:
        raise SystemExit("E(5) disagrees with the horizon mean")

    transforms = {}
    for kind in ("M", "R"):
        original = windows[1]["inputs"]
        if kind == "M":
            tx = mirror(original)
        else:
            tx = rotate(original, k=1)
        ftx, _unused = _predict(model, tx, device, time_it=False)
        transforms[kind] = _score_transform(fx, ftx, windows[1]["targets"], kind)

    speed_horizons = {1: horizons_1x}
    for k in (2, 3):
        pred, _unused = _predict(model, windows[k]["inputs"], device, time_it=False)
        _scalar, horiz = _reduced_mse(pred, windows[k]["targets"])
        if k == 3 and len(horiz) != 2:
            raise SystemExit(f"3x scored {len(horiz)} horizons, expected 2")
        if k == 2 and len(horiz) != 5:
            raise SystemExit(f"2x scored {len(horiz)} horizons, expected 5")
        speed_horizons[k] = horiz

    def e2(horizons: list[float]) -> float:
        if len(horizons) < 2:
            raise SystemExit("E(2) needs two horizons")
        return float((horizons[0] + horizons[1]) / 2.0)

    metrics = {
        "E5_1x": e5,
        "C_M": transforms["M"]["C"],
        "E_M": transforms["M"]["E"],
        "C_R": transforms["R"]["C"],
        "E_R": transforms["R"]["E"],
        "E2_1x": e2(speed_horizons[1]),
        "E2_2x": e2(speed_horizons[2]),
        "E2_3x": e2(speed_horizons[3]),
    }
    print(
        f"seed={seed} "
        + " ".join(f"{key}={three_sig(value)}" for key, value in metrics.items())
        + f" inference_seconds={seconds:.3f}",
        flush=True,
    )
    del model
    torch.cuda.empty_cache()
    return {
        "metrics": metrics,
        "horizons": {f"{k}x": speed_horizons[k] for k in SPEEDS},
        "inference_seconds": seconds,
    }


def _metric_block(values: dict[str, float]) -> dict:
    block = {key: float(val) for key, val in values.items()}
    block["display"] = {key: three_sig(val) for key, val in values.items()}
    return block


def _load_splits():
    from dl4stg.data import load_moving_mnist

    root = _data_root()
    print(f"loading MovingMNIST split=None from {root}", flush=True)
    started = time.perf_counter()
    train, val, test = load_moving_mnist(root, download=True)
    print(
        f"loaded shapes train={tuple(train.shape)} val={tuple(val.shape)} test={tuple(test.shape)} "
        f"seconds={time.perf_counter() - started:.1f}",
        flush=True,
    )
    if tuple(train.shape) != (8000, 20, 1, 64, 64):
        raise SystemExit(f"unexpected train shape {tuple(train.shape)}")
    if tuple(test.shape) != (1000, 20, 1, 64, 64):
        raise SystemExit(f"unexpected test shape {tuple(test.shape)}")
    if tuple(val.shape) != (1000, 20, 1, 64, 64):
        raise SystemExit(f"unexpected val shape {tuple(val.shape)}")
    train = train.clone()
    test = test.clone()
    del val
    return train, test


def main() -> None:
    _ensure_src_on_path()
    import torch

    from dl4stg.model_a import ModelA

    probe = ModelA()
    _require_param_count(probe, "startup")
    del probe
    _self_check()

    gpu_name, _smi = _print_nvidia_smi()
    cuda_ok = bool(torch.cuda.is_available())
    print(f"torch.cuda.is_available()={cuda_ok}", flush=True)
    print(f"torch_version={torch.__version__} device_count={torch.cuda.device_count()}", flush=True)
    if not cuda_ok:
        raise SystemExit("CUDA is not available. Fix the kernel metadata before training.")

    legal = _legal_starts()
    train, test = _load_splits()
    _check_training_video(train, legal)
    device = torch.device("cuda")
    print(f"cuda_device={torch.cuda.get_device_name(0)}", flush=True)
    out = _output_root()
    print(f"output_root={out}", flush=True)

    seed_reports = []
    for seed in SEEDS:
        seed_reports.append(_train_seed(seed, train, device, out, legal))
    del train

    windows = _build_eval_windows(test)
    del test
    evaluated = {}
    for seed in SEEDS:
        evaluated[seed] = _evaluate_seed(seed, windows, device, out)

    metric_keys = ("E5_1x", "C_M", "E_M", "C_R", "E_R", "E2_1x", "E2_2x", "E2_3x")
    per_seed = {}
    for seed in SEEDS:
        values = evaluated[seed]["metrics"]
        block = _metric_block({key: values[key] for key in metric_keys})
        block["horizons"] = evaluated[seed]["horizons"]
        block["inference_seconds"] = evaluated[seed]["inference_seconds"]
        block["mean_epoch_seconds"] = float(
            sum(item["seconds"] for item in seed_reports[seed]["epochs"]) / EPOCHS
        )
        per_seed[f"seed_{seed}"] = block
        seed_path = out / "results" / f"train_c_seed{seed}.json"
        report = json.loads(seed_path.read_text(encoding="utf-8"))
        report["test"] = {
            "t": 1,
            "split": "9000:10000",
            "n_test": 1000,
            "metrics": {key: values[key] for key in metric_keys},
            "horizons": evaluated[seed]["horizons"],
            "inference_seconds": evaluated[seed]["inference_seconds"],
            "mean_epoch_seconds": block["mean_epoch_seconds"],
        }
        _write_text(seed_path, json.dumps(report, indent=2) + "\n")

    mean_values = {
        key: float(per_seed["seed_0"][key] + per_seed["seed_1"][key]) / 2.0 for key in metric_keys
    }
    epoch_seconds = [item["seconds"] for report in seed_reports for item in report["epochs"]]
    inference_seconds = [evaluated[seed]["inference_seconds"] for seed in SEEDS]
    mean_epoch_seconds = float(sum(epoch_seconds) / len(epoch_seconds))
    mean_inference_seconds = float(sum(inference_seconds) / len(inference_seconds))
    table = {
        "model": "C",
        "architecture": "ModelA",
        "params": EXPECTED_PARAMS,
        "symmetry_augmentation": False,
        "gpu_name": gpu_name,
        "cuda_device": torch.cuda.get_device_name(0),
        "cuda": True,
        "torch_version": torch.__version__,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "weight_decay": 0.0,
        "seeds": [0, 1],
        "training": {
            "k": [1, 2, 3],
            "k_draw": "uniform",
            "t_draw": "uniform over legal_starts('C', k)",
            "legal_starts": {str(k): legal[k] for k in SPEEDS},
            "loss": (
                "Squared error averaged over pixels and available target frames "
                "of that video, then over the batch. The model emits 5 frames."
            ),
        },
        "eval": {
            "t": 1,
            "n_test": 1000,
            "split": "9000:10000",
            "forecaster": "original Model C, not a symmetrised forecaster",
            "inference": (
                "Forward time for all 1000 original 1x t=1 test videos, batch 32, "
                "eval, no_grad, started after each batch is on device."
            ),
        },
        "windows": {
            f"{k}x": {
                "inputs": windows[k]["input_frames"],
                "targets": windows[k]["target_frames"],
            }
            for k in SPEEDS
        },
        "reduction": (
            "E(5) and C_T / E_T average squared error over the 64x64 pixels and "
            "the five 1x forecast horizons of each video, then over the 1000 test "
            "videos. C_T = MSE(f(T(X)), T(f(X))). E_T = MSE(f(T(X)), T(Y)). "
            "T is M (mirror, flip width) or R (rot90 k=1). E(2) is the mean of "
            "the first two horizon MSEs. At 3x only those two horizons exist."
        ),
        "per_seed": per_seed,
        "mean": _metric_block(mean_values),
        "mean_epoch_seconds": mean_epoch_seconds,
        "inference_seconds": {
            "seed_0": inference_seconds[0],
            "seed_1": inference_seconds[1],
            "mean": mean_inference_seconds,
        },
    }
    table["mean"]["mean_epoch_seconds"] = mean_epoch_seconds
    table["mean"]["inference_seconds"] = mean_inference_seconds
    _write_text(out / "results" / "table5_c.json", json.dumps(table, indent=2) + "\n")
    print(
        json.dumps(
            {
                "params": EXPECTED_PARAMS,
                "gpu_name": gpu_name,
                "mean": table["mean"]["display"],
                "mean_epoch_seconds": mean_epoch_seconds,
                "inference_seconds": table["inference_seconds"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
