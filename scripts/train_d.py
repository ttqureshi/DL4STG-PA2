"""Train Model D (two seeds) and write Table 5.

Model D is the dilated ConvLSTM in src/dl4stg/model_d.py. This script does not
change that architecture. Training matches Model C: no symmetry augmentation.
For each video, draw k uniformly from {1, 2, 3}, then t uniformly from Model
D's legal starts at that k. Targets may be fewer than five. The loss scores
only the available targets, and every video contributes equally.

The validation split is not read. The test split is used only after both
seeds have finished, for the reported metrics. Final-epoch weights are kept.
The timed inference pass is the original forecaster, not a symmetrised one.
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


EXPECTED_PARAMS = 480_784
PARAMETER_CAP = 721_176
EPOCHS = 30
BATCH_SIZE = 32
LR = 1e-3
SEEDS = (0, 1)
LEGAL_COUNT = {1: 14, 2: 8, 3: 2}


def _ensure_src_on_path() -> None:
    here = Path(__file__).resolve().parent
    candidates = [here, here.parent, here.parent.parent, Path("/kaggle/working"), Path.cwd()]
    for root in candidates:
        src = root / "src"
        if (src / "dl4stg" / "model_d.py").is_file():
            text = str(src)
            if text not in sys.path:
                sys.path.insert(0, text)
            return
    raise RuntimeError("Could not find src/dl4stg/model_d.py")


def _output_root() -> Path:
    working = Path("/kaggle/working")
    if working.is_dir():
        return working
    here = Path(__file__).resolve().parent
    for root in (here, here.parent, here.parent.parent):
        if (root / "src" / "dl4stg" / "model_d.py").is_file() and (root / "results").is_dir():
            return root
    return Path.cwd()


def _data_root() -> str:
    if Path("/kaggle/working").is_dir():
        return "/tmp/moving_mnist"
    here = Path(__file__).resolve().parent
    for root in (here, here.parent, here.parent.parent):
        if (root / "src" / "dl4stg" / "model_d.py").is_file():
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
    full = subprocess.run(["nvidia-smi"], check=False, capture_output=True, text=True)
    full_text = (full.stdout or "") + (full.stderr or "")
    print(full_text, flush=True)
    if full.returncode != 0:
        raise SystemExit("nvidia-smi failed; refusing to train.")
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        check=False,
        capture_output=True,
        text=True,
    )
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
    if "T4" not in gpu_name:
        raise SystemExit(f"Expected an Nvidia Tesla T4, got {gpu_name}. Refusing to train.")
    return gpu_name, full_text


def _count_params(model) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _abort_unless_params(params: int, where: str) -> None:
    if params != EXPECTED_PARAMS or params > PARAMETER_CAP:
        raise SystemExit(
            f"{where} trainable count {params} is not {EXPECTED_PARAMS} "
            f"or exceeds {PARAMETER_CAP}; aborting."
        )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _draw_kt(batch: int, generator) -> tuple:
    """For each window, draw k then t. t is uniform on that k's legal starts."""
    import torch

    ks = torch.empty(batch, dtype=torch.long)
    ts = torch.empty(batch, dtype=torch.long)
    for i in range(batch):
        k = int(torch.randint(1, 4, (), generator=generator).item())
        n_legal = LEGAL_COUNT[k]
        t = int(torch.randint(1, n_legal + 1, (), generator=generator).item())
        ks[i] = k
        ts[i] = t
    return ks, ts


def _batch_windows(videos, ks, ts):
    """Inputs are always five frames. ``valid`` marks target horizons that exist."""
    import torch

    ks = ks.to(device=videos.device, dtype=torch.long)
    ts = ts.to(device=videos.device, dtype=torch.long)
    step = torch.arange(5, device=videos.device)
    in_frames = ts.unsqueeze(1) + step.unsqueeze(0) * ks.unsqueeze(1)
    tg_frames = ts.unsqueeze(1) + (step + 5).unsqueeze(0) * ks.unsqueeze(1)
    if int(in_frames.min()) < 1 or int(in_frames.max()) > 20:
        raise SystemExit(
            f"Illegal input frames in [{int(in_frames.min())}, {int(in_frames.max())}]."
        )
    valid = tg_frames <= 20
    counts = valid.sum(dim=1)
    if int(counts.min()) < 2:
        raise SystemExit("A window has fewer than 2 targets.")
    prefix = torch.arange(5, device=videos.device).unsqueeze(0) < counts.unsqueeze(1)
    if not torch.equal(valid, prefix):
        raise SystemExit("Target mask is not a prefix of the five horizons.")
    batch_index = torch.arange(videos.shape[0], device=videos.device).unsqueeze(1)
    inputs = videos[batch_index, in_frames - 1]
    target_index = (tg_frames - 1).clamp(0, 19)
    targets = videos[batch_index, target_index]
    return inputs, targets, valid


def _available_loss(pred, targets, valid):
    """Mean over videos of the pixel MSE on that video's available targets."""
    valid_f = valid.to(dtype=pred.dtype)
    per_horizon = (pred - targets).pow(2).mean(dim=(2, 3, 4))
    counts = valid_f.sum(dim=1)
    per_video = (per_horizon * valid_f).sum(dim=1) / counts
    return per_video.mean()


def _assert_architecture(model) -> int:
    params = _count_params(model)
    _abort_unless_params(params, "architecture check")
    children = [name for name, _module in model.named_children()]
    if children != ["unshuffle", "layer1", "layer2", "head", "shuffle"]:
        raise SystemExit(f"Unexpected Model D children {children}; aborting.")
    for layer_name in ("layer1", "layer2"):
        layer = getattr(model, layer_name)
        if layer.dilation != 2:
            raise SystemExit(f"{layer_name} dilation is {layer.dilation}, expected 2.")
        if tuple(layer.conv.padding) != (2, 2):
            raise SystemExit(f"{layer_name} padding is {layer.conv.padding}, expected (2, 2).")
        if tuple(layer.conv.kernel_size) != (3, 3):
            raise SystemExit(f"{layer_name} kernel is {layer.conv.kernel_size}, expected 3x3.")
        if layer.hidden_channels != 64:
            raise SystemExit(f"{layer_name} hidden is {layer.hidden_channels}, expected 64.")
    if model.head.kernel_size != (1, 1):
        raise SystemExit(f"Head kernel is {model.head.kernel_size}, expected 1x1.")
    banned = []
    for name, module in model.named_modules():
        label = f"{name}:{type(module).__name__}"
        if "atten" in label.lower():
            banned.append(label)
    if banned:
        raise SystemExit(f"Attention is not allowed, found {banned}.")
    return params


def _preflight() -> None:
    """Architecture, legal starts, window indexing, and the equal-video loss."""
    import torch

    from dl4stg.model_d import ModelD
    from dl4stg.windows import legal_starts, make_window

    model = ModelD()
    params = _assert_architecture(model)
    print(f"preflight trainable_parameters={params}", flush=True)
    for k, n_legal in LEGAL_COUNT.items():
        starts = legal_starts("D", k)
        expected = list(range(1, n_legal + 1))
        if starts != expected:
            raise SystemExit(f"Model D k={k} legal starts {starts}, expected {expected}.")
    video = torch.arange(20, dtype=torch.float32).view(20, 1, 1, 1).expand(20, 1, 4, 4).contiguous()
    for k, n_legal in LEGAL_COUNT.items():
        for t in range(1, n_legal + 1):
            inputs, targets, in_frames, tg_frames = make_window(video, t, k)
            got_in, got_tg, valid = _batch_windows(
                video.unsqueeze(0),
                torch.tensor([k]),
                torch.tensor([t]),
            )
            n = int(valid[0].sum())
            if n != len(tg_frames) or n < 2 or n > 5:
                raise SystemExit(f"t={t} k={k} scored {n} targets, frames {tg_frames}.")
            if not torch.equal(got_in[0], inputs) or not torch.equal(got_tg[0, :n], targets):
                raise SystemExit(f"batched window mismatch at t={t} k={k}.")
            if in_frames != [t + i * k for i in range(5)]:
                raise SystemExit(f"t={t} k={k} inputs {in_frames}.")
    # Two targets and five targets must contribute the same amount per video.
    pred = torch.zeros(2, 5, 1, 2, 2)
    held = torch.zeros(2, 5, 1, 2, 2)
    pred[1] = 1
    valid = torch.tensor(
        [
            [True, True, True, True, True],
            [True, True, False, False, False],
        ]
    )
    loss = _available_loss(pred, held, valid)
    if abs(float(loss) - 0.5) > 1e-6:
        raise SystemExit(f"equal-video loss is {float(loss)}, expected 0.5.")
    pixel_weighted = (0.0 * 5 + 1.0 * 2) / 7.0
    if abs(float(loss) - pixel_weighted) < 1e-6:
        raise SystemExit("Loss weighted videos by target count.")
    masked_pred = pred.clone()
    masked_tgt = held.clone()
    masked_pred[1, 2:] = 50
    masked_tgt[1, 2:] = -50
    held_out = _available_loss(masked_pred, masked_tgt, valid)
    if abs(float(held_out) - float(loss)) > 1e-6:
        raise SystemExit("Loss used a target horizon that does not exist.")
    with torch.no_grad():
        batched = model(torch.zeros(2, 5, 1, 64, 64))
        single = model(torch.zeros(5, 1, 64, 64))
    if tuple(batched.shape) != (2, 5, 1, 64, 64) or tuple(single.shape) != (5, 1, 64, 64):
        raise SystemExit(f"Unexpected rollout shapes {tuple(batched.shape)} {tuple(single.shape)}.")
    if float(batched.min()) < 0.0 or float(batched.max()) > 1.0:
        raise SystemExit("Model D output left [0, 1].")
    print("preflight ok: dilation 2, padding 2, legal windows, equal-video loss", flush=True)


def _check_train_windows(train) -> None:
    import torch

    from dl4stg.windows import legal_starts, make_window

    video = train[0]
    for k, n_legal in LEGAL_COUNT.items():
        if legal_starts("D", k) != list(range(1, n_legal + 1)):
            raise SystemExit(f"Model D legal starts changed for k={k}.")
        for t in range(1, n_legal + 1):
            inputs, targets, _in_frames, tg_frames = make_window(video, t, k)
            got_in, got_tg, valid = _batch_windows(
                video.unsqueeze(0),
                torch.tensor([k]),
                torch.tensor([t]),
            )
            n = int(valid.sum())
            if n != len(tg_frames) or not torch.equal(got_in[0], inputs):
                raise SystemExit(f"train window mismatch at t={t} k={k}.")
            if not torch.equal(got_tg[0, :n], targets):
                raise SystemExit(f"train targets mismatch at t={t} k={k}.")
    print("window check ok: Model D 1x t=1..14, 2x t=1..8, 3x t=1..2", flush=True)


def _train_seed(seed: int, train, device, out: Path) -> dict:
    import torch
    from torch.utils.data import DataLoader, Dataset

    from dl4stg.model_d import ModelD

    _set_seed(seed)
    model = ModelD().to(device)
    params = _assert_architecture(model)
    print(f"seed={seed} trainable_parameters={params}", flush=True)

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
    )

    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        started = time.perf_counter()
        running = 0.0
        seen = 0
        k_counts = {1: 0, 2: 0, 3: 0}
        short_targets = 0
        for videos in loader:
            videos = videos.to(device, non_blocking=True)
            ks, ts = _draw_kt(videos.shape[0], g_windows)
            inputs, targets, valid = _batch_windows(videos, ks, ts)
            pred = model(inputs)
            loss = _available_loss(pred, targets, valid)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            batch = int(inputs.shape[0])
            running += float(loss.detach()) * batch
            seen += batch
            if epoch == 1:
                for k in (1, 2, 3):
                    k_counts[k] += int((ks == k).sum())
                short_targets += int((valid.sum(dim=1) < 5).sum())
        seconds = time.perf_counter() - started
        if seen != train.shape[0]:
            raise SystemExit(f"seed {seed} epoch {epoch} saw {seen} videos, expected {train.shape[0]}.")
        train_loss = running / seen
        history.append({"epoch": epoch, "seconds": seconds, "train_loss": train_loss})
        print(
            f"seed={seed} epoch={epoch}/{EPOCHS} train_loss={train_loss:.6f} seconds={seconds:.3f}",
            flush=True,
        )
        if epoch == 1:
            print(
                f"seed={seed} epoch=1 k_counts={k_counts} windows_with_fewer_than_5_targets={short_targets}",
                flush=True,
            )
            if min(k_counts.values()) == 0 or short_targets == 0:
                raise SystemExit("Epoch 1 did not sample every speed, or never dropped a target.")

    weights = out / "weights" / f"model_d_seed{seed}.pt"
    weights.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weights)
    report = {
        "seed": seed,
        "params": params,
        "symmetry_augmentation": False,
        "epochs": history,
    }
    _write_text(
        out / "results" / f"train_d_seed{seed}.json",
        json.dumps(report, indent=2) + "\n",
    )
    print(f"saved {weights}", flush=True)
    del model
    torch.cuda.empty_cache()
    return report


def _load_model(seed: int, device, out: Path):
    import torch

    from dl4stg.model_d import ModelD

    model = ModelD().to(device)
    params = _assert_architecture(model)
    weights_path = out / "weights" / f"model_d_seed{seed}.pt"
    try:
        state = torch.load(weights_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, params


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


def _score_fixed(model, inputs, targets, device, time_forward: bool) -> tuple[list[float], float]:
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
            pred = model(batch_in)
            if time_forward:
                torch.cuda.synchronize()
                seconds += time.perf_counter() - started
            horizons = _horizon_means(pred, batch_tg)
            total += horizons * int(batch_in.shape[0])
            seen += int(batch_in.shape[0])
    if seen != inputs.shape[0]:
        raise SystemExit(f"scored {seen} videos, expected {inputs.shape[0]}.")
    return (total / seen).tolist(), seconds


def _score_consistency(model, test, device) -> dict[str, list[float]]:
    """C_T and E_T at 1x, t=1. These forwards are not part of the inference timer."""
    import torch

    from dl4stg.transforms import mirror, rotate

    totals = {key: torch.zeros(5, dtype=torch.float64) for key in ("C_M", "E_M", "C_R", "E_R")}
    seen = 0
    with torch.no_grad():
        for start in range(0, test.shape[0], BATCH_SIZE):
            videos = test[start : start + BATCH_SIZE].to(device, non_blocking=True)
            inputs = videos[:, 0:5]
            targets = videos[:, 5:10]
            pred = model(inputs)
            pred_m = model(mirror(inputs))
            pred_r = model(rotate(inputs, k=1))
            pairs = {
                "C_M": (pred_m, mirror(pred)),
                "E_M": (pred_m, mirror(targets)),
                "C_R": (pred_r, rotate(pred, k=1)),
                "E_R": (pred_r, rotate(targets, k=1)),
            }
            batch = int(inputs.shape[0])
            for key, (left, right) in pairs.items():
                totals[key] += _horizon_means(left, right) * batch
            seen += batch
    if seen != 1000:
        raise SystemExit(f"consistency saw {seen} test videos, expected 1000.")
    return {key: (value / seen).tolist() for key, value in totals.items()}


def _stack_window(videos, t: int, k: int):
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
    import torch

    return torch.stack(inputs, 0), torch.stack(targets, 0), in_frames, tg_frames


def _evaluate(seed: int, test, device, out: Path) -> dict:
    model, params = _load_model(seed, device, out)
    inputs_1x = test[:, 0:5]
    targets_1x = test[:, 5:10]
    horizons_1x, infer_seconds = _score_fixed(model, inputs_1x, targets_1x, device, time_forward=True)
    if len(horizons_1x) != 5:
        raise SystemExit(f"1x t=1 produced {len(horizons_1x)} horizons, expected 5.")
    consistency = _score_consistency(model, test, device)
    speed = {"1x": horizons_1x}
    expected_targets = {2: [11, 13, 15, 17, 19], 3: [16, 19]}
    for k, expected in expected_targets.items():
        inputs, targets, in_frames, tg_frames = _stack_window(test, 1, k)
        if in_frames != [1 + i * k for i in range(5)] or tg_frames != expected:
            raise SystemExit(f"{k}x t=1 frames inputs {in_frames} targets {tg_frames}.")
        horizons, _unused = _score_fixed(model, inputs, targets, device, time_forward=False)
        if len(horizons) != len(expected):
            raise SystemExit(f"{k}x scored {len(horizons)} horizons, frames {tg_frames}.")
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
    print(
        f"seed={seed} E5={three_sig(metrics['E5'])} "
        f"C_M={three_sig(metrics['C_M'])} E_M={three_sig(metrics['E_M'])} "
        f"C_R={three_sig(metrics['C_R'])} E_R={three_sig(metrics['E_R'])} "
        f"E2_1x={three_sig(metrics['E2_1x'])} E2_2x={three_sig(metrics['E2_2x'])} "
        f"E2_3x={three_sig(metrics['E2_3x'])} inference_seconds={infer_seconds:.3f}",
        flush=True,
    )
    import torch

    del model
    torch.cuda.empty_cache()
    return {
        "params": params,
        "metrics": metrics,
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


def _metric_block(metrics: dict) -> dict:
    block = {key: float(value) for key, value in metrics.items()}
    block["display"] = {key: three_sig(value) for key, value in metrics.items()}
    return block


def _mean_metrics(a: dict, b: dict) -> dict:
    return {key: (float(a[key]) + float(b[key])) / 2.0 for key in a}


def _load_splits():
    from dl4stg.data import load_moving_mnist

    root = _data_root()
    print(f"loading MovingMNIST split=None from {root}", flush=True)
    started = time.perf_counter()
    train, val, test = load_moving_mnist(root, download=True)
    load_seconds = time.perf_counter() - started
    print(
        f"loaded shapes train={tuple(train.shape)} val={tuple(val.shape)} test={tuple(test.shape)} "
        f"seconds={load_seconds:.1f}",
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
    return train, test, load_seconds


def _eval_windows() -> dict:
    import torch

    from dl4stg.windows import make_window

    dummy = torch.zeros(20, 1, 64, 64)
    described = {}
    for k in (1, 2, 3):
        _inputs, _targets, in_frames, tg_frames = make_window(dummy, 1, k)
        described[f"{k}x"] = {
            "t": 1,
            "inputs": in_frames,
            "targets": tg_frames,
            "scored_targets": tg_frames[:2] if k == 3 else tg_frames,
        }
    described["3x"]["note"] = (
        "The model emits 5 frames. Only the first two targets exist, and only those are scored."
    )
    return described


def main() -> None:
    _ensure_src_on_path()
    import torch

    gpu_name, _smi = _print_nvidia_smi()
    cuda_ok = bool(torch.cuda.is_available())
    print(f"torch.cuda.is_available()={cuda_ok}", flush=True)
    print(f"torch_version={torch.__version__} device_count={torch.cuda.device_count()}", flush=True)
    if not cuda_ok:
        raise SystemExit("CUDA is not available. Stopping.")
    cuda_device = torch.cuda.get_device_name(0)
    print(f"cuda_device={cuda_device}", flush=True)
    if "T4" not in cuda_device:
        raise SystemExit(f"Expected an Nvidia Tesla T4, got {cuda_device}. Stopping.")

    _preflight()
    train, test, load_seconds = _load_splits()
    _check_train_windows(train)
    device = torch.device("cuda")
    out = _output_root()
    print(f"output_root={out}", flush=True)

    # Both seeds finish before any test metric is computed.
    seed_reports = []
    for seed in SEEDS:
        seed_reports.append(_train_seed(seed, train, device, out))
    del train

    per_seed = {}
    evaluated = {}
    for seed in SEEDS:
        evaluated[seed] = _evaluate(seed, test, device, out)
        per_seed[str(seed)] = {
            "metrics": _metric_block(evaluated[seed]["metrics"]),
            "horizons": evaluated[seed]["horizons"],
            "inference_seconds": evaluated[seed]["inference_seconds"],
            "mean_epoch_seconds": float(
                sum(item["seconds"] for item in seed_reports[seed]["epochs"]) / EPOCHS
            ),
        }

    mean_metrics = _mean_metrics(evaluated[0]["metrics"], evaluated[1]["metrics"])
    epoch_seconds = [item["seconds"] for report in seed_reports for item in report["epochs"]]
    inference_seconds = [per_seed[str(seed)]["inference_seconds"] for seed in SEEDS]
    table = {
        "model": "D",
        "params": EXPECTED_PARAMS,
        "parameter_cap": PARAMETER_CAP,
        "gpu_name": gpu_name,
        "cuda_device": cuda_device,
        "cuda": True,
        "torch_version": torch.__version__,
        "kaggle_kernel": "mtayyabtahir/dl4stg-pa2-train-d",
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "weight_decay": 0.0,
        "optimizer": "Adam",
        "seeds": [0, 1],
        "symmetry_augmentation": False,
        "final_epoch_weights_only": True,
        "train_split": "0:8000",
        "test_split": "9000:10000",
        "validation_used": False,
        "test_used_for_selection": False,
        "sampling": {
            "k": [1, 2, 3],
            "legal_starts": {str(k): list(range(1, n + 1)) for k, n in LEGAL_COUNT.items()},
        },
        "loss": (
            "Squared error averaged over the pixels and the available target frames of each "
            "video, then over the videos. Horizons past frame 20 are left out."
        ),
        "reduction": (
            "E(5), C_T and E_T average squared error over the 64x64 pixels and the five "
            "forecast horizons of each video, then over the 1000 test videos, at 1x with t=1. "
            "C_T = MSE(f(T(X)), T(f(X))). E_T = MSE(f(T(X)), T(Y)). "
            "E(2) averages the first two horizon MSEs. At 3x only those two horizons exist."
        ),
        "inference": (
            "Original Model D, not the symmetrised forecaster. All 1000 test videos at 1x t=1, "
            "batch 32, eval, no_grad. The clock starts after each batch is on device."
        ),
        "dataset_load_seconds_excluded_from_epochs": load_seconds,
        "windows": _eval_windows(),
        "per_seed": per_seed,
        "mean": {
            "metrics": _metric_block(mean_metrics),
            "mean_epoch_seconds": float(sum(epoch_seconds) / len(epoch_seconds)),
            "inference_seconds": float(sum(inference_seconds) / len(inference_seconds)),
        },
        "mean_epoch_seconds": float(sum(epoch_seconds) / len(epoch_seconds)),
        "inference_seconds": {
            "seed_0": per_seed["0"]["inference_seconds"],
            "seed_1": per_seed["1"]["inference_seconds"],
            "mean": float(sum(inference_seconds) / len(inference_seconds)),
        },
    }
    _write_text(out / "results" / "table5_d.json", json.dumps(table, indent=2) + "\n")
    print(
        json.dumps(
            {
                "params": EXPECTED_PARAMS,
                "gpu_name": gpu_name,
                "cuda_device": cuda_device,
                "mean": table["mean"]["metrics"]["display"],
                "seed_0": per_seed["0"]["metrics"]["display"],
                "seed_1": per_seed["1"]["metrics"]["display"],
                "mean_epoch_seconds": table["mean_epoch_seconds"],
                "epoch_seconds_per_seed": {
                    "0": per_seed["0"]["mean_epoch_seconds"],
                    "1": per_seed["1"]["mean_epoch_seconds"],
                },
                "inference_seconds": table["inference_seconds"],
                "dataset_load_seconds_excluded": load_seconds,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    _ensure_src_on_path()
    if "--check" in sys.argv:
        _preflight()
    else:
        main()
