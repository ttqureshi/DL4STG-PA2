"""Train Model B (two seeds) and write the Model B columns of Table 5.

Model B is Model A: the same ConvLSTM, 480,784 parameters, 1x only.
For every training window, one group element g is drawn uniformly from the
eight elements of G, including the identity, and that same g is applied to
the input and the target. There is no speed augmentation. The test split is
not read until both seeds have finished.
"""

from __future__ import annotations

import json
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
# Index i is R^{i % 4} when i < 4, and M R^{i % 4} when i >= 4. Index 0 is R^0.
GROUP = tuple((a, mirrored) for mirrored in (False, True) for a in range(4))


def _ensure_src_on_path() -> None:
    here = Path(__file__).resolve().parent
    candidates = [here, here.parent, here.parent.parent, Path("/kaggle/working"), Path.cwd()]
    for root in candidates:
        src = root / "src"
        if (src / "dl4stg" / "model_a.py").is_file():
            text = str(src)
            if text not in sys.path:
                sys.path.insert(0, text)
            return
    raise RuntimeError("Could not find src/dl4stg/model_a.py")


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


def _print_nvidia_smi() -> str:
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
    return gpu_name


def _count_params(model) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _confirm_params() -> int:
    from dl4stg.model_a import ModelA

    model = ModelA()
    params = _count_params(model)
    print(f"trainable_parameters={params}", flush=True)
    if params != EXPECTED_PARAMS:
        raise SystemExit(
            f"Model B parameter count {params} != {EXPECTED_PARAMS}; aborting."
        )
    del model
    return params


def _windows_from_t(videos, t):
    import torch

    t = t.to(device=videos.device, dtype=torch.long)
    batch = videos.shape[0]
    b = torch.arange(batch, device=videos.device).unsqueeze(1)
    offsets = torch.arange(5, device=videos.device)
    base = (t - 1).unsqueeze(1)
    inputs = videos[b, base + offsets]
    targets = videos[b, base + offsets + 5]
    return inputs, targets


def apply_g_batch(x, g_index):
    """Apply one group element per batch row. g_index values are 0..7."""
    import torch

    from dl4stg.transforms import apply_g

    if g_index.shape[0] != x.shape[0]:
        raise RuntimeError(
            f"Group index length {g_index.shape[0]} != batch {x.shape[0]}."
        )
    g_index = g_index.to(device=x.device, dtype=torch.long)
    out = x.clone()
    filled = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
    for index, (a, mirrored) in enumerate(GROUP):
        mask = g_index == index
        if not bool(mask.any().item()):
            continue
        out[mask] = apply_g(x[mask], a, mirrored)
        filled[mask] = True
    if not bool(filled.all().item()):
        raise RuntimeError("A training window was not assigned a group element.")
    return out


def _batch_loss(pred, targets):
    """Squared error averaged over pixels and available frames, then the batch."""
    n = int(targets.shape[1])
    if n < 1 or pred.shape[0] != targets.shape[0]:
        raise RuntimeError(
            f"Cannot score pred {tuple(pred.shape)} against {tuple(targets.shape)}."
        )
    err = (pred[:, :n] - targets).pow(2)
    per_video = err.mean(dim=(1, 2, 3, 4))
    return per_video.mean()


def _check_group_and_windows() -> None:
    import torch

    from dl4stg.transforms import apply_g
    from dl4stg.windows import legal_starts, make_window

    if len(GROUP) != 8 or len(set(GROUP)) != 8:
        raise SystemExit(f"G must have 8 elements, got {GROUP}.")
    if GROUP[0] != (0, False):
        raise SystemExit("Index 0 of G must be the identity.")
    sample = torch.randn(2, 5, 1, 64, 64)
    if not torch.equal(apply_g(sample, 0, False), sample):
        raise SystemExit("apply_g identity is not a no-op.")

    marker = torch.zeros(1, 64, 64)
    marker[0, 10, 20] = 1.0
    positions = []
    for a, mirrored in GROUP:
        moved = apply_g(marker, a, mirrored)
        coords = torch.nonzero(moved[0] > 0.5)
        if coords.shape[0] != 1:
            raise SystemExit("A group element did not preserve the marker pixel.")
        positions.append(tuple(int(v) for v in coords[0].tolist()))
    if len(set(positions)) != 8:
        raise SystemExit(f"G collapsed a generic marker: {positions}.")

    video = torch.zeros(20, 1, 64, 64)
    for frame_no in range(1, 21):
        video[frame_no - 1] = frame_no
    starts = legal_starts("B", 1)
    if starts != list(range(1, 12)):
        raise SystemExit(f"Model B legal starts are {starts}, expected 1..11.")
    if legal_starts("B", 2) or legal_starts("B", 3):
        raise SystemExit("Model B training is 1x only.")
    for t in starts:
        inputs, targets, in_frames, tg_frames = make_window(video, t, 1)
        if tg_frames != [t + 5, t + 6, t + 7, t + 8, t + 9]:
            raise SystemExit(f"t={t} targets {tg_frames}")
        if targets.shape[0] != 5:
            raise SystemExit(f"t={t} has {targets.shape[0]} targets")
        got_in, got_tg = _windows_from_t(video.unsqueeze(0), torch.tensor([t]))
        if not torch.equal(got_in[0], inputs) or not torch.equal(got_tg[0], targets):
            raise SystemExit(f"batched window mismatch at t={t}")
        if in_frames != [t + i for i in range(5)]:
            raise SystemExit(f"t={t} inputs {in_frames}")
    expected = {
        1: ([1, 2, 3, 4, 5], [6, 7, 8, 9, 10]),
        2: ([1, 3, 5, 7, 9], [11, 13, 15, 17, 19]),
        3: ([1, 4, 7, 10, 13], [16, 19]),
    }
    for k, (want_in, want_tg) in expected.items():
        _inputs, _targets, in_frames, tg_frames = make_window(video, 1, k)
        if in_frames != want_in or tg_frames != want_tg:
            raise SystemExit(f"k={k} frames {in_frames} {tg_frames}")
    g_index = torch.arange(8)
    batch = video[:5].unsqueeze(0).repeat(8, 1, 1, 1, 1)
    augmented = apply_g_batch(batch, g_index)
    for row, index in enumerate(g_index.tolist()):
        a, mirrored = GROUP[index]
        if not torch.equal(augmented[row], apply_g(batch[row], a, mirrored)):
            raise SystemExit(f"per-sample g mismatch at row {row}.")
    print("group and window checks ok", flush=True)


def _check_train_windows(train) -> None:
    import torch

    from dl4stg.windows import make_window

    video = train[0]
    for t in range(1, 12):
        inputs, targets, _in_frames, tg_frames = make_window(video, t, 1)
        got_in, got_tg = _windows_from_t(video.unsqueeze(0), torch.tensor([t]))
        if not torch.equal(got_in[0], inputs) or not torch.equal(got_tg[0], targets):
            raise SystemExit(f"train window mismatch at t={t}, targets {tg_frames}")
        if targets.shape[0] != 5:
            raise SystemExit(f"train t={t} has {targets.shape[0]} targets")
    print("train window check ok: t=1..11, five targets", flush=True)


def _train_seed(seed: int, train, device, out: Path) -> dict:
    import torch
    from torch.utils.data import DataLoader, Dataset

    from dl4stg.model_a import ModelA

    _set_seed(seed)
    model = ModelA().to(device)
    params = _count_params(model)
    print(f"seed={seed} trainable_parameters={params}", flush=True)
    if params != EXPECTED_PARAMS:
        raise SystemExit(
            f"seed {seed} parameter count {params} != {EXPECTED_PARAMS}; aborting."
        )

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
    g_group = torch.Generator()
    g_group.manual_seed(seed)
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
        for videos in loader:
            videos = videos.to(device, non_blocking=True)
            t = torch.randint(1, 12, (videos.shape[0],), generator=g_windows)
            inputs, targets = _windows_from_t(videos, t)
            if int(targets.shape[1]) != 5:
                raise SystemExit("Model B training produced a short target.")
            g_index = torch.randint(0, 8, (videos.shape[0],), generator=g_group)
            inputs = apply_g_batch(inputs, g_index)
            targets = apply_g_batch(targets, g_index)
            pred = model(inputs)
            loss = _batch_loss(pred, targets)
            if epoch == 1 and seen == 0:
                with torch.no_grad():
                    plain = (pred.detach() - targets).pow(2).mean()
                if abs(float(loss.detach()) - float(plain)) > 1e-4:
                    raise SystemExit("Per-video loss disagrees with the full mean.")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            batch = int(inputs.shape[0])
            running += float(loss.detach()) * batch
            seen += batch
        seconds = time.perf_counter() - started
        if seen != train.shape[0]:
            raise SystemExit(
                f"seed {seed} epoch {epoch} saw {seen} videos, expected {train.shape[0]}."
            )
        train_loss = running / seen
        history.append({"epoch": epoch, "seconds": seconds, "train_loss": train_loss})
        print(
            f"seed={seed} epoch={epoch}/{EPOCHS} train_loss={train_loss:.6f} seconds={seconds:.3f}",
            flush=True,
        )

    weights = out / "weights" / f"model_b_seed{seed}.pt"
    weights.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weights)
    report = {
        "seed": seed,
        "params": params,
        "epochs": history,
        "augmentation": "g uniform on G, same g on input and target, 1x only",
    }
    _write_text(
        out / "results" / f"train_b_seed{seed}.json",
        json.dumps(report, indent=2) + "\n",
    )
    print(f"saved {weights}", flush=True)
    del model
    torch.cuda.empty_cache()
    return report


def _load_state(model, path: Path, device) -> None:
    import torch

    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()


def _score_horizons(pred, targets):
    """Return horizon MSEs as a float64 CPU tensor of length n_targets."""
    import torch

    from dl4stg.metrics import mse_per_horizon

    n = int(targets.shape[1])
    horizons = mse_per_horizon(pred, targets).detach().cpu().to(torch.float64)
    manual = (pred[:, :n] - targets).pow(2).mean(dim=(2, 3, 4)).mean(dim=0)
    manual = manual.detach().cpu().to(torch.float64)
    if torch.max(torch.abs(horizons - manual)) > 1e-5:
        raise SystemExit("mse_per_horizon disagrees with the pixel mean.")
    return horizons


def _evaluate(seed: int, test, device, out: Path) -> dict:
    import torch
    from torch.utils.data import DataLoader, Dataset

    from dl4stg.model_a import ModelA
    from dl4stg.transforms import mirror, rotate
    from dl4stg.windows import make_window

    class _Videos(Dataset):
        def __init__(self, videos):
            self.videos = videos

        def __len__(self):
            return int(self.videos.shape[0])

        def __getitem__(self, idx):
            return self.videos[idx]

    model = ModelA().to(device)
    params = _count_params(model)
    if params != EXPECTED_PARAMS:
        raise SystemExit(
            f"eval seed {seed} parameter count {params} != {EXPECTED_PARAMS}; aborting."
        )
    _load_state(model, out / "weights" / f"model_b_seed{seed}.pt", device)

    def apply_T(x, kind: str):
        if kind == "M":
            return mirror(x)
        if kind == "R":
            return rotate(x, k=1)
        raise ValueError(kind)

    loader = DataLoader(_Videos(test), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    sum_h = torch.zeros(5, dtype=torch.float64)
    sum_c = {"M": 0.0, "R": 0.0}
    sum_e = {"M": 0.0, "R": 0.0}
    seen = 0
    seconds = 0.0
    checked = False
    with torch.no_grad():
        for videos in loader:
            videos = videos.to(device, non_blocking=True)
            inputs = videos[:, 0:5]
            targets = videos[:, 5:10]
            reference_in, reference_tg, in_frames, tg_frames = make_window(videos[0].detach().cpu(), 1, 1)
            if in_frames != [1, 2, 3, 4, 5] or tg_frames != [6, 7, 8, 9, 10]:
                raise SystemExit(f"1x t=1 frames {in_frames} {tg_frames}")
            if not torch.equal(reference_in, inputs[0].detach().cpu()) or not torch.equal(
                reference_tg, targets[0].detach().cpu()
            ):
                raise SystemExit("1x t=1 slice disagrees with make_window.")
            torch.cuda.synchronize()
            started = time.perf_counter()
            pred = model(inputs)
            torch.cuda.synchronize()
            seconds += time.perf_counter() - started
            horizons = _score_horizons(pred, targets)
            batch = int(inputs.shape[0])
            sum_h += horizons * batch
            for kind in ("M", "R"):
                transformed_in = apply_T(inputs, kind)
                transformed_y = apply_T(targets, kind)
                transformed_pred = model(transformed_in)
                transformed_fx = apply_T(pred, kind)
                c_h = _score_horizons(transformed_pred, transformed_fx)
                e_h = _score_horizons(transformed_pred, transformed_y)
                sum_c[kind] += float(c_h.mean()) * batch
                sum_e[kind] += float(e_h.mean()) * batch
            seen += batch
            checked = True
    if not checked or seen != 1000:
        raise SystemExit(f"seed {seed} inference saw {seen} test videos, expected 1000.")
    horizons_1x = (sum_h / seen).tolist()
    e5 = float(sum(horizons_1x) / 5.0)
    e2_1x = float(sum(horizons_1x[:2]) / 2.0)

    speed = {1: {"horizons": horizons_1x, "E2": e2_1x}}
    for k, n_targets in ((2, 5), (3, 2)):
        pieces_in = []
        pieces_tg = []
        frames = None
        for i in range(test.shape[0]):
            inp, tg, in_frames, tg_frames = make_window(test[i], 1, k)
            pieces_in.append(inp)
            pieces_tg.append(tg)
            frames = (in_frames, tg_frames)
        inputs = torch.stack(pieces_in, 0)
        targets = torch.stack(pieces_tg, 0)
        if int(targets.shape[1]) != n_targets:
            raise SystemExit(f"{k}x has {targets.shape[1]} targets, expected {n_targets}.")
        print(f"seed={seed} speed={k}x inputs={frames[0]} targets={frames[1]}", flush=True)
        sum_speed = torch.zeros(n_targets, dtype=torch.float64)
        seen_speed = 0
        with torch.no_grad():
            for start in range(0, inputs.shape[0], BATCH_SIZE):
                batch_in = inputs[start : start + BATCH_SIZE].to(device)
                batch_tg = targets[start : start + BATCH_SIZE].to(device)
                pred = model(batch_in)
                sum_speed += _score_horizons(pred, batch_tg) * int(batch_in.shape[0])
                seen_speed += int(batch_in.shape[0])
        if seen_speed != 1000:
            raise SystemExit(f"seed {seed} {k}x saw {seen_speed} videos.")
        got = (sum_speed / seen_speed).tolist()
        if k == 3 and len(got) != 2:
            raise SystemExit("3x scored a horizon count other than 2.")
        speed[k] = {"horizons": got, "E2": float(sum(got[:2]) / 2.0)}

    metrics = {
        "E5": e5,
        "C_M": sum_c["M"] / seen,
        "E_M": sum_e["M"] / seen,
        "C_R": sum_c["R"] / seen,
        "E_R": sum_e["R"] / seen,
        "E2_1x": e2_1x,
        "E2_2x": speed[2]["E2"],
        "E2_3x": speed[3]["E2"],
    }
    print(
        f"seed={seed} E5={three_sig(metrics['E5'])} "
        f"C_M={three_sig(metrics['C_M'])} E_M={three_sig(metrics['E_M'])} "
        f"C_R={three_sig(metrics['C_R'])} E_R={three_sig(metrics['E_R'])} "
        f"E2_1x={three_sig(metrics['E2_1x'])} E2_2x={three_sig(metrics['E2_2x'])} "
        f"E2_3x={three_sig(metrics['E2_3x'])} inference_seconds={seconds:.3f}",
        flush=True,
    )
    del model
    torch.cuda.empty_cache()
    return {
        "metrics": metrics,
        "horizons": {
            "1x": speed[1]["horizons"],
            "2x": speed[2]["horizons"],
            "3x": speed[3]["horizons"],
        },
        "inference_seconds": seconds,
    }


def _load_splits():
    from dl4stg.data import load_moving_mnist

    root = _data_root()
    print(f"loading MovingMNIST split=None from {root}", flush=True)
    started = time.perf_counter()
    train, val, test = load_moving_mnist(root, download=True)
    download_seconds = time.perf_counter() - started
    print(
        f"loaded shapes train={tuple(train.shape)} val={tuple(val.shape)} test={tuple(test.shape)} "
        f"seconds={download_seconds:.1f}",
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
    return train, test, download_seconds


def _mean_pair(a: float, b: float) -> float:
    return (float(a) + float(b)) / 2.0


def main() -> None:
    _ensure_src_on_path()
    import torch

    params = _confirm_params()
    _check_group_and_windows()

    gpu_name = _print_nvidia_smi()
    cuda_ok = bool(torch.cuda.is_available())
    print(f"torch.cuda.is_available()={cuda_ok}", flush=True)
    print(f"torch_version={torch.__version__} device_count={torch.cuda.device_count()}", flush=True)
    if not cuda_ok:
        raise SystemExit("CUDA is not available. Fix the kernel metadata before training.")

    train, test, download_seconds = _load_splits()
    _check_train_windows(train)
    device = torch.device("cuda")
    out = _output_root()
    print(f"output_root={out}", flush=True)

    seed_reports = []
    for seed in SEEDS:
        seed_reports.append(_train_seed(seed, train, device, out))
    del train

    per_seed = {}
    for seed in SEEDS:
        evaluated = _evaluate(seed, test, device, out)
        epoch_seconds = [item["seconds"] for item in seed_reports[seed]["epochs"]]
        per_seed[str(seed)] = {
            **evaluated["metrics"],
            "display": {key: three_sig(val) for key, val in evaluated["metrics"].items()},
            "horizons": evaluated["horizons"],
            "mean_epoch_seconds": float(sum(epoch_seconds) / EPOCHS),
            "inference_seconds": evaluated["inference_seconds"],
        }

    metric_keys = ["E5", "C_M", "E_M", "C_R", "E_R", "E2_1x", "E2_2x", "E2_3x"]
    mean_metrics = {
        key: _mean_pair(per_seed["0"][key], per_seed["1"][key]) for key in metric_keys
    }
    epoch_seconds = [item["seconds"] for report in seed_reports for item in report["epochs"]]
    inference_seconds = [per_seed[str(seed)]["inference_seconds"] for seed in SEEDS]
    mean_epoch_seconds = float(sum(epoch_seconds) / len(epoch_seconds))
    mean_inference = float(sum(inference_seconds) / len(inference_seconds))
    table = {
        "model": "B",
        "params": params,
        "gpu_name": gpu_name,
        "cuda": True,
        "cuda_device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "weight_decay": 0.0,
        "seeds": [0, 1],
        "training": {
            "speed": "1x",
            "legal_t": list(range(1, 12)),
            "augmentation": (
                "For each window, g is drawn uniformly from the 8 elements of G, "
                "including identity. The same g is applied to the input and the target. "
                "No speed augmentation."
            ),
            "group_index": {
                str(index): {"a": a, "mirrored": mirrored}
                for index, (a, mirrored) in enumerate(GROUP)
            },
        },
        "eval": {
            "t": 1,
            "n_test": 1000,
            "split": "9000:10000",
            "inference": "original 1x t=1 only; not the symmetrised 8-transform forecaster",
            "E2_3x": "first two horizons only (frames 16 and 19)",
        },
        "dataset_download_seconds": download_seconds,
        "per_seed": per_seed,
        "mean": {
            **mean_metrics,
            "display": {key: three_sig(val) for key, val in mean_metrics.items()},
            "mean_epoch_seconds": mean_epoch_seconds,
            "inference_seconds": mean_inference,
        },
        "mean_epoch_seconds": mean_epoch_seconds,
        "inference_seconds": {
            "seed_0": per_seed["0"]["inference_seconds"],
            "seed_1": per_seed["1"]["inference_seconds"],
            "mean": mean_inference,
        },
    }
    _write_text(out / "results" / "table5_b.json", json.dumps(table, indent=2) + "\n")
    print(json.dumps({
        "params": params,
        "gpu_name": gpu_name,
        "mean": table["mean"]["display"],
        "mean_epoch_seconds": mean_epoch_seconds,
        "inference_seconds": table["inference_seconds"],
        "dataset_download_seconds_excluded": download_seconds,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
