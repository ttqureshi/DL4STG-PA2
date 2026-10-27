"""Train D+B (two seeds) and write Table 5.

D+B is Model D: the dilated ConvLSTM in src/dl4stg/model_d.py, dilation 2,
480,784 parameters. This script does not change that architecture and does not
load the weights of B, C, or D.

Each training video contributes one window. For that window, k is drawn
uniformly from {1, 2, 3}, then t uniformly from Model D's legal starts at
that k, then g uniformly from the eight elements of G. The same g is applied
to the input and the target. Windows come from windows.make_window and
windows.legal_starts. Group elements come from transforms.apply_g.

The loss is the per-video mean squared error over pixels and available
targets, then the mean over the batch. The validation split is not read. The
test split is used only after both seeds have finished. Final-epoch weights
are kept. The timed inference pass is the original forecaster, not a
symmetrised wrapper.
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
# Index 0 is the identity R^0. Then R^1..R^3, then M R^0 .. M R^3.
GROUP = tuple((a, mirrored) for mirrored in (False, True) for a in range(4))


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
    total = int(sum(p.numel() for p in model.parameters()))
    trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    if total != trainable:
        raise SystemExit(f"Frozen parameters present: total {total}, trainable {trainable}.")
    return trainable


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


def _legal_starts() -> dict[int, list[int]]:
    from dl4stg.windows import legal_starts

    expected = {1: list(range(1, 15)), 2: list(range(1, 9)), 3: [1, 2]}
    legal = {}
    for k, want in expected.items():
        got = legal_starts("D", k)
        if got != want:
            raise SystemExit(f"Model D legal starts at {k}x are {got}, expected {want}.")
        legal[k] = list(got)
    return legal


def _draw_ktg(batch: int, generator, legal: dict[int, list[int]]) -> tuple[list[int], list[int], list[int]]:
    """For each window, draw k, then t, then g, in that order."""
    import torch

    ks: list[int] = []
    ts: list[int] = []
    gs: list[int] = []
    for _ in range(batch):
        k = int(torch.randint(1, 4, (), generator=generator).item())
        starts = legal[k]
        t = starts[int(torch.randint(0, len(starts), (), generator=generator).item())]
        g = int(torch.randint(0, len(GROUP), (), generator=generator).item())
        ks.append(k)
        ts.append(t)
        gs.append(g)
    return ks, ts, gs


def _prepare_batch(videos, generator, legal: dict[int, list[int]]):
    """One window per video, then the same g on its input and target."""
    import torch

    from dl4stg.transforms import apply_g
    from dl4stg.windows import make_window

    ks, ts, gs = _draw_ktg(int(videos.shape[0]), generator, legal)
    inputs = []
    targets = []
    short = 0
    for i, (k, t, g) in enumerate(zip(ks, ts, gs)):
        raw_in, raw_tg, in_frames, tg_frames = make_window(videos[i], t, k)
        if in_frames != [t + step * k for step in range(5)]:
            raise SystemExit(f"input frames {in_frames} at t={t} k={k}")
        if tg_frames != [t + step * k for step in range(5, 10) if t + step * k <= 20]:
            raise SystemExit(f"target frames {tg_frames} at t={t} k={k}")
        if int(raw_tg.shape[0]) != len(tg_frames) or len(tg_frames) < 2:
            raise SystemExit(f"t={t} k={k} has {len(tg_frames)} targets")
        a, mirrored = GROUP[g]
        inputs.append(apply_g(raw_in, a, mirrored))
        targets.append(apply_g(raw_tg, a, mirrored))
        if len(tg_frames) < 5:
            short += 1
    stacked = torch.stack(inputs, 0)
    if tuple(stacked.shape) != (len(ks), 5, 1, 64, 64) and videos.shape[-1] == 64:
        raise SystemExit(f"augmented inputs {tuple(stacked.shape)}")
    return stacked, targets, ks, gs, short


def _available_loss(pred, targets: list):
    """Mean over videos of the pixel MSE on that video's available targets."""
    import torch

    losses = []
    for i, target in enumerate(targets):
        n = int(target.shape[0])
        if n < 2 or n > int(pred.shape[1]):
            raise RuntimeError(f"expected 2..5 targets, got {n}")
        losses.append((pred[i, :n] - target).pow(2).mean())
    return torch.stack(losses).mean()


def _assert_architecture(model) -> int:
    params = _count_params(model)
    _abort_unless_params(params, "architecture check")
    if type(model).__name__ != "ModelD":
        raise SystemExit(f"Expected ModelD, got {type(model).__name__}.")
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


def _preflight() -> dict[int, list[int]]:
    """Architecture, legal starts, group action, and the equal-video loss."""
    import torch

    from dl4stg.model_d import ModelD
    from dl4stg.transforms import apply_g
    from dl4stg.windows import make_window

    if len(GROUP) != 8 or len(set(GROUP)) != 8:
        raise SystemExit(f"G must have 8 elements, got {GROUP}.")
    if GROUP[0] != (0, False):
        raise SystemExit("Index 0 of G must be the identity.")
    probe = torch.randn(2, 5, 1, 8, 8)
    if not torch.equal(apply_g(probe, 0, False), probe):
        raise SystemExit("apply_g identity is not a no-op.")
    marker = torch.zeros(1, 8, 8)
    marker[0, 1, 3] = 1.0
    positions = []
    for a, mirrored in GROUP:
        moved = apply_g(marker, a, mirrored)
        coords = torch.nonzero(moved[0] > 0.5)
        if coords.shape[0] != 1:
            raise SystemExit("A group element did not preserve the marker pixel.")
        positions.append(tuple(int(v) for v in coords[0].tolist()))
    if len(set(positions)) != 8:
        raise SystemExit(f"G collapsed a generic marker: {positions}.")

    model = ModelD()
    params = _assert_architecture(model)
    print(f"preflight trainable_parameters={params}", flush=True)
    legal = _legal_starts()

    video = torch.arange(20, dtype=torch.float32).view(20, 1, 1, 1).expand(20, 1, 8, 8).contiguous()
    for k, starts in legal.items():
        for t in starts:
            inputs, targets, in_frames, tg_frames = make_window(video, t, k)
            if int(targets.shape[0]) < 2:
                raise SystemExit(f"t={t} k={k} has fewer than 2 targets.")
            if not torch.equal(inputs, video[[f - 1 for f in in_frames]]):
                raise SystemExit(f"make_window inputs mismatch at t={t} k={k}.")
            if not torch.equal(targets, video[[f - 1 for f in tg_frames]]):
                raise SystemExit(f"make_window targets mismatch at t={t} k={k}.")

    batch = video.unsqueeze(0).repeat(4, 1, 1, 1, 1)
    generator = torch.Generator().manual_seed(0)
    got_in, got_tg, ks, gs, _short = _prepare_batch(batch, generator, legal)
    replay = torch.Generator().manual_seed(0)
    for i in range(4):
        k = int(torch.randint(1, 4, (), generator=replay).item())
        starts = legal[k]
        t = starts[int(torch.randint(0, len(starts), (), generator=replay).item())]
        g = int(torch.randint(0, len(GROUP), (), generator=replay).item())
        raw_in, raw_tg, _in_frames, _tg_frames = make_window(batch[i], t, k)
        a, mirrored = GROUP[g]
        if k != ks[i] or g != gs[i]:
            raise SystemExit("k, t, g draw order is not stable.")
        if not torch.equal(got_in[i], apply_g(raw_in, a, mirrored)):
            raise SystemExit(f"input g mismatch at row {i}.")
        if not torch.equal(got_tg[i], apply_g(raw_tg, a, mirrored)):
            raise SystemExit(f"target g mismatch at row {i}.")
        if tuple(apply_g(raw_in, a, mirrored).shape) != tuple(raw_in.shape):
            raise SystemExit("g changed the input shape.")
        if tuple(got_tg[i].shape) != tuple(raw_tg.shape):
            raise SystemExit("g changed the target shape.")
    if set(gs) == {0}:
        raise SystemExit("preflight group draws were only the identity.")

    pred = torch.zeros(2, 5, 1, 2, 2)
    long_target = torch.ones(5, 1, 2, 2)
    short_target = torch.zeros(2, 1, 2, 2)
    loss = _available_loss(pred, [long_target, short_target])
    if abs(float(loss) - 0.5) > 1e-6:
        raise SystemExit(f"equal-video loss is {float(loss)}, expected 0.5.")
    # Five error-1 frames and two error-0 frames. A frame-weighted pool is 5/7.
    pooled = (1.0 * 5 + 0.0 * 2) / 7.0
    if abs(float(loss) - pooled) < 1e-3:
        raise SystemExit("Loss weighted videos by target count.")
    graph = torch.randn(1, 5, 1, 2, 2, requires_grad=True)
    _available_loss(graph, [torch.randn(2, 1, 2, 2)]).backward()
    if graph.grad is None or float(graph.grad[0, :2].abs().sum()) <= 0.0:
        raise SystemExit("Loss did not reach the available frames.")
    if float(graph.grad[0, 2:].abs().sum()) != 0.0:
        raise SystemExit("Loss reached frames past the available targets.")

    with torch.no_grad():
        batched = model(torch.zeros(2, 5, 1, 64, 64))
        single = model(torch.zeros(5, 1, 64, 64))
    if tuple(batched.shape) != (2, 5, 1, 64, 64) or tuple(single.shape) != (5, 1, 64, 64):
        raise SystemExit(f"Unexpected rollout shapes {tuple(batched.shape)} {tuple(single.shape)}.")
    if float(batched.min()) < 0.0 or float(batched.max()) > 1.0:
        raise SystemExit("Model D output left [0, 1].")
    print("preflight ok: dilation 2, Model D legal starts, same g, equal-video loss", flush=True)
    return legal


def _train_seed(seed: int, train, device, out: Path, legal: dict[int, list[int]]) -> dict:
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
    g_aug = torch.Generator()
    g_aug.manual_seed(seed)
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
    g_counts = {index: 0 for index in range(len(GROUP))}
    for epoch in range(1, EPOCHS + 1):
        model.train()
        started = time.perf_counter()
        running = 0.0
        seen = 0
        epoch_k = {1: 0, 2: 0, 3: 0}
        epoch_g = {index: 0 for index in range(len(GROUP))}
        short_targets = 0
        for videos in loader:
            # Window and group act on CPU through make_window and apply_g.
            # Only the five-frame window is moved, once the targets exist.
            inputs, targets, ks, gs, short = _prepare_batch(videos, g_aug, legal)
            inputs = inputs.to(device, non_blocking=True)
            targets = [item.to(device, non_blocking=True) for item in targets]
            pred = model(inputs)
            if tuple(pred.shape) != (inputs.shape[0], 5, 1, 64, 64):
                raise SystemExit(f"model emitted {tuple(pred.shape)}, expected 5 frames")
            loss = _available_loss(pred, targets)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            batch = int(inputs.shape[0])
            running += float(loss.detach()) * batch
            seen += batch
            short_targets += short
            for k, g in zip(ks, gs):
                epoch_k[k] += 1
                epoch_g[g] += 1
        seconds = time.perf_counter() - started
        if seen != train.shape[0]:
            raise SystemExit(f"seed {seed} epoch {epoch} saw {seen} videos, expected {train.shape[0]}.")
        for k in (1, 2, 3):
            k_counts[k] += epoch_k[k]
        for index, count in epoch_g.items():
            g_counts[index] += count
        train_loss = running / seen
        history.append({"epoch": epoch, "seconds": seconds, "train_loss": train_loss})
        print(
            f"seed={seed} epoch={epoch}/{EPOCHS} train_loss={train_loss:.6f} seconds={seconds:.3f}",
            flush=True,
        )
        if epoch == 1:
            print(
                f"seed={seed} epoch=1 k_counts={epoch_k} g_counts={epoch_g} "
                f"windows_with_fewer_than_5_targets={short_targets}",
                flush=True,
            )
            if min(epoch_k.values()) == 0 or short_targets == 0 or min(epoch_g.values()) == 0:
                raise SystemExit(
                    "Epoch 1 missed a speed, a group element, or never dropped a target."
                )

    weights = out / "weights" / f"model_db_seed{seed}.pt"
    weights.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weights)
    report = {
        "seed": seed,
        "model": "D+B",
        "architecture": "ModelD",
        "dilation": 2,
        "params": params,
        "symmetry_augmentation": True,
        "speed_augmentation": True,
        "k_counts": {str(k): k_counts[k] for k in (1, 2, 3)},
        "g_counts": {str(index): g_counts[index] for index in range(len(GROUP))},
        "epochs": history,
    }
    _write_text(out / "results" / f"train_db_seed{seed}.json", json.dumps(report, indent=2) + "\n")
    print(f"saved {weights}", flush=True)
    del model
    torch.cuda.empty_cache()
    return report


def _load_model(seed: int, device, out: Path):
    import torch

    from dl4stg.model_d import ModelD

    model = ModelD().to(device)
    params = _assert_architecture(model)
    weights_path = out / "weights" / f"model_db_seed{seed}.pt"
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
    was_training = model.training
    model.eval()
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
            if type(model).__name__ != "ModelD":
                raise SystemExit("Inference used a wrapper, not ModelD.")
            horizons = _horizon_means(pred, batch_tg)
            total += horizons * int(batch_in.shape[0])
            seen += int(batch_in.shape[0])
    if was_training:
        model.train()
    if seen != inputs.shape[0]:
        raise SystemExit(f"scored {seen} videos, expected {inputs.shape[0]}.")
    return (total / seen).tolist(), seconds


def _score_consistency(model, test, device) -> dict[str, list[float]]:
    """C_T and E_T at 1x, t=1. These forwards are not part of the inference timer."""
    import torch

    from dl4stg.transforms import mirror, rotate

    totals = {key: torch.zeros(5, dtype=torch.float64) for key in ("C_M", "E_M", "C_R", "E_R")}
    seen = 0
    model.eval()
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


def _evaluate(seed: int, test, device, out: Path) -> dict:
    model, params = _load_model(seed, device, out)
    inputs_1x = test[:, 0:5]
    targets_1x = test[:, 5:10]
    check_in, check_tg, in_frames, tg_frames = _stack_window(test[:1], 1, 1)
    if in_frames != [1, 2, 3, 4, 5] or tg_frames != [6, 7, 8, 9, 10]:
        raise SystemExit(f"1x t=1 frames {in_frames} {tg_frames}")
    if not torch_equal(check_in[0], inputs_1x[0]) or not torch_equal(check_tg[0], targets_1x[0]):
        raise SystemExit("1x t=1 slice disagrees with make_window.")
    horizons_1x, infer_seconds = _score_fixed(model, inputs_1x, targets_1x, device, time_forward=True)
    if len(horizons_1x) != 5:
        raise SystemExit(f"1x t=1 produced {len(horizons_1x)} horizons, expected 5.")
    consistency = _score_consistency(model, test, device)
    speed = {"1x": horizons_1x}
    expected_targets = {2: [11, 13, 15, 17, 19], 3: [16, 19]}
    for k, expected in expected_targets.items():
        inputs, targets, speed_in, speed_tg = _stack_window(test, 1, k)
        if speed_in != [1 + i * k for i in range(5)] or speed_tg != expected:
            raise SystemExit(f"{k}x t=1 frames inputs {speed_in} targets {speed_tg}.")
        horizons, _unused = _score_fixed(model, inputs, targets, device, time_forward=False)
        if len(horizons) != len(expected):
            raise SystemExit(f"{k}x scored {len(horizons)} horizons, frames {speed_tg}.")
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


def torch_equal(left, right) -> bool:
    import torch

    return bool(torch.equal(left, right))


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

    legal = _preflight()
    train, test, load_seconds = _load_splits()
    device = torch.device("cuda")
    out = _output_root()
    print(f"output_root={out}", flush=True)

    # Both seeds finish before any test metric is computed.
    seed_reports = []
    for seed in SEEDS:
        seed_reports.append(_train_seed(seed, train, device, out, legal))
    del train

    per_seed = {}
    evaluated = {}
    for seed in SEEDS:
        evaluated[seed] = _evaluate(seed, test, device, out)
        mean_epoch = float(sum(item["seconds"] for item in seed_reports[seed]["epochs"]) / EPOCHS)
        per_seed[str(seed)] = {
            "metrics": _metric_block(evaluated[seed]["metrics"]),
            "horizons": evaluated[seed]["horizons"],
            "inference_seconds": evaluated[seed]["inference_seconds"],
            "mean_epoch_seconds": mean_epoch,
        }
        seed_path = out / "results" / f"train_db_seed{seed}.json"
        report = json.loads(seed_path.read_text(encoding="utf-8"))
        report["mean_epoch_seconds"] = mean_epoch
        report["test"] = {
            "t": 1,
            "split": "9000:10000",
            "n_test": 1000,
            "forecaster": "original Model D, not a symmetrised wrapper",
            "metrics": evaluated[seed]["metrics"],
            "horizons": evaluated[seed]["horizons"],
            "inference_seconds": evaluated[seed]["inference_seconds"],
        }
        _write_text(seed_path, json.dumps(report, indent=2) + "\n")

    mean_metrics = _mean_metrics(evaluated[0]["metrics"], evaluated[1]["metrics"])
    epoch_seconds = [item["seconds"] for report in seed_reports for item in report["epochs"]]
    inference_seconds = [per_seed[str(seed)]["inference_seconds"] for seed in SEEDS]
    table = {
        "model": "D+B",
        "architecture": "ModelD",
        "dilation": 2,
        "params": EXPECTED_PARAMS,
        "parameter_cap": PARAMETER_CAP,
        "gpu_name": gpu_name,
        "cuda_device": cuda_device,
        "cuda": True,
        "torch_version": torch.__version__,
        "kaggle_kernel": "mtayyabtahir/dl4stg-pa2-train-db",
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "weight_decay": 0.0,
        "optimizer": "Adam",
        "seeds": [0, 1],
        "symmetry_augmentation": True,
        "speed_augmentation": True,
        "final_epoch_weights_only": True,
        "train_split": "0:8000",
        "test_split": "9000:10000",
        "validation_used": False,
        "test_used_for_selection": False,
        "sampling": {
            "order": "for each window: k uniform on {1,2,3}, then t uniform on legal_starts('D', k), then g uniform on G",
            "k": [1, 2, 3],
            "legal_starts": {str(k): legal[k] for k in (1, 2, 3)},
            "group": [
                {"index": index, "a": a, "mirrored": mirrored}
                for index, (a, mirrored) in enumerate(GROUP)
            ],
            "same_g_on_input_and_target": True,
        },
        "loss": (
            "Squared error averaged over the pixels and the available target frames of each "
            "video, then over the videos in the batch. Horizons past frame 20 are left out."
        ),
        "reduction": (
            "E(5), C_T and E_T average squared error over the 64x64 pixels and the five "
            "forecast horizons of each video, then over the 1000 test videos, at 1x with t=1. "
            "C_T = MSE(f(T(X)), T(f(X))). E_T = MSE(f(T(X)), T(Y)). "
            "E(2) averages the first two horizon MSEs. At 3x only those two horizons exist."
        ),
        "inference": (
            "Original Model D, not a symmetrised wrapper. All 1000 test videos at 1x t=1, "
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
    _write_text(out / "results" / "table5_db.json", json.dumps(table, indent=2) + "\n")
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
