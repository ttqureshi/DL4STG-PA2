"""Train Model A (two seeds) and write Table 2 plus Figure 2.

Model A is 1x only. Default eval is t=1, k=1: inputs frames 1-5, targets 6-10.
The validation split is not read. The test split is used only after both seeds
have finished, for the reported metrics and Figure 2. Final-epoch weights are
kept. Mirror, rotation, and speed-up evaluations are not run.
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
OVERLAP_AREA = 200


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
    return gpu_name, full_text


def _count_params(model) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def _check_windows(train) -> None:
    import torch

    from dl4stg.windows import legal_starts, make_window

    starts = legal_starts("A", 1)
    if starts != list(range(1, 12)):
        raise SystemExit(f"Model A legal starts are {starts}, expected 1..11.")
    video = train[0]
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
    print("window check ok: Model A t=1..11, all five targets", flush=True)


def _components(frame_hw: np.ndarray, thresh: float = 0.1) -> tuple[list[int], list[tuple[float, float]]]:
    import cv2

    mask = (np.asarray(frame_hw) > thresh).astype(np.uint8)
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, count)]
    centres = [(float(centroids[i, 1]), float(centroids[i, 0])) for i in range(1, count)]
    return areas, centres


def _component_areas(frame_hw: np.ndarray, thresh: float = 0.1) -> list[int]:
    areas, _centres = _components(frame_hw, thresh=thresh)
    return areas


def _centroid_distance(frame_hw: np.ndarray, thresh: float = 0.1) -> float | None:
    _areas, centres = _components(frame_hw, thresh=thresh)
    if len(centres) != 2:
        return None
    (r1, c1), (r2, c2) = centres
    return float(np.hypot(r1 - r2, c1 - c2))


def _frame_record(frame_hw: np.ndarray) -> dict:
    areas = _component_areas(frame_hw)
    return {
        "n_components": len(areas),
        "areas": areas,
        "merged": len(areas) == 1 and (areas[0] >= OVERLAP_AREA if areas else False),
    }


def _ensure_cv2() -> None:
    try:
        import cv2  # noqa: F401
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", "opencv-python-headless"]
        )


def _select_figure_videos(test) -> dict:
    """Pick three test videos after training. One shows merged digits in frames 6-10."""
    _ensure_cv2()
    n = int(test.shape[0])
    single_digit_areas: list[int] = []
    merged_candidates: list[tuple[int, int, int]] = []
    separate: list[tuple[float, int]] = []
    for local in range(n):
        video = test[local, :, 0].numpy()
        for offset, frame in enumerate(video[5:10]):
            areas = _component_areas(frame)
            if len(areas) == 2:
                single_digit_areas.extend(areas)
            elif len(areas) == 1:
                merged_candidates.append((areas[0], 9000 + local, 6 + offset))
        distances = []
        ok = True
        for frame in video[0:10]:
            dist = _centroid_distance(frame)
            if dist is None:
                ok = False
                break
            distances.append(dist)
        if ok:
            separate.append((float(min(distances)), 9000 + local))
    if not single_digit_areas or not merged_candidates:
        raise SystemExit("Could not measure digit components on the test split.")
    digit_hi = float(np.percentile(np.asarray(single_digit_areas, dtype=np.float64), 99))
    area_floor = max(OVERLAP_AREA, int(round(digit_hi + 40)))
    qualified = [item for item in merged_candidates if item[0] >= area_floor]
    pool = qualified or merged_candidates
    overlap_best = max(pool, key=lambda item: (item[0], -item[1], -item[2]))
    print(
        f"overlap area floor={area_floor} digit_p99={digit_hi:.1f} "
        f"qualified={len(qualified)} best_area={overlap_best[0]}",
        flush=True,
    )
    if len(separate) < 2:
        raise SystemExit(f"Need two clearly separate test videos, found {len(separate)}.")
    separate.sort(key=lambda item: (-item[0], item[1]))
    overlap_index = overlap_best[1]
    clear = [index for _, index in separate if index != overlap_index][:2]
    if len(clear) < 2:
        raise SystemExit("Could not find two non-overlapping videos distinct from the overlap example.")
    indices = [clear[0], overlap_index, clear[1]]
    per_video = {}
    for dataset_index in indices:
        local = dataset_index - 9000
        frames = []
        for frame_no in range(1, 11):
            record = _frame_record(test[local, frame_no - 1, 0].numpy())
            record["frame"] = frame_no
            frames.append(record)
        per_video[str(dataset_index)] = frames
    print(
        f"figure2 indices={indices} overlap={overlap_index} "
        f"frame={overlap_best[2]} area={overlap_best[0]}",
        flush=True,
    )
    return {
        "dataset_indices": indices,
        "overlap_dataset_index": overlap_index,
        "overlap_frame": overlap_best[2],
        "overlap_area": overlap_best[0],
        "overlap_area_floor": area_floor,
        "clear_min_centroid_distance": {
            str(index): dist for dist, index in separate if index in clear
        },
        "frames_1_to_10": per_video,
        "overlap_rule": (
            "8-connected components of pixels above 0.1; a future frame overlaps when "
            f"it has exactly one component and that component has area >= {OVERLAP_AREA}."
        ),
    }


def _train_seed(seed: int, train, device, out: Path) -> dict:
    import torch
    from torch.utils.data import DataLoader, Dataset

    from dl4stg.model_a import ModelA

    _set_seed(seed)
    model = ModelA().to(device)
    params = _count_params(model)
    print(f"seed={seed} trainable_parameters={params}", flush=True)
    if params != EXPECTED_PARAMS:
        raise SystemExit(f"seed {seed} parameter count {params} != {EXPECTED_PARAMS}; aborting.")

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
        for videos in loader:
            videos = videos.to(device, non_blocking=True)
            t = torch.randint(1, 12, (videos.shape[0],), generator=g_windows)
            inputs, targets = _windows_from_t(videos, t)
            pred = model(inputs)
            loss = (pred - targets).pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            batch = int(inputs.shape[0])
            running += float(loss.detach()) * batch
            seen += batch
        seconds = time.perf_counter() - started
        if seen != train.shape[0]:
            raise SystemExit(f"seed {seed} epoch {epoch} saw {seen} videos, expected {train.shape[0]}.")
        train_loss = running / seen
        history.append({"epoch": epoch, "seconds": seconds, "train_loss": train_loss})
        print(
            f"seed={seed} epoch={epoch}/{EPOCHS} train_loss={train_loss:.6f} seconds={seconds:.3f}",
            flush=True,
        )

    weights = out / "weights" / f"model_a_seed{seed}.pt"
    weights.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weights)
    report = {"seed": seed, "params": params, "epochs": history}
    _write_text(
        out / "results" / f"train_a_seed{seed}.json",
        json.dumps(report, indent=2) + "\n",
    )
    print(f"saved {weights}", flush=True)
    del model
    torch.cuda.empty_cache()
    return report


def _evaluate(seed: int, test, device, out: Path, figure_meta: dict | None):
    import torch
    from torch.utils.data import DataLoader, Dataset

    from dl4stg.metrics import mse_per_horizon
    from dl4stg.model_a import ModelA

    class _Indexed(Dataset):
        def __init__(self, videos):
            self.videos = videos

        def __len__(self):
            return int(self.videos.shape[0])

        def __getitem__(self, idx):
            return idx, self.videos[idx]

    model = ModelA().to(device)
    params = _count_params(model)
    if params != EXPECTED_PARAMS:
        raise SystemExit(f"eval seed {seed} parameter count {params} != {EXPECTED_PARAMS}; aborting.")
    weights_path = out / "weights" / f"model_a_seed{seed}.pt"
    try:
        state = torch.load(weights_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    wanted = set()
    stored = {}
    if figure_meta is not None and seed == 0:
        wanted = set(figure_meta["dataset_indices"])

    loader = DataLoader(_Indexed(test), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    total = torch.zeros(5, dtype=torch.float64)
    checked = False
    seen = 0
    seconds = 0.0
    with torch.no_grad():
        for local_idx, videos in loader:
            videos = videos.to(device, non_blocking=True)
            inputs = videos[:, 0:5]
            targets = videos[:, 5:10]
            torch.cuda.synchronize()
            started = time.perf_counter()
            pred = model(inputs)
            torch.cuda.synchronize()
            seconds += time.perf_counter() - started
            batch_mse = mse_per_horizon(pred, targets).detach().cpu().to(torch.float64)
            manual = (pred - targets).pow(2).sum(dim=(0, 2, 3, 4)).detach().cpu().to(torch.float64)
            manual = manual / (inputs.shape[0] * 64 * 64)
            if torch.max(torch.abs(batch_mse - manual)) > 1e-5:
                raise SystemExit("mse_per_horizon disagrees with the pixel mean.")
            if not checked:
                checked = True
            total += batch_mse * int(inputs.shape[0])
            seen += int(inputs.shape[0])
            if wanted:
                dataset_idx = (local_idx + 9000).tolist()
                pred_cpu = pred.detach().cpu()
                input_cpu = inputs.detach().cpu()
                target_cpu = targets.detach().cpu()
                for row, dataset_index in enumerate(dataset_idx):
                    if dataset_index in wanted:
                        stored[dataset_index] = {
                            "inputs": input_cpu[row],
                            "pred": pred_cpu[row],
                            "targets": target_cpu[row],
                        }
    if seen != 1000:
        raise SystemExit(f"seed {seed} inference saw {seen} test videos, expected 1000.")
    mse = (total / seen).tolist()
    print(
        f"seed={seed} test_mse={[three_sig(v) for v in mse]} "
        f"E5={three_sig(sum(mse) / 5)} inference_seconds={seconds:.3f}",
        flush=True,
    )
    del model
    torch.cuda.empty_cache()
    return mse, seconds, stored


def _reference_metrics(test) -> tuple[list[float], list[float]]:
    from dl4stg.metrics import blank_forecast, copy_last_forecast, mse_per_horizon

    inputs = test[:, 0:5]
    targets = test[:, 5:10]
    blank = mse_per_horizon(blank_forecast(inputs), targets).tolist()
    copy_last = mse_per_horizon(copy_last_forecast(inputs), targets).tolist()
    return blank, copy_last


def _metric_block(mse: list[float]) -> dict:
    e5 = float(sum(mse) / len(mse))
    block = {f"MSE_{h}": float(mse[h - 1]) for h in range(1, 6)}
    block["E5"] = e5
    block["display"] = {key: three_sig(value) for key, value in block.items()}
    return block


def _mean_metrics(a: list[float], b: list[float]) -> list[float]:
    return [(float(x) + float(y)) / 2.0 for x, y in zip(a, b)]


def _save_figure(path: Path, test, stored: dict, figure_meta: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    indices = figure_meta["dataset_indices"]
    overlap_index = figure_meta["overlap_dataset_index"]
    overlap_frame = figure_meta["overlap_frame"]
    fig, axes = plt.subplots(9, 5, figsize=(8.4, 11.2))
    row_names = ("Input", "Model A", "Ground truth")
    for video_i, dataset_index in enumerate(indices):
        pack = stored[dataset_index]
        rows = (pack["inputs"], pack["pred"], pack["targets"])
        frame_numbers = (list(range(1, 6)), list(range(6, 11)), list(range(6, 11)))
        for row in range(3):
            for col in range(5):
                ax = axes[video_i * 3 + row, col]
                ax.imshow(rows[row][col, 0].numpy(), cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_color("#b8b8b8")
                    spine.set_linewidth(0.4)
                if row == 0 and col == 0:
                    tag = "overlap" if dataset_index == overlap_index else "separate digits"
                    ax.set_ylabel(
                        f"{dataset_index}\n{tag}\n{row_names[row]}",
                        fontsize=8,
                    )
                elif col == 0:
                    ax.set_ylabel(row_names[row], fontsize=8)
                frame_no = frame_numbers[row][col]
                title = f"frame {frame_no}"
                if dataset_index == overlap_index and frame_no == overlap_frame and row == 2:
                    title = f"frame {frame_no} overlap"
                ax.set_title(title, fontsize=8, color="#9b2335" if "overlap" in title else "black")
    caption = (
        f"Figure 2. Model A, seed 0, on MovingMNIST test videos {indices[0]}, "
        f"{indices[1]}, and {indices[2]} (dataset indices). Each block shows input frames 1–5, "
        f"the five predicted frames, and ground-truth frames 6–10. "
        f"Index {overlap_index} has overlapping digits in ground-truth frame {overlap_frame}."
    )
    fig.suptitle("Figure 2. Model A forecasts on three test videos", fontsize=12)
    fig.text(0.5, 0.01, caption, ha="center", va="bottom", fontsize=8, wrap=True)
    fig.tight_layout(rect=(0.02, 0.06, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)
    print(f"saved {path}", flush=True)


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

    gpu_name, _smi = _print_nvidia_smi()
    cuda_ok = bool(torch.cuda.is_available())
    print(f"torch.cuda.is_available()={cuda_ok}", flush=True)
    print(f"torch_version={torch.__version__} device_count={torch.cuda.device_count()}", flush=True)
    if not cuda_ok:
        raise SystemExit("CUDA is not available. Fix the kernel metadata before training.")

    train, test = _load_splits()
    _check_windows(train)
    device = torch.device("cuda")
    out = _output_root()
    print(f"output_root={out}", flush=True)

    # Both seeds finish before any test metric or Figure 2 pass.
    seed_reports = []
    for seed in SEEDS:
        seed_reports.append(_train_seed(seed, train, device, out))
    del train

    figure_meta = _select_figure_videos(test)
    blank, copy_last = _reference_metrics(test)
    per_seed = {}
    stored = {}
    for seed in SEEDS:
        mse, seconds, packs = _evaluate(seed, test, device, out, figure_meta if seed == 0 else None)
        per_seed[str(seed)] = {
            "metrics": _metric_block(mse),
            "inference_seconds": seconds,
            "mean_epoch_seconds": float(
                sum(item["seconds"] for item in seed_reports[seed]["epochs"]) / EPOCHS
            ),
        }
        if seed == 0:
            stored = packs
    if set(stored) != set(figure_meta["dataset_indices"]):
        raise SystemExit(f"figure predictions missing: have {sorted(stored)} want {figure_meta['dataset_indices']}")

    mean_mse = _mean_metrics(
        [per_seed["0"]["metrics"][f"MSE_{h}"] for h in range(1, 6)],
        [per_seed["1"]["metrics"][f"MSE_{h}"] for h in range(1, 6)],
    )
    epoch_seconds = [item["seconds"] for report in seed_reports for item in report["epochs"]]
    inference_seconds = [per_seed[str(seed)]["inference_seconds"] for seed in SEEDS]
    table = {
        "model": "A",
        "params": EXPECTED_PARAMS,
        "gpu_name": gpu_name,
        "cuda": True,
        "torch_version": torch.__version__,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "weight_decay": 0.0,
        "seeds": [0, 1],
        "eval": {"t": 1, "k": 1, "n_test": 1000, "split": "9000:10000"},
        "per_seed": per_seed,
        "mean": {
            "metrics": _metric_block(mean_mse),
            "mean_epoch_seconds": float(sum(epoch_seconds) / len(epoch_seconds)),
            "inference_seconds": float(sum(inference_seconds) / len(inference_seconds)),
        },
        "blank": _metric_block(blank),
        "copy_last": _metric_block(copy_last),
        "mean_epoch_seconds": float(sum(epoch_seconds) / len(epoch_seconds)),
        "inference_seconds": {
            "seed_0": per_seed["0"]["inference_seconds"],
            "seed_1": per_seed["1"]["inference_seconds"],
            "mean": float(sum(inference_seconds) / len(inference_seconds)),
        },
        "figure2": {
            "seed": 0,
            "dataset_indices": figure_meta["dataset_indices"],
            "overlap_dataset_index": figure_meta["overlap_dataset_index"],
            "overlap_frame": figure_meta["overlap_frame"],
            "overlap_area": figure_meta["overlap_area"],
            "path": "figures/figure2.png",
            "frames_1_to_10": figure_meta["frames_1_to_10"],
            "overlap_rule": figure_meta["overlap_rule"],
            "caption": (
                "Figure 2. Model A, seed 0, on MovingMNIST test videos "
                f"{figure_meta['dataset_indices'][0]}, {figure_meta['dataset_indices'][1]}, and "
                f"{figure_meta['dataset_indices'][2]} (dataset indices). Each block shows input "
                "frames 1-5, the five predicted frames, and ground-truth frames 6-10. "
                f"Index {figure_meta['overlap_dataset_index']} has overlapping digits in "
                f"ground-truth frame {figure_meta['overlap_frame']}."
            ),
        },
    }
    _write_text(out / "results" / "table2.json", json.dumps(table, indent=2) + "\n")
    _save_figure(out / "figures" / "figure2.png", test, stored, figure_meta)
    array_path = out / "results" / "figure2_arrays.npz"
    np.savez_compressed(
        array_path,
        indices=np.asarray(figure_meta["dataset_indices"], dtype=np.int64),
        overlap_index=np.asarray([figure_meta["overlap_dataset_index"]], dtype=np.int64),
        overlap_frame=np.asarray([figure_meta["overlap_frame"]], dtype=np.int64),
        **{f"video_{idx}": test[idx - 9000].numpy() for idx in figure_meta["dataset_indices"]},
        **{f"pred_{idx}": stored[idx]["pred"].numpy() for idx in figure_meta["dataset_indices"]},
    )
    print(f"saved {array_path}", flush=True)
    print(json.dumps({
        "params": EXPECTED_PARAMS,
        "gpu_name": gpu_name,
        "mean": table["mean"]["metrics"]["display"],
        "blank": table["blank"]["display"],
        "copy_last": table["copy_last"]["display"],
        "mean_epoch_seconds": table["mean_epoch_seconds"],
        "inference_seconds": table["inference_seconds"],
        "figure2": figure_meta["dataset_indices"],
        "overlap": figure_meta["overlap_dataset_index"],
        "overlap_frame": figure_meta["overlap_frame"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
