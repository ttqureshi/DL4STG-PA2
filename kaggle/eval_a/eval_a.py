"""Evaluate the trained Model A checkpoints. No training.

Table 3 and Figures 3–4: mirror and rotation at 1x, t=1.
Table 4 and Figure 5: 1x, 2x, and 3x at t=1.
Blank and Copy-last are scored with the same reductions.
Transforms and windows come from src/dl4stg. This script does not
reimplement mirror or rotation.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import torch

EXPECTED_PARAMS = 480784
SEEDS = (0, 1)
TEST_START = 9000
N_TEST = 1000
# Table 2 seed-wise E(5), used only as a sanity check against these weights.
TABLE2_E5 = {0: 0.022503882157802583, 1: 0.021606205317378044}

mirror = rotate = make_window = None
blank_forecast = copy_last_forecast = mse_per_horizon = None
ModelA = load_moving_mnist = None
ROOT: Path | None = None


def locate_root() -> Path:
    here = Path(__file__).resolve().parent
    needed = [
        "src/dl4stg/__init__.py",
        "src/dl4stg/convlstm.py",
        "src/dl4stg/model_a.py",
        "src/dl4stg/transforms.py",
        "src/dl4stg/windows.py",
        "src/dl4stg/metrics.py",
        "src/dl4stg/data.py",
        "weights/model_a_seed0.pt",
        "weights/model_a_seed1.pt",
    ]
    candidates = [here, here.parent.parent, Path("/kaggle/working")]
    seen: set[Path] = set()
    for cand in candidates:
        try:
            cand = cand.resolve()
        except OSError:
            continue
        if cand in seen:
            continue
        seen.add(cand)
        if all((cand / rel).is_file() for rel in needed):
            return cand
    raise SystemExit(
        "Could not find src/dl4stg and both Model A weight files. "
        "Include the source package and the weights in the kernel directory."
    )


def boot() -> None:
    global ROOT
    global mirror, rotate, make_window
    global blank_forecast, copy_last_forecast, mse_per_horizon
    global ModelA, load_moving_mnist
    ROOT = locate_root()
    src = str(ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from dl4stg.data import load_moving_mnist as _load
    from dl4stg.metrics import (
        blank_forecast as _blank,
        copy_last_forecast as _copy,
        mse_per_horizon as _mse,
    )
    from dl4stg.model_a import ModelA as _model
    from dl4stg.transforms import mirror as _mirror, rotate as _rotate
    from dl4stg.windows import make_window as _window

    mirror = _mirror
    rotate = _rotate
    make_window = _window
    blank_forecast = _blank
    copy_last_forecast = _copy
    mse_per_horizon = _mse
    ModelA = _model
    load_moving_mnist = _load
    print(f"root={ROOT}", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def three_sig(value: float) -> str:
    value = float(value)
    if value == 0.0:
        return "0"
    if abs(value) < 1e-3:
        return f"{value:.2e}"
    return f"{value:.3g}"


def apply_T(x: torch.Tensor, kind: str) -> torch.Tensor:
    """Apply mirror or rotation with the shared functions, on every frame."""
    if kind == "M":
        return mirror(x)
    if kind == "R":
        return rotate(x, k=1)
    raise ValueError(f"Unknown transform {kind!r}.")


def print_nvidia_smi() -> str:
    full = subprocess.run(
        ["nvidia-smi"], check=False, capture_output=True, text=True
    )
    text = (full.stdout or "") + (full.stderr or "")
    print(text, flush=True)
    if full.returncode != 0:
        raise SystemExit("nvidia-smi failed; refusing to evaluate.")
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        check=False,
        capture_output=True,
        text=True,
    )
    if query.returncode != 0:
        raise SystemExit("nvidia-smi GPU-name query failed; refusing to evaluate.")
    names = [line.strip() for line in (query.stdout or "").splitlines() if line.strip()]
    if not names:
        raise SystemExit("nvidia-smi reported no GPU; refusing to evaluate.")
    if len(names) == 1:
        gpu_name = names[0]
    else:
        gpu_name = f"{len(names)} x {names[0]}"
    print(f"gpu_name={gpu_name}", flush=True)
    return gpu_name


def require_cuda() -> torch.device:
    ok = bool(torch.cuda.is_available())
    print(f"torch.cuda.is_available()={ok}", flush=True)
    print(
        f"torch_version={torch.__version__} device_count={torch.cuda.device_count()}",
        flush=True,
    )
    if not ok:
        raise SystemExit("CUDA is not available. Refusing to evaluate.")
    device = torch.device("cuda")
    print(f"cuda_device={torch.cuda.get_device_name(0)}", flush=True)
    return device


def stack_windows(videos: torch.Tensor, t: int, k: int):
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


def reduced_mse(pred: torch.Tensor, target: torch.Tensor):
    """E-style reduction, checked against mse_per_horizon.

    Returns the mean of the horizon MSEs, the per-video mean over those
    horizons, and the horizon MSEs themselves. Only target.shape[1]
    horizons are scored.
    """
    n = int(target.shape[1])
    if pred.shape[0] != target.shape[0]:
        raise SystemExit(
            f"Batch mismatch pred {tuple(pred.shape)} target {tuple(target.shape)}."
        )
    ref = mse_per_horizon(pred, target)
    err = (pred[:, :n] - target).pow(2)
    per_horizon_video = err.mean(dim=(2, 3, 4))
    horizons = per_horizon_video.mean(dim=0)
    if torch.max(torch.abs(ref - horizons)).item() > 1e-5:
        raise SystemExit("mse_per_horizon disagrees with the pixel mean.")
    per_video = per_horizon_video.mean(dim=1)
    scalar = float(horizons.mean().item())
    if abs(scalar - float(per_video.mean().item())) > 1e-5:
        raise SystemExit("Horizon mean and per-video mean disagree.")
    return scalar, per_video, [float(v) for v in horizons.tolist()]


def predict_all(model, inputs: torch.Tensor, device: torch.device, batch_size: int = 32):
    model.eval()
    pieces = []
    with torch.no_grad():
        for start in range(0, inputs.shape[0], batch_size):
            batch = inputs[start : start + batch_size].to(device)
            pred = model(batch)
            pieces.append(pred.detach().cpu())
    return torch.cat(pieces, 0)


def load_model(seed: int, device: torch.device):
    model = ModelA().to(device)
    params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(f"seed={seed} trainable_parameters={params}", flush=True)
    if params != EXPECTED_PARAMS:
        raise SystemExit(f"Parameter count {params} != {EXPECTED_PARAMS}.")
    path = ROOT / "weights" / f"model_a_seed{seed}.pt"
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def metric_block(values: dict) -> dict:
    block = {key: float(val) for key, val in values.items()}
    block["display"] = {key: three_sig(val) for key, val in values.items()}
    return block


def e_from_horizons(horizons: list[float]) -> dict:
    out = {"horizons": [float(v) for v in horizons]}
    out["E2"] = float(sum(horizons[:2]) / 2.0)
    out["E5"] = float(sum(horizons) / len(horizons)) if len(horizons) == 5 else None
    return out


def score_transform(fx, ftx, x, y, kind: str):
    if fx.shape[0] != x.shape[0] or y.shape[0] != x.shape[0] or ftx.shape[0] != x.shape[0]:
        raise SystemExit(f"Transform batch mismatch for {kind}.")
    tfx = apply_T(fx, kind)
    ty = apply_T(y, kind)
    c_scalar, c_video, c_horizons = reduced_mse(ftx, tfx)
    e_scalar, _, e_horizons = reduced_mse(ftx, ty)
    return {
        "C": c_scalar,
        "E": e_scalar,
        "C_horizons": c_horizons,
        "E_horizons": e_horizons,
        "per_video_C": c_video,
    }


def _ndimage():
    try:
        from scipy import ndimage
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "scipy"])
        from scipy import ndimage
    return ndimage


def component_report(frame_hw: np.ndarray, min_area: int = 15) -> dict:
    ndimage = _ndimage()
    hw = np.asarray(frame_hw, dtype=np.float64)
    mask = hw > 0.1
    labeled, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.int32))
    comps = []
    for label in range(1, int(count) + 1):
        ys, xs = np.nonzero(labeled == label)
        area = int(ys.size)
        if area < min_area:
            continue
        comps.append(
            {
                "area": area,
                "row": float(ys.mean()),
                "col": float(xs.mean()),
            }
        )
    comps.sort(key=lambda c: (-c["area"], c["row"], c["col"]))
    return {
        "n": len(comps),
        "areas": [c["area"] for c in comps],
        "centroids": [[round(c["row"], 2), round(c["col"], 2)] for c in comps],
        "max": float(hw.max()) if hw.size else 0.0,
        "frac_above_0.1": float(mask.mean()) if hw.size else 0.0,
        "_centroids_raw": [(c["row"], c["col"]) for c in comps],
    }


def displacement(src: dict, dst: dict) -> dict:
    if src["n"] != 2 or dst["n"] != 2:
        return {"matched": False, "src_n": src["n"], "dst_n": dst["n"]}
    s = src["_centroids_raw"]
    d = dst["_centroids_raw"]
    best = None
    for pairing in (((0, 0), (1, 1)), ((0, 1), (1, 0))):
        moves = []
        total = 0.0
        for i, j in pairing:
            dr = d[j][0] - s[i][0]
            dc = d[j][1] - s[i][1]
            total += math.hypot(dr, dc)
            moves.append(
                {
                    "drow": round(dr, 2),
                    "dcol": round(dc, 2),
                    "src": [round(s[i][0], 2), round(s[i][1], 2)],
                    "dst": [round(d[j][0], 2), round(d[j][1], 2)],
                    "distance": round(math.hypot(dr, dc), 2),
                }
            )
        if best is None or total < best[0]:
            best = (total, moves)
    return {
        "matched": True,
        "total_distance": round(best[0], 2),
        "moves": best[1],
    }


def public_components(report: dict) -> dict:
    return {
        "n": report["n"],
        "areas": report["areas"],
        "centroids": report["centroids"],
        "max": round(report["max"], 4),
        "frac_above_0.1": round(report["frac_above_0.1"], 4),
    }


def frame_np(frame: torch.Tensor) -> np.ndarray:
    array = frame.detach().cpu().numpy()
    if array.ndim == 3:
        array = array[0]
    return array


def track_row(row: torch.Tensor) -> dict:
    """row is (10, 1, 64, 64): five inputs then five futures."""
    last = component_report(frame_np(row[4]))
    futures = []
    moves = []
    for h in range(5):
        report = component_report(frame_np(row[5 + h]))
        futures.append(public_components(report))
        moves.append(displacement(last, report))
    return {"last_input": public_components(last), "futures": futures, "from_last_input": moves}


def displayed_true_frame_numbers() -> list[int]:
    dummy = torch.zeros(20, 1, 64, 64)
    frames: set[int] = set()
    for k in (1, 2, 3):
        _, _, in_frames, tg_frames = make_window(dummy, 1, k)
        frames.update(in_frames)
        frames.update(tg_frames[:2])
    return sorted(frames)


def select_figure5(test: torch.Tensor) -> dict:
    """One test video whose displayed true frames do not contain overlapping digits."""
    frame_numbers = displayed_true_frame_numbers()
    rule = (
        "8-connected components of pixels above 0.1, specks with area under 15 ignored. "
        "A displayed true frame is non-overlapping when exactly two components remain and "
        "each has area at least 80. Displayed true frames are the five inputs and the first "
        "two targets at 1x, 2x, and 3x (t=1), from make_window. Among videos that pass every "
        "displayed frame, and whose two digits move a combined 4 pixels from frame 5 to "
        "frame 7, the chosen video maximises the minimum centroid distance. Ties go to the "
        "smaller dataset index."
    )

    def scan(min_area: int, require_motion: bool):
        best = None
        qualified = 0
        for local in range(test.shape[0]):
            video = test[local]
            ok = True
            min_dist = None
            per_frame = []
            for frame_no in frame_numbers:
                report = component_report(frame_np(video[frame_no - 1]), min_area=15)
                # Specks under 15 pixels are already dropped. Require exactly two
                # components, each at least min_area.
                if report["n"] != 2 or report["areas"][0] < min_area or report["areas"][1] < min_area:
                    ok = False
                    break
                dist = math.hypot(
                    report["_centroids_raw"][0][0] - report["_centroids_raw"][1][0],
                    report["_centroids_raw"][0][1] - report["_centroids_raw"][1][1],
                )
                if min_dist is None or dist < min_dist:
                    min_dist = dist
                per_frame.append(
                    {
                        "frame": frame_no,
                        "n_components": 2,
                        "areas": report["areas"],
                        "centroids": report["centroids"],
                        "centroid_distance": round(dist, 2),
                    }
                )
            if not ok:
                continue
            src = component_report(frame_np(video[4]))
            dst = component_report(frame_np(video[6]))
            motion = displacement(src, dst)
            if require_motion and (not motion["matched"] or motion["total_distance"] < 4.0):
                continue
            qualified += 1
            candidate = (min_dist, -local, local, per_frame, motion)
            if best is None or candidate[0] > best[0] or (
                candidate[0] == best[0] and candidate[1] > best[1]
            ):
                best = candidate
        return best, qualified

    best, qualified = scan(80, True)
    used = "area>=80 and combined frame-5-to-7 displacement>=4"
    if best is None:
        best, qualified = scan(50, True)
        used = "relaxed to area>=50 and combined frame-5-to-7 displacement>=4"
    if best is None:
        best, qualified = scan(50, False)
        used = "relaxed to area>=50 with no motion filter"
    if best is None:
        raise SystemExit("No test video passed the Figure 5 non-overlap check.")
    _dist, _neg, local, per_frame, motion = best
    return {
        "dataset_index": TEST_START + local,
        "local_index": local,
        "rule": rule,
        "rule_applied": used,
        "n_qualified": qualified,
        "displayed_true_frames": frame_numbers,
        "min_centroid_distance": round(float(best[0]), 2),
        "frame5_to_frame7": motion,
        "per_frame": per_frame,
    }


def _draw(ax, image, title: str | None = None, ylabel: str | None = None, future: bool = False):
    ax.imshow(np.asarray(image), cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    color = "#9a3412" if future else "#b0b0b0"
    for spine in ax.spines.values():
        spine.set_color(color)
        spine.set_linewidth(0.6 if future else 0.4)
    if title:
        ax.set_title(title, fontsize=7, pad=2)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=7)


def _fig_image(fig, dpi: int):
    import matplotlib.pyplot as plt
    from PIL import Image

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, facecolor="white")
    plt.close(fig)
    buffer.seek(0)
    image = Image.open(buffer).convert("RGB")
    image.load()
    return image


def save_figure3(panels: list[dict], path: Path, caption: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    fig = plt.figure(figsize=(12.8, 20.0))
    outer = GridSpec(
        4, 1, figure=fig, hspace=0.22, left=0.13, right=0.992, top=0.96, bottom=0.11
    )
    fig.suptitle(
        "Figure 3. Seed-0 Model A, largest per-video consistency gaps",
        fontsize=12,
        y=0.985,
    )
    for p, panel in enumerate(panels):
        block = GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[p], height_ratios=[0.14, 0.86], hspace=0.22
        )
        title_ax = fig.add_subplot(block[0, 0])
        title_ax.axis("off")
        title_ax.text(
            0.5,
            0.55,
            panel["title"],
            ha="center",
            va="center",
            fontsize=9,
            transform=title_ax.transAxes,
            clip_on=False,
        )
        inner = GridSpecFromSubplotSpec(
            4, 10, subplot_spec=block[1, 0], wspace=0.04, hspace=0.28
        )
        rows = panel["rows"]
        for r in range(4):
            frames = rows[r].detach().cpu().numpy()
            for c in range(10):
                ax = fig.add_subplot(inner[r, c])
                title = str(c + 1) if r == 0 else None
                ylabel = panel["row_labels"][r] if c == 0 else None
                _draw(ax, frames[c, 0], title=title, ylabel=ylabel, future=c >= 5)
    wrapped_lines = []
    for paragraph in caption.split("\n"):
        wrapped_lines.extend(textwrap.wrap(paragraph, width=155) or [""])
    wrapped = "\n".join(wrapped_lines)
    fig.text(0.5, 0.012, wrapped, ha="center", va="bottom", fontsize=7.5)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)
    print(f"saved {path}", flush=True)


def save_gif(panel: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames = []
    for h in range(5):
        fig, axes = plt.subplots(4, 6, figsize=(8.6, 5.4))
        fig.suptitle(
            f"{panel['gif_title']}  ·  horizon {h + 1}/5  ·  future frame {6 + h}",
            fontsize=10,
        )
        for r in range(4):
            row = panel["rows"][r]
            shown = torch.cat([row[:5], row[5 + h : 6 + h]], dim=0).detach().cpu().numpy()
            for c in range(6):
                title = str(c + 1) if c < 5 else str(6 + h)
                if r != 0:
                    title = None
                ylabel = panel["row_labels"][r] if c == 0 else None
                _draw(axes[r, c], shown[c, 0], title=title, ylabel=ylabel, future=c == 5)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        frames.append(_fig_image(fig, dpi=90))
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=800,
        loop=0,
        disposal=2,
    )
    print(f"saved {path}", flush=True)


def save_figure4(c_m, c_r, marks_m, marks_r, path: Path, caption: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hi = max(max(c_m), max(c_r), 1e-6) * 1.05
    bins = np.linspace(0.0, hi, 30)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))
    specs = (
        (axes[0], c_m, marks_m, r"$C_M$", "#c2410c"),
        (axes[1], c_r, marks_r, r"$C_R$", "#1d4ed8"),
    )
    for ax, data, marks, label, color in specs:
        ax.hist(data, bins=bins, color="#cbd5e1", edgecolor="white", linewidth=0.4)
        ymax = ax.get_ylim()[1]
        for j, (index, value) in enumerate(marks):
            ax.axvline(value, color=color, linewidth=1.2)
            ax.text(
                value,
                ymax * 0.98,
                str(index),
                rotation=90,
                va="top",
                ha="right" if j == 0 else "left",
                color=color,
                fontsize=8,
            )
        ax.set_xlim(0.0, hi)
        ax.set_xlabel(f"per-video {label}")
        ax.set_ylabel("test videos")
        ax.set_title(f"Seed-0 Model A, {label}")
    fig.suptitle(
        "Figure 4. Per-video consistency on the 1,000 test videos",
        fontsize=12,
    )
    wrapped = "\n".join(textwrap.wrap(caption, width=140))
    fig.text(0.5, 0.01, wrapped, ha="center", va="bottom", fontsize=8)
    fig.tight_layout(rect=(0, 0.12, 1, 0.90))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, facecolor="white")
    plt.close(fig)
    print(f"saved {path}", flush=True)


def save_figure5(windows, preds, choice: dict, path: Path, caption: str) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    local = choice["local_index"]
    fig, axes = plt.subplots(9, 5, figsize=(10.6, 8.6))
    motion = {}
    for s, k in enumerate((1, 2, 3)):
        base = s * 3
        inputs = windows[k]["inputs"][local]
        targets = windows[k]["targets"][local]
        pred = preds[k][local]
        in_frames = windows[k]["input_frames"]
        tg_frames = windows[k]["target_frames"][:2]
        for c in range(5):
            _draw(axes[base, c], frame_np(inputs[c]), title=f"in {in_frames[c]}")
        axes[base, 0].set_ylabel(f"{k}x input", fontsize=8)
        for c in range(5):
            if c < 2:
                _draw(axes[base + 1, c], frame_np(pred[c]), title=f"pred {tg_frames[c]}")
                _draw(axes[base + 2, c], frame_np(targets[c]), title=f"true {tg_frames[c]}")
            else:
                axes[base + 1, c].axis("off")
                axes[base + 2, c].axis("off")
        axes[base + 1, 0].set_ylabel("predicted", fontsize=8)
        axes[base + 2, 0].set_ylabel("true", fontsize=8)
        last = component_report(frame_np(inputs[-1]))
        speed_motion = []
        for h in range(2):
            true_r = component_report(frame_np(targets[h]))
            pred_r = component_report(frame_np(pred[h]))
            mse = float((pred[h] - targets[h]).pow(2).mean().item())
            speed_motion.append(
                {
                    "frame": tg_frames[h],
                    "mse": mse,
                    "true": public_components(true_r),
                    "predicted": public_components(pred_r),
                    "true_from_last_input": displacement(last, true_r),
                    "predicted_from_last_input": displacement(last, pred_r),
                }
            )
        motion[f"{k}x"] = {
            "input_frames": list(in_frames),
            "target_frames_scored_in_figure": list(tg_frames),
            "last_input": public_components(last),
            "horizons": speed_motion,
        }
    fig.suptitle(
        f"Figure 5. Seed-0 Model A, test video {choice['dataset_index']}, speeds 1x 2x 3x",
        fontsize=11,
    )
    wrapped = "\n".join(textwrap.wrap(caption, width=140))
    fig.text(0.5, 0.008, wrapped, ha="center", va="bottom", fontsize=7.5)
    fig.tight_layout(rect=(0, 0.14, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, facecolor="white")
    plt.close(fig)
    print(f"saved {path}", flush=True)
    return motion


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"saved {path}", flush=True)


def output_dir() -> Path:
    working = Path("/kaggle/working")
    if working.is_dir():
        return working
    raise SystemExit("/kaggle/working is missing. Refusing to write elsewhere.")


def build_windows(test: torch.Tensor) -> dict:
    expected = {
        1: ([1, 2, 3, 4, 5], [6, 7, 8, 9, 10]),
        2: ([1, 3, 5, 7, 9], [11, 13, 15, 17, 19]),
        3: ([1, 4, 7, 10, 13], [16, 19]),
    }
    windows = {}
    for k, (want_in, want_tg) in expected.items():
        inputs, targets, in_frames, tg_frames = stack_windows(test, 1, k)
        if in_frames != want_in or tg_frames != want_tg:
            raise SystemExit(f"k={k} frames {in_frames} {tg_frames} != {want_in} {want_tg}")
        if k == 3 and targets.shape[1] != 2:
            raise SystemExit("3x window should have exactly two targets.")
        if k != 3 and targets.shape[1] != 5:
            raise SystemExit(f"{k}x window should have five targets.")
        windows[k] = {
            "inputs": inputs,
            "targets": targets,
            "input_frames": in_frames,
            "target_frames": tg_frames,
        }
        print(
            f"window k={k} inputs={in_frames} targets={tg_frames} "
            f"input_shape={tuple(inputs.shape)} target_shape={tuple(targets.shape)}",
            flush=True,
        )
    return windows


def score_baseline(name: str, forecast_fn, windows: dict) -> dict:
    x = windows[1]["inputs"]
    y = windows[1]["targets"]
    fx = forecast_fn(x)
    e5, _, horizons = reduced_mse(fx, y)
    block = {"E5": e5, "horizons_1x": horizons}
    for kind, c_name, e_name in (("M", "C_M", "E_M"), ("R", "C_R", "E_R")):
        tx = apply_T(x, kind)
        ftx = forecast_fn(tx)
        scored = score_transform(fx, ftx, x, y, kind)
        if float(scored["per_video_C"].abs().max().item()) != 0.0:
            raise SystemExit(f"{name} is not exactly consistent under {kind}.")
        block[c_name] = scored["C"]
        block[e_name] = scored["E"]
    speeds = {}
    for k in (1, 2, 3):
        pred = forecast_fn(windows[k]["inputs"])
        _scalar, _per, horiz = reduced_mse(pred, windows[k]["targets"])
        speeds[k] = e_from_horizons(horiz)
        if k == 3 and speeds[k]["E5"] is not None:
            raise SystemExit("3x E(5) should not be scored.")
    block["speeds"] = speeds
    print(
        f"{name} E5={three_sig(block['E5'])} C_M={three_sig(block['C_M'])} "
        f"E_M={three_sig(block['E_M'])} C_R={three_sig(block['C_R'])} "
        f"E_R={three_sig(block['E_R'])}",
        flush=True,
    )
    return block


def main() -> None:
    boot()
    gpu_name = print_nvidia_smi()
    device = require_cuda()
    for seed in SEEDS:
        path = ROOT / "weights" / f"model_a_seed{seed}.pt"
        print(f"weight seed={seed} sha256={sha256_file(path)} bytes={path.stat().st_size}", flush=True)
    print(f"transforms.py sha256={sha256_file(ROOT / 'src' / 'dl4stg' / 'transforms.py')}", flush=True)
    print(f"windows.py sha256={sha256_file(ROOT / 'src' / 'dl4stg' / 'windows.py')}", flush=True)

    out = output_dir()
    data_root = "/tmp/moving_mnist"
    cached = (Path(data_root) / "mnist_test_seq.npy").is_file()
    print(f"moving_mnist_npy_already_present={cached} root={data_root}", flush=True)
    print(f"loading MovingMNIST split=None download={not cached}", flush=True)
    _train, _val, test = load_moving_mnist(data_root, download=not cached)
    if tuple(test.shape) != (N_TEST, 20, 1, 64, 64):
        raise SystemExit(f"Unexpected test shape {tuple(test.shape)}.")
    test = test.contiguous()
    windows = build_windows(test)

    blank = score_baseline("blank", blank_forecast, windows)
    copy_last = score_baseline("copy_last", copy_last_forecast, windows)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA disappeared before the eval loop.")
    print(
        f"CUDA confirmed; starting the eval loop on {torch.cuda.get_device_name(0)}.",
        flush=True,
    )

    per_seed = {}
    kept = None
    for seed in SEEDS:
        model = load_model(seed, device)
        x = windows[1]["inputs"]
        y = windows[1]["targets"]
        fx = predict_all(model, x, device)
        e5, _, horizons = reduced_mse(fx, y)
        delta = abs(e5 - TABLE2_E5[seed])
        print(
            f"seed={seed} E5={e5:.8f} table2={TABLE2_E5[seed]:.8f} abs_delta={delta:.3e}",
            flush=True,
        )
        if delta > 1e-3:
            raise SystemExit(
                f"Seed {seed} E(5) differs from Table 2 by {delta}. Refusing to publish."
            )
        transforms = {}
        ftx_store = {}
        for kind in ("M", "R"):
            tx = apply_T(x, kind)
            ftx = predict_all(model, tx, device)
            transforms[kind] = score_transform(fx, ftx, x, y, kind)
            if seed == 0:
                ftx_store[kind] = ftx
            else:
                del ftx
        speeds = {}
        speed_preds = {1: fx}
        for k in (2, 3):
            pred = predict_all(model, windows[k]["inputs"], device)
            _scalar, _per, horiz = reduced_mse(pred, windows[k]["targets"])
            speeds[k] = e_from_horizons(horiz)
            if seed == 0:
                speed_preds[k] = pred
            else:
                del pred
        speeds[1] = e_from_horizons(horizons)
        per_seed[seed] = {
            "E5": e5,
            "horizons": horizons,
            "M": transforms["M"],
            "R": transforms["R"],
            "speeds": speeds,
        }
        if seed == 0:
            kept = {"fx": fx, "ftx": ftx_store, "speeds": speed_preds}
        else:
            del fx
        del model
        torch.cuda.empty_cache()
        print(
            f"seed={seed} C_M={three_sig(transforms['M']['C'])} "
            f"E_M={three_sig(transforms['M']['E'])} "
            f"C_R={three_sig(transforms['R']['C'])} "
            f"E_R={three_sig(transforms['R']['E'])} "
            f"E2_1x={three_sig(speeds[1]['E2'])} E2_2x={three_sig(speeds[2]['E2'])} "
            f"E2_3x={three_sig(speeds[3]['E2'])} E5_2x={three_sig(speeds[2]['E5'])}",
            flush=True,
        )

    c_m = [float(v) for v in per_seed[0]["M"]["per_video_C"].tolist()]
    c_r = [float(v) for v in per_seed[0]["R"]["per_video_C"].tolist()]
    if len(c_m) != N_TEST or len(c_r) != N_TEST:
        raise SystemExit("Per-video consistency length is not 1000.")
    cm_top = sorted(range(N_TEST), key=lambda i: (-c_m[i], i))[:2]
    cr_top = sorted(range(N_TEST), key=lambda i: (-c_r[i], i))[:2]
    if cm_top[0] == cm_top[1] or cr_top[0] == cr_top[1]:
        raise SystemExit("Top-two selection collapsed to one video.")

    indices = list(range(TEST_START, TEST_START + N_TEST))
    per_video = {
        "seed": 0,
        "t": 1,
        "k": 1,
        "n_test": N_TEST,
        "dataset_indices": indices,
        "reduction": (
            "Per video, mean squared error over the five forecast horizons and "
            "the 64x64 pixels. Dataset C_T is the mean of these 1000 numbers. "
            "Index i is dataset index 9000+i."
        ),
        "C_M": c_m,
        "C_R": c_r,
        "C_M_mean": per_seed[0]["M"]["C"],
        "C_R_mean": per_seed[0]["R"]["C"],
    }
    write_json(out / "results" / "per_video_consistency_seed0.json", per_video)

    panel_specs = [
        ("M", 1, cm_top[0], "figures/gifs/figure3_cm_rank1.gif"),
        ("M", 2, cm_top[1], "figures/gifs/figure3_cm_rank2.gif"),
        ("R", 1, cr_top[0], "figures/gifs/figure3_cr_rank1.gif"),
        ("R", 2, cr_top[1], "figures/gifs/figure3_cr_rank2.gif"),
    ]
    panels = []
    panel_records = []
    x_all = windows[1]["inputs"]
    y_all = windows[1]["targets"]
    for kind, rank, local, gif_rel in panel_specs:
        score = c_m[local] if kind == "M" else c_r[local]
        letter = "M" if kind == "M" else "R"
        word = "mirror" if kind == "M" else "rotation"
        x = x_all[local]
        y = y_all[local]
        fx = kept["fx"][local]
        ftx = kept["ftx"][kind][local]
        tx = apply_T(x, kind)
        ty = apply_T(y, kind)
        tfx = apply_T(fx, kind)
        rows = [
            torch.cat([x, fx], 0),
            torch.cat([tx, ty], 0),
            torch.cat([tx, ftx], 0),
            torch.cat([tx, tfx], 0),
        ]
        labels = [
            "X | f(X)",
            f"{letter}(X) | {letter}(Y)",
            f"{letter}(X) | f({letter}(X))",
            f"{letter}(X) | {letter}(f(X))",
        ]
        dataset_index = TEST_START + local
        c_name = "C_M" if kind == "M" else "C_R"
        title = (
            f"{word}, rank {rank}  ·  dataset index {dataset_index}  ·  "
            f"{c_name} = {three_sig(score)}"
        )
        horizon_c = [float(v) for v in (ftx - tfx).pow(2).mean(dim=(1, 2, 3)).tolist()]
        if abs(sum(horizon_c) / 5.0 - score) > 1e-5:
            raise SystemExit(f"Panel C does not match per-video list at {dataset_index}.")
        panel = {
            "rows": rows,
            "row_labels": labels,
            "title": title,
            "gif_title": f"{word} rank {rank}, video {dataset_index}, {c_name}={three_sig(score)}",
        }
        panels.append(panel)
        save_gif(panel, out / gif_rel)
        names = ["f(X)", "T(Y)", "f(T(X))", "T(f(X))"]
        panel_records.append(
            {
                "rank": rank,
                "transform": kind,
                "dataset_index": dataset_index,
                "C": score,
                "C_display": three_sig(score),
                "gif": gif_rel,
                "horizon_C": horizon_c,
                "tracks": {name: track_row(row) for name, row in zip(names, rows)},
            }
        )

    shared = sorted({TEST_START + i for i in cm_top} & {TEST_START + i for i in cr_top})
    gif_paths = ", ".join(spec[3] for spec in panel_specs)
    caption3 = (
        "Figure 3. Seed-0 Model A at 1x, t=1. The two test videos with the largest "
        f"per-video C_M are dataset indices {TEST_START + cm_top[0]} "
        f"(C_M={three_sig(c_m[cm_top[0]])}) and {TEST_START + cm_top[1]} "
        f"(C_M={three_sig(c_m[cm_top[1]])}). The two with the largest per-video C_R are "
        f"{TEST_START + cr_top[0]} (C_R={three_sig(c_r[cr_top[0]])}) and "
        f"{TEST_START + cr_top[1]} (C_R={three_sig(c_r[cr_top[1]])}). "
        "Each panel is four rows by ten frames: columns 1-5 are the input side and "
        "columns 6-10 are the future side. Row 1 is X | f(X), row 2 is T(X) | T(Y), "
        "row 3 is T(X) | f(T(X)), row 4 is T(X) | T(f(X)). "
        "GIFs keep those four rows and step through the five forecast horizons, "
        "with the five input frames held on the left and the current future frame on the right.\n"
        f"GIF paths: {gif_paths}."
    )
    if shared:
        caption3 += f" Dataset index {shared[0]} is shown twice, once for each transform."
    save_figure3(panels, out / "figures" / "figure3.png", caption3)

    marks_m = [(TEST_START + i, c_m[i]) for i in cm_top]
    marks_r = [(TEST_START + i, c_r[i]) for i in cr_top]
    caption4 = (
        "Figure 4. Histograms of per-video C_M and C_R for seed-0 Model A on the 1,000 "
        "test videos, dataset indices 9000-9999. Vertical lines mark the Figure 3 videos. "
        f"C_M lines: {marks_m[0][0]} ({three_sig(marks_m[0][1])}), "
        f"{marks_m[1][0]} ({three_sig(marks_m[1][1])}). "
        f"C_R lines: {marks_r[0][0]} ({three_sig(marks_r[0][1])}), "
        f"{marks_r[1][0]} ({three_sig(marks_r[1][1])})."
    )
    save_figure4(c_m, c_r, marks_m, marks_r, out / "figures" / "figure4.png", caption4)

    choice = select_figure5(test)
    caption5 = (
        f"Figure 5. Seed-0 Model A on test video {choice['dataset_index']}. "
        "Each speed shows five inputs and the first two predicted and true futures at t=1. "
        "Non-overlap: every displayed true frame has exactly two 8-connected components "
        f"of pixels above 0.1, each of area at least 80 ({choice['rule_applied']}). "
        f"{choice['n_qualified']} videos qualified. Minimum centroid distance "
        f"{choice['min_centroid_distance']} px."
    )
    motion5 = save_figure5(
        windows, kept["speeds"], choice, out / "figures" / "figure5.png", caption5
    )

    def learned_row(seed: int) -> dict:
        item = per_seed[seed]
        values = {
            "E5": item["E5"],
            "C_M": item["M"]["C"],
            "E_M": item["M"]["E"],
            "C_R": item["R"]["C"],
            "E_R": item["R"]["E"],
        }
        return metric_block(values)

    def mean_row() -> dict:
        a = learned_row(0)
        b = learned_row(1)
        values = {key: (a[key] + b[key]) / 2.0 for key in ("E5", "C_M", "E_M", "C_R", "E_R")}
        return metric_block(values)

    def ref_row(block: dict) -> dict:
        return metric_block(
            {key: block[key] for key in ("E5", "C_M", "E_M", "C_R", "E_R")}
        )

    table3 = {
        "model": "A",
        "eval": {"t": 1, "k": 1, "n_test": N_TEST, "split": "9000:10000"},
        "gpu_name": gpu_name,
        "cuda_device": torch.cuda.get_device_name(0),
        "cuda": True,
        "torch_version": torch.__version__,
        "params": EXPECTED_PARAMS,
        "reduction": (
            "Same as E(5): squared error averaged over the 64x64 pixels and the five "
            "forecast horizons of each video, then over the 1000 test videos. "
            "C_T = MSE(f(T(X)), T(f(X))). E_T = MSE(f(T(X)), T(Y)). "
            "T is M (mirror, flip width) or R (rot90 k=1 on the last two dims)."
        ),
        "model_A": {"seed_0": learned_row(0), "seed_1": learned_row(1), "mean": mean_row()},
        "blank": ref_row(blank),
        "copy_last": ref_row(copy_last),
        "per_video_seed0": {
            "path": "results/per_video_consistency_seed0.json",
            "C_M_median": float(np.median(np.asarray(c_m))),
            "C_R_median": float(np.median(np.asarray(c_r))),
            "C_M_max": max(c_m),
            "C_R_max": max(c_r),
        },
        "figure3": {
            "seed": 0,
            "path": "figures/figure3.png",
            "shared_dataset_indices": shared,
            "panels": panel_records,
            "caption": caption3,
        },
        "figure4": {"path": "figures/figure4.png", "caption": caption4},
    }
    write_json(out / "results" / "table3.json", table3)

    def speed_values(seed_speeds: dict) -> dict:
        return {
            "E2_1x": seed_speeds[1]["E2"],
            "E2_2x": seed_speeds[2]["E2"],
            "E2_3x": seed_speeds[3]["E2"],
            "E5_1x": seed_speeds[1]["E5"],
            "E5_2x": seed_speeds[2]["E5"],
        }

    def speed_block(values: dict, horizons: dict | None = None) -> dict:
        block = metric_block(values)
        if horizons is not None:
            block["horizons"] = horizons
        return block

    a_values = {seed: speed_values(per_seed[seed]["speeds"]) for seed in SEEDS}
    mean_values = {
        key: (a_values[0][key] + a_values[1][key]) / 2.0 for key in a_values[0]
    }
    table4 = {
        "model": "A",
        "t": 1,
        "gpu_name": gpu_name,
        "cuda_device": torch.cuda.get_device_name(0),
        "cuda": True,
        "windows": {
            "1x": {
                "inputs": windows[1]["input_frames"],
                "targets": windows[1]["target_frames"],
            },
            "2x": {
                "inputs": windows[2]["input_frames"],
                "targets": windows[2]["target_frames"],
            },
            "3x": {
                "inputs": windows[3]["input_frames"],
                "targets": windows[3]["target_frames"],
                "scored_targets": windows[3]["target_frames"],
                "note": "The model emits 5 frames. Only the first two targets exist, and only those are scored. E(5) is not reported at 3x.",
            },
        },
        "model_A": {
            "seed_0": speed_block(
                a_values[0],
                {f"{k}x": per_seed[0]["speeds"][k]["horizons"] for k in (1, 2, 3)},
            ),
            "seed_1": speed_block(
                a_values[1],
                {f"{k}x": per_seed[1]["speeds"][k]["horizons"] for k in (1, 2, 3)},
            ),
            "mean": speed_block(mean_values),
        },
        "copy_last": speed_block(
            speed_values(copy_last["speeds"]),
            {f"{k}x": copy_last["speeds"][k]["horizons"] for k in (1, 2, 3)},
        ),
        "blank": speed_block(
            speed_values(blank["speeds"]),
            {f"{k}x": blank["speeds"][k]["horizons"] for k in (1, 2, 3)},
        ),
        "figure5": {
            "seed": 0,
            "path": "figures/figure5.png",
            "dataset_index": choice["dataset_index"],
            "overlap_check": choice,
            "motion": motion5,
            "caption": caption5,
        },
    }
    # choice contains local_index only; fine. Drop nothing required.
    write_json(out / "results" / "table4.json", table4)

    summary = {
        "gpu_name": gpu_name,
        "cuda_device": torch.cuda.get_device_name(0),
        "table3_mean": table3["model_A"]["mean"]["display"],
        "table3_seed0": table3["model_A"]["seed_0"]["display"],
        "table3_seed1": table3["model_A"]["seed_1"]["display"],
        "blank": table3["blank"]["display"],
        "copy_last": table3["copy_last"]["display"],
        "figure3": [
            {
                "transform": p["transform"],
                "rank": p["rank"],
                "dataset_index": p["dataset_index"],
                "C": p["C_display"],
                "gif": p["gif"],
            }
            for p in panel_records
        ],
        "table4_mean": table4["model_A"]["mean"]["display"],
        "table4_seed0": table4["model_A"]["seed_0"]["display"],
        "table4_seed1": table4["model_A"]["seed_1"]["display"],
        "table4_copy_last": table4["copy_last"]["display"],
        "figure5_dataset_index": choice["dataset_index"],
    }
    print("EVAL_A_SUMMARY_BEGIN", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print("EVAL_A_SUMMARY_END", flush=True)


def smoke() -> None:
    """CPU checks that do not download MovingMNIST or run the test-set loop."""
    boot()
    x = torch.zeros(1, 1, 64, 64)
    x[0, 0, 10, 20] = 1.0
    mirrored = mirror(x)
    if float(mirrored[0, 0, 10, 43]) != 1.0:
        raise SystemExit("Mirror did not flip width: (10, 20) should land on (10, 43).")
    turned = rotate(x, k=1)
    if float(turned[0, 0, 43, 10]) != 1.0:
        raise SystemExit("Rotation did not match rot90 k=1 on dims (-2, -1).")

    video = torch.zeros(20, 1, 64, 64)
    for frame in range(20):
        video[frame] = (frame + 1) / 20.0
    expected = {
        1: ([1, 2, 3, 4, 5], [6, 7, 8, 9, 10]),
        2: ([1, 3, 5, 7, 9], [11, 13, 15, 17, 19]),
        3: ([1, 4, 7, 10, 13], [16, 19]),
    }
    for k, (want_in, want_tg) in expected.items():
        inputs, targets, in_frames, tg_frames = make_window(video, 1, k)
        if in_frames != want_in or tg_frames != want_tg:
            raise SystemExit(f"smoke window k={k} {in_frames} {tg_frames}")
        if not torch.equal(inputs[:, 0, 0, 0], torch.tensor(want_in, dtype=torch.float32) / 20.0):
            raise SystemExit(f"smoke window values mismatch at k={k}")

    batch = torch.rand(3, 5, 1, 64, 64)
    target = torch.rand(3, 5, 1, 64, 64)
    for kind in ("M", "R"):
        fx = blank_forecast(batch)
        ftx = blank_forecast(apply_T(batch, kind))
        scored = score_transform(fx, ftx, batch, target, kind)
        if float(scored["per_video_C"].abs().max()) != 0.0:
            raise SystemExit(f"smoke blank C_{kind} is not 0")
        fx = copy_last_forecast(batch)
        ftx = copy_last_forecast(apply_T(batch, kind))
        scored = score_transform(fx, ftx, batch, target, kind)
        if float(scored["per_video_C"].abs().max()) != 0.0:
            raise SystemExit(f"smoke copy-last C_{kind} is not 0")

    model = ModelA()
    params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    if params != EXPECTED_PARAMS:
        raise SystemExit(f"smoke params {params}")
    state = torch.load(ROOT / "weights" / "model_a_seed0.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        pred = model(torch.zeros(2, 5, 1, 64, 64))
    if tuple(pred.shape) != (2, 5, 1, 64, 64):
        raise SystemExit(f"smoke forward shape {tuple(pred.shape)}")
    if not torch.isfinite(pred).all() or float(pred.min()) < 0.0 or float(pred.max()) > 1.0:
        raise SystemExit("smoke forward left [0, 1] or was non-finite")

    left = np.zeros((64, 64))
    right = np.zeros((64, 64))
    left[10:20, 10:20] = 1.0
    left[40:50, 40:50] = 1.0
    right[10:20, 14:24] = 1.0
    right[40:50, 44:54] = 1.0
    moved = displacement(component_report(left), component_report(right))
    if not moved["matched"] or any(abs(m["dcol"] - 4.0) > 0.2 for m in moved["moves"]):
        raise SystemExit(f"smoke displacement failed: {moved}")

    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="dl4stg_eval_smoke_"))
    rows = [torch.rand(10, 1, 64, 64) for _ in range(4)]
    panel = {
        "rows": rows,
        "row_labels": ["X | f(X)", "M(X) | M(Y)", "M(X) | f(M(X))", "M(X) | M(f(X))"],
        "title": "mirror, rank 1  ·  dataset index 9000  ·  C_M = 0.0123",
        "gif_title": "mirror rank 1, video 9000, C_M=0.0123",
    }
    save_gif(panel, tmp / "figures" / "gifs" / "figure3_cm_rank1.gif")
    from PIL import Image

    gif = Image.open(tmp / "figures" / "gifs" / "figure3_cm_rank1.gif")
    if getattr(gif, "n_frames", 1) != 5:
        raise SystemExit(f"smoke gif frames {getattr(gif, 'n_frames', 1)}")
    save_figure3([panel, panel, panel, panel], tmp / "figures" / "figure3.png", "caption " * 40)
    save_figure4(
        [0.01, 0.02, 0.05],
        [0.02, 0.03, 0.08],
        [(9123, 0.05), (9456, 0.02)],
        [(9789, 0.08), (9010, 0.03)],
        tmp / "figures" / "figure4.png",
        "caption",
    )
    print(f"smoke ok tmp={tmp}", flush=True)


if __name__ == "__main__":
    if "--smoke" in sys.argv:
        smoke()
    else:
        main()
