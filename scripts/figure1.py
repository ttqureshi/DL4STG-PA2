"""Figure 1: Farneback optical flow on the first 50 MovingMNIST training videos.

Writeup frames are 1-based. Tensor index is writeup frame number minus 1, so the
pair of writeup frames (s, s+1) for s = 1..19 is tensor indices (s-1, s).

Fifty training videos (dataset indices 0..49) and 19 consecutive pairs give
50 * 19 = 950 pairs. Intensities used for the keep-mask are float32 / 255.
Farneback is run on the matching uint8 frames.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.datasets import MovingMNIST

N_VIDEOS = 50
N_FRAMES = 20
N_PAIRS_EXPECTED = N_VIDEOS * (N_FRAMES - 1)  # 950
INTENSITY_MIN = 0.2
MAGNITUDE_MIN = 0.5
DIRECTION_BINS = 18
DIRECTION_WIDTH = 360.0 / DIRECTION_BINS  # 20 degrees
SPEED_BIN_WIDTH = 0.5
SPEED_MAX = 8.0  # 0.5-wide bins on [0.5, 8.0); last bin is [8.0, +inf)


def direction_degrees(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Screen direction: 0 is right, 90 is up, range [0, 360)."""
    theta = np.degrees(np.arctan2(-np.asarray(dy, dtype=np.float64), np.asarray(dx, dtype=np.float64)))
    return np.mod(theta, 360.0)


def _check_direction_convention() -> None:
    checks = (
        (1.0, 0.0, 0.0),
        (0.0, -1.0, 90.0),
        (-1.0, 0.0, 180.0),
        (0.0, 1.0, 270.0),
    )
    for dx, dy, expected in checks:
        got = float(direction_degrees(dx, dy))
        if abs(got - expected) > 1e-6:
            raise RuntimeError(f"direction convention failed for dx={dx}, dy={dy}: {got} != {expected}")
    # Horizontal mirror sends dx -> -dx. Right becomes left; up stays up.
    if abs(float(direction_degrees(-1.0, 0.0)) - 180.0) > 1e-6:
        raise RuntimeError("mirror direction check failed")
    if abs(float(direction_degrees(0.0, -1.0)) - 90.0) > 1e-6:
        raise RuntimeError("mirror-up direction check failed")


def _check_farneback_rightward() -> None:
    """A bright blob shifted right should produce positive dx and small dy."""
    yy, xx = np.mgrid[0:64, 0:64]
    prev_f = np.exp(-((yy - 32.0) ** 2 + (xx - 24.0) ** 2) / (2.0 * 4.0**2))
    next_f = np.exp(-((yy - 32.0) ** 2 + (xx - 27.0) ** 2) / (2.0 * 4.0**2))
    prev_u8 = np.clip(np.round(prev_f * 255.0), 0, 255).astype(np.uint8)
    next_u8 = np.clip(np.round(next_f * 255.0), 0, 255).astype(np.uint8)
    flow = cv2.calcOpticalFlowFarneback(prev_u8, next_u8, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    dx = float(flow[32, 25, 0])
    dy = float(flow[32, 25, 1])
    print(f"farneback sanity (rightward blob): dx={dx:.3f} dy={dy:.3f}", flush=True)
    if not (dx > 1.5 and abs(dy) < 0.8):
        raise RuntimeError(f"Farneback rightward sanity failed: dx={dx}, dy={dy}")


def _as_frame_stack(item: object) -> torch.Tensor:
    if isinstance(item, (tuple, list)):
        item = item[0]
    if not torch.is_tensor(item):
        item = torch.as_tensor(item)
    video = item.detach().cpu()
    if video.ndim == 4 and video.shape[-1] == 1:
        video = video[..., 0]
    if video.ndim == 4 and video.shape[1] == 1:
        video = video[:, 0]
    if video.ndim != 3:
        raise RuntimeError(f"expected a 20-frame video, got shape {tuple(video.shape)}")
    if video.shape[0] != N_FRAMES and video.shape[-1] == N_FRAMES:
        video = video.permute(2, 0, 1)
    if video.shape != (N_FRAMES, 64, 64):
        raise RuntimeError(f"expected shape (20, 64, 64), got {tuple(video.shape)}")
    return video


def _uint8_and_float(frame: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Return uint8 for Farneback and float32 in [0, 1] for the intensity mask.

    uint8 frames stay uint8 for Farneback. The intensity image is that uint8
    cast to float32 and divided by 255. If a frame is already float, Farneback
    receives round(frame * 255) as uint8.
    """
    arr = frame.numpy()
    if arr.dtype == np.uint8:
        u8 = np.ascontiguousarray(arr)
        floats = u8.astype(np.float32) / 255.0
        return u8, floats
    floats = arr.astype(np.float32)
    if float(floats.max()) > 1.5:
        floats = floats / 255.0
    u8 = np.clip(np.round(floats * 255.0), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(u8), np.ascontiguousarray(floats)


def _load_training_videos(root: str) -> list[torch.Tensor]:
    print(f"loading MovingMNIST split=None from {root}", flush=True)
    dataset = MovingMNIST(root=root, split=None, download=True)
    data_shape = tuple(dataset.data.shape) if hasattr(dataset, "data") else None
    print(
        f"dataset length={len(dataset)} data_shape={data_shape} type={type(dataset.data).__name__ if hasattr(dataset, 'data') else 'n/a'}",
        flush=True,
    )
    if len(dataset) != 10000:
        raise RuntimeError(
            f"split=None should expose all 10000 videos, got len={len(dataset)} shape={data_shape}"
        )
    videos = []
    for index in range(N_VIDEOS):
        videos.append(_as_frame_stack(dataset[index]))
    return videos


def _collect_flow(videos: list[torch.Tensor]) -> tuple[np.ndarray, np.ndarray, int]:
    dx_parts: list[np.ndarray] = []
    dy_parts: list[np.ndarray] = []
    n_pairs = 0
    for index, video in enumerate(videos):
        for s in range(1, N_FRAMES):
            # Writeup frame s is tensor index s - 1. Pair is (s, s+1).
            prev_u8, prev_f = _uint8_and_float(video[s - 1])
            next_u8, next_f = _uint8_and_float(video[s])
            flow = cv2.calcOpticalFlowFarneback(
                prev_u8, next_u8, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            dx = flow[:, :, 0]
            dy = flow[:, :, 1]
            magnitude = np.sqrt(dx * dx + dy * dy)
            keep = (prev_f > INTENSITY_MIN) & (next_f > INTENSITY_MIN) & (magnitude > MAGNITUDE_MIN)
            if np.any(keep):
                dx_parts.append(np.asarray(dx[keep], dtype=np.float32))
                dy_parts.append(np.asarray(dy[keep], dtype=np.float32))
            n_pairs += 1
        if (index + 1) % 10 == 0:
            kept = sum(part.size for part in dx_parts)
            print(f"videos {index + 1}/{N_VIDEOS} pairs={n_pairs} kept={kept}", flush=True)
    if n_pairs != N_PAIRS_EXPECTED:
        raise RuntimeError(f"expected {N_PAIRS_EXPECTED} pairs, computed {n_pairs}")
    if not dx_parts:
        raise RuntimeError("no pixels passed the intensity and magnitude masks")
    return np.concatenate(dx_parts), np.concatenate(dy_parts), n_pairs


def _speed_edges() -> np.ndarray:
    finite = np.arange(MAGNITUDE_MIN, SPEED_MAX + 1e-9, SPEED_BIN_WIDTH, dtype=np.float64)
    return np.concatenate([finite, [np.inf]])


def _histogram_fractions(values: np.ndarray, bins: np.ndarray | int, value_range: tuple[float, float] | None = None) -> tuple[np.ndarray, np.ndarray]:
    counts, edges = np.histogram(values, bins=bins, range=value_range)
    total = counts.sum()
    if total != values.size:
        raise RuntimeError(f"histogram dropped values: counts {total} vs {values.size}")
    fractions = counts.astype(np.float64) / float(values.size)
    return fractions, edges.astype(np.float64)


def _three_sig(value: float) -> str:
    magnitude = abs(value)
    if magnitude != 0.0 and magnitude < 1e-3:
        return f"{value:.2e}"
    if magnitude == 0.0:
        return "0"
    decimals = max(0, 2 - math.floor(math.log10(magnitude)))
    return f"{value:.{decimals}f}"


def _json_edges(edges: np.ndarray) -> list[float | str]:
    out: list[float | str] = []
    for edge in edges.tolist():
        if math.isinf(edge):
            out.append("+inf")
        else:
            out.append(float(edge))
    return out


def _plot(direction_fractions: np.ndarray, direction_edges: np.ndarray, speed_fractions: np.ndarray, speed_edges: np.ndarray, median_speed: float, n_kept: int) -> bytes:
    import io

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.7))

    dir_width = float(direction_edges[1] - direction_edges[0])
    axes[0].bar(
        direction_edges[:-1],
        direction_fractions,
        width=dir_width,
        align="edge",
        color="#3C6E9F",
        edgecolor="white",
        linewidth=0.4,
    )
    axes[0].axhline(1.0 / DIRECTION_BINS, color="#B45309", linestyle="--", linewidth=1.4, label="uniform (1/18)")
    axes[0].set_xlim(0, 360)
    axes[0].set_xticks([0, 90, 180, 270, 360])
    axes[0].set_xticklabels(["0°\nright", "90°\nup", "180°\nleft", "270°\ndown", "360°"])
    axes[0].set_xlabel("Direction θ (degrees)")
    axes[0].set_ylabel("Fraction of kept pixels")
    axes[0].set_title("(a) Direction")
    axes[0].legend(frameon=False, loc="upper right")
    dir_top = max(float(direction_fractions.max()), 1.0 / DIRECTION_BINS)
    axes[0].set_ylim(0, dir_top * 1.28)

    # Finite bins are every edge except the final +inf. The overflow fraction
    # is drawn at SPEED_MAX with the same width so the tail stays on the axis.
    finite_left = speed_edges[:-2]
    body = speed_fractions[:-1]
    overflow = float(speed_fractions[-1])
    if len(finite_left) != len(body):
        raise RuntimeError(f"speed plot bins {len(finite_left)} != fractions {len(body)}")
    axes[1].bar(
        finite_left,
        body,
        width=SPEED_BIN_WIDTH,
        align="edge",
        color="#3C6E9F",
        edgecolor="white",
        linewidth=0.4,
    )
    axes[1].bar(
        [SPEED_MAX],
        [overflow],
        width=SPEED_BIN_WIDTH,
        align="edge",
        color="#8FB4D4",
        edgecolor="#3C6E9F",
        hatch="//",
        linewidth=0.6,
        label=f"≥ {SPEED_MAX:.0f}",
    )
    axes[1].axvline(median_speed, color="#9B2335", linewidth=1.5, label=f"median {_three_sig(median_speed)}")
    axes[1].set_xlim(MAGNITUDE_MIN, SPEED_MAX + SPEED_BIN_WIDTH)
    speed_ticks = list(np.arange(1.0, SPEED_MAX, 1.0)) + [SPEED_MAX + SPEED_BIN_WIDTH / 2.0]
    speed_labels = [str(int(tick)) for tick in np.arange(1.0, SPEED_MAX, 1.0)] + [f"≥{SPEED_MAX:.0f}"]
    axes[1].set_xticks(speed_ticks)
    axes[1].set_xticklabels(speed_labels)
    axes[1].set_xlabel("Speed (pixels / frame)")
    axes[1].set_ylabel("Fraction of kept pixels")
    axes[1].set_title("(b) Speed")
    axes[1].legend(frameon=False, loc="upper right")
    speed_top = max(float(speed_fractions.max()), 1e-6)
    axes[1].set_ylim(0, speed_top * 1.30)

    fig.suptitle(
        "Figure 1. Optical flow on the first 50 MovingMNIST training videos",
        fontsize=12,
        y=1.04,
    )
    caption = (
        f"Kept pixels have intensity > {INTENSITY_MIN} in both frames and flow magnitude > {MAGNITUDE_MIN}. "
        f"Median speed = {_three_sig(median_speed)} pixels/frame. "
        f"Pixels kept = {n_kept:,}. "
        f"Panel (a): 18 bins of {DIRECTION_WIDTH:.0f}°. Dashed line is 1/18. "
        f"Panel (b): {SPEED_BIN_WIDTH}-wide bins; hatched bar is the overflow bin ≥ {SPEED_MAX:.0f}."
    )
    fig.text(0.5, -0.08, caption, ha="center", va="top", fontsize=9)
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)
    return buffer.getvalue()


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _write_outputs(png: bytes, report_text: str) -> list[str]:
    written: list[str] = []
    targets: list[tuple[Path, Path]] = []
    here = Path(__file__).resolve()
    for root in here.parents:
        if (root / "figures").is_dir() and (root / "results").is_dir() and (root / "scripts").is_dir():
            targets.append((root / "figures" / "figure1.png", root / "results" / "figure1.json"))
            break
    kaggle_working = Path("/kaggle/working")
    if kaggle_working.is_dir():
        targets.append((kaggle_working / "figure1.png", kaggle_working / "figure1.json"))
    if not targets:
        cwd = Path.cwd()
        targets.append((cwd / "figure1.png", cwd / "figure1.json"))
    seen: set[Path] = set()
    for png_path, json_path in targets:
        if png_path.resolve() in seen:
            continue
        seen.add(png_path.resolve())
        _write_bytes(png_path, png)
        _write_bytes(json_path, report_text.encode("utf-8"))
        written.append(str(png_path))
        written.append(str(json_path))
    return written


def _percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q))


def build_report(dx: np.ndarray, dy: np.ndarray, n_pairs: int) -> tuple[dict, bytes]:
    n_kept = int(dx.size)
    speed = np.sqrt(dx.astype(np.float64) ** 2 + dy.astype(np.float64) ** 2)
    theta = direction_degrees(dx, dy)
    direction_fractions, direction_edges = _histogram_fractions(theta, bins=DIRECTION_BINS, value_range=(0.0, 360.0))
    speed_fractions, speed_edges = _histogram_fractions(speed, bins=_speed_edges())
    if direction_fractions.size != DIRECTION_BINS:
        raise RuntimeError(f"expected {DIRECTION_BINS} direction bins, got {direction_fractions.size}")
    if abs(float(direction_fractions.sum()) - 1.0) > 1e-8:
        raise RuntimeError("direction fractions do not sum to 1")
    if abs(float(speed_fractions.sum()) - 1.0) > 1e-8:
        raise RuntimeError("speed fractions do not sum to 1")
    median_speed = float(np.median(speed))

    mirrored = direction_degrees(-dx, dy)
    rotated = np.mod(theta + 90.0, 360.0)
    mirrored_fractions, _ = _histogram_fractions(mirrored, bins=DIRECTION_BINS, value_range=(0.0, 360.0))
    rotated_fractions, _ = _histogram_fractions(rotated, bins=DIRECTION_BINS, value_range=(0.0, 360.0))

    direction_list = [float(v) for v in direction_fractions.tolist()]
    report = {
        "n_pairs": int(n_pairs),
        "n_pixels_kept": n_kept,
        "median_speed": median_speed,
        "bin_edges": [float(v) for v in direction_edges.tolist()],
        "direction_fractions": direction_list,
        "fraction_of_kept_pixels_in_each_direction_bin": direction_list,
        "speed_histogram": {
            "bin_width": SPEED_BIN_WIDTH,
            "bin_edges": _json_edges(speed_edges),
            "fractions": [float(v) for v in speed_fractions.tolist()],
            "overflow_bin": f">= {SPEED_MAX}",
        },
        "uniform_fraction": 1.0 / DIRECTION_BINS,
        "direction_fractions_if_mirrored": [float(v) for v in mirrored_fractions.tolist()],
        "direction_fractions_if_rotated_90ccw": [float(v) for v in rotated_fractions.tolist()],
        "speed_summary": {
            "min": float(speed.min()),
            "p25": _percentile(speed, 25),
            "median": median_speed,
            "p75": _percentile(speed, 75),
            "p90": _percentile(speed, 90),
            "p99": _percentile(speed, 99),
            "max": float(speed.max()),
        },
        "n_videos": N_VIDEOS,
        "dataset_indices": [0, N_VIDEOS - 1],
        "intensity_threshold": INTENSITY_MIN,
        "magnitude_threshold": MAGNITUDE_MIN,
        "farneback": [0.5, 3, 15, 3, 5, 1.2, 0],
        "gpu_used": bool(torch.cuda.is_available()),
        "torch_version": torch.__version__,
    }
    if report["n_pairs"] != N_PAIRS_EXPECTED:
        raise RuntimeError("n_pairs is not 950")
    png = _plot(direction_fractions, direction_edges, speed_fractions, speed_edges, median_speed, n_kept)
    return report, png


def main() -> None:
    _check_direction_convention()
    _check_farneback_rightward()
    if Path("/kaggle/working").is_dir():
        root = os.environ.get("MOVING_MNIST_ROOT", "/kaggle/working/data")
    else:
        root = os.environ.get("MOVING_MNIST_ROOT", str(Path.cwd() / "data"))
    videos = _load_training_videos(root)
    dx, dy, n_pairs = _collect_flow(videos)
    report, png = build_report(dx, dy, n_pairs)
    text = json.dumps(report, indent=2) + "\n"
    written = _write_outputs(png, text)
    print(text, end="", flush=True)
    print("wrote:", flush=True)
    for path in written:
        print(f"  {path}", flush=True)


if __name__ == "__main__":
    main()
