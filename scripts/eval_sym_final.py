"""Evaluate symmetrised D+B. Does not train and does not write weights.

The underlying network is Model D, loaded from weights/model_db_seed*.pt.
Each checkpoint is wrapped as

    f_bar(X) = (1/8) sum_{g in G} g^{-1}(f(g(X))).

Test indices are 9000-9999 from MovingMNIST with split=None. Training time per
epoch is copied from results/table5_db.json. Figure 6 uses seed 0 on the same
videos as Figures 3 and 5.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import subprocess
import sys
import textwrap
import time
from pathlib import Path

EXPECTED_PARAMS = 480784
BATCH_SIZE = 32
SEEDS = (0, 1)
N_TEST = 1000
TEST_START = 9000
# A consistency gap this large is not a floating-point residual.
GAP_LIMIT = 1e-5
# table5_db.json mean. About 20.72 s. This script does not retime training.
EPOCH_SECONDS_ABOUT = 20.72
FIGURE3_PANELS = (
    ("M", 1, 9347, "figures/gifs/figure6a_cm_rank1.gif"),
    ("M", 2, 9257, "figures/gifs/figure6a_cm_rank2.gif"),
    ("R", 1, 9118, "figures/gifs/figure6a_cr_rank1.gif"),
    ("R", 2, 9493, "figures/gifs/figure6a_cr_rank2.gif"),
)
FIGURE5_DATASET = 9883
WEIGHTS = {0: "model_db_seed0.pt", 1: "model_db_seed1.pt"}


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
    if abs(copied["mean"] - EPOCH_SECONDS_ABOUT) > 0.05:
        raise SystemExit(
            f"{path.name} mean epoch seconds {copied['mean']} is not about {EPOCH_SECONDS_ABOUT}."
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
    """Time only f_bar on the original 1x t=1 clips. Consistency is extra.

    One f_bar call runs all 8 transforms, underlying forwards, inverses, and
    the average. The clock starts after each batch is already on device.
    """
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
    timed_batches = 0
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
            timed_batches += 1
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
            if start % (BATCH_SIZE * 10) == 0:
                print(f"  1x consistency videos={seen}", flush=True)
    if seen != N_TEST:
        raise SystemExit(f"inference saw {seen} test videos, expected {N_TEST}.")
    if timed_batches != math.ceil(N_TEST / BATCH_SIZE):
        raise SystemExit(f"timed {timed_batches} batches, expected {math.ceil(N_TEST / BATCH_SIZE)}.")
    horizons_1x = (sum_h / seen).tolist()
    consistency = {key: (value / seen).tolist() for key, value in totals.items()}
    return horizons_1x, consistency, seconds


def _measure_dispersion(forecaster, test, device) -> dict:
    """Mean squared deviation of the eight restored forecasts from their average.

    Outside the inference clock. A large value is the blur of the group average.
    """
    import torch

    from dl4stg.symmetrise import GROUP
    from dl4stg.transforms import apply_g, invert_g

    underlying = forecaster.model
    total = torch.zeros((), dtype=torch.float64)
    worst = 0.0
    seen = 0
    checked = False
    with torch.no_grad():
        for start in range(0, test.shape[0], BATCH_SIZE):
            inputs = test[start : start + BATCH_SIZE, 0:5].to(device, non_blocking=True)
            members = []
            for a, mirrored in GROUP:
                predicted = underlying(apply_g(inputs, a, mirrored))
                members.append(invert_g(predicted, a, mirrored))
            stacked = torch.stack(members, 0)
            average = stacked.mean(dim=0)
            if not checked:
                wrapped = forecaster(inputs)
                gap = float((wrapped - average).abs().max())
                if gap > 1e-5:
                    raise SystemExit(f"Wrapper output disagrees with the eight-term average by {gap}.")
                checked = True
            per_video = (stacked - average.unsqueeze(0)).pow(2).mean(dim=(0, 2, 3, 4, 5))
            per_video = per_video.detach().to(dtype=torch.float64, device="cpu")
            total += per_video.sum()
            worst = max(worst, float(per_video.max()))
            seen += int(inputs.shape[0])
    if seen != N_TEST or not checked:
        raise SystemExit(f"dispersion saw {seen} videos.")
    return {
        "group_member_mse": float(total / seen),
        "group_member_mse_max_video": worst,
    }


def _count_wrapped_forwards(forecaster, device) -> None:
    import torch

    calls = {"n": 0}

    def _hook(_module, _inputs, _output):
        calls["n"] += 1

    handle = forecaster.model.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            out = forecaster(torch.zeros(2, 5, 1, 64, 64, device=device))
    finally:
        handle.remove()
    if calls["n"] != 8:
        raise SystemExit(f"The timed wrapper ran the underlying model {calls['n']} times, expected 8.")
    if tuple(out.shape) != (2, 5, 1, 64, 64):
        raise SystemExit(f"Wrapper probe shape {tuple(out.shape)}.")
    print("wrapper forwards per call=8", flush=True)


def _evaluate_seed(weight_path: Path, test, device) -> dict:
    import torch

    from dl4stg.model_d import ModelD
    from dl4stg.symmetrise import SymmetrisedForecaster

    underlying = ModelD().to(device)
    params = _count_params(underlying)
    if params != EXPECTED_PARAMS:
        raise SystemExit(f"parameter count {params} != {EXPECTED_PARAMS}.")
    if underlying.layer1.dilation != 2 or underlying.layer2.dilation != 2:
        raise SystemExit(f"{weight_path.name} dilation is not 2.")
    digest = _load_state(underlying, weight_path, device)
    forecaster = SymmetrisedForecaster(underlying).to(device)
    forecaster.eval()
    if _count_params(forecaster) != params:
        raise SystemExit(
            f"wrapper parameter count {_count_params(forecaster)} != {params}."
        )
    if forecaster.training or underlying.training:
        raise SystemExit(f"{weight_path.name} is not in eval mode.")
    _count_wrapped_forwards(forecaster, device)

    horizons_1x, consistency, infer_seconds = _score_original_and_consistency(
        forecaster, test, device
    )
    if len(horizons_1x) != 5:
        raise SystemExit(f"{weight_path.name} 1x t=1 produced {len(horizons_1x)} horizons.")
    dispersion = _measure_dispersion(forecaster, test, device)
    speed = {"1x": horizons_1x}
    expected_targets = {2: [11, 13, 15, 17, 19], 3: [16, 19]}
    for k, expected in expected_targets.items():
        inputs, targets, in_frames, tg_frames = _stack_window(test, 1, k)
        if in_frames != [1 + i * k for i in range(5)] or tg_frames != expected:
            raise SystemExit(f"{k}x t=1 frames inputs {in_frames} targets {tg_frames}.")
        horizons, _unused = _score_batched(
            forecaster, inputs, targets, device, time_forward=False
        )
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
    _check_gaps(weight_path.name, metrics)
    print(
        f"{weight_path.name} params={params} E5={three_sig(metrics['E5'])} "
        f"C_M={sci(metrics['C_M'])} E_M={three_sig(metrics['E_M'])} "
        f"C_R={sci(metrics['C_R'])} E_R={three_sig(metrics['E_R'])} "
        f"E2_1x={three_sig(metrics['E2_1x'])} E2_2x={three_sig(metrics['E2_2x'])} "
        f"E2_3x={three_sig(metrics['E2_3x'])} "
        f"member_mse={sci(dispersion['group_member_mse'])} "
        f"inference_seconds={infer_seconds:.6f}",
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
        "group_member_mse": dispersion["group_member_mse"],
        "group_member_mse_max_video": dispersion["group_member_mse_max_video"],
        "inference_seconds": infer_seconds,
    }


def _local_index(dataset_index: int) -> int:
    if not (TEST_START <= dataset_index < TEST_START + N_TEST):
        raise SystemExit(f"dataset index {dataset_index} is outside 9000-9999.")
    return dataset_index - TEST_START


def _ndimage():
    try:
        from scipy import ndimage
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "scipy"])
        from scipy import ndimage
    return ndimage


def _component_report(frame_hw, min_area: int = 15) -> dict:
    import numpy as np

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
        comps.append({"area": area, "row": float(ys.mean()), "col": float(xs.mean())})
    comps.sort(key=lambda c: (-c["area"], c["row"], c["col"]))
    return {
        "n": len(comps),
        "areas": [c["area"] for c in comps],
        "centroids": [[round(c["row"], 2), round(c["col"], 2)] for c in comps],
        "max": float(hw.max()) if hw.size else 0.0,
        "_centroids_raw": [(c["row"], c["col"]) for c in comps],
    }


def _public_components(report: dict) -> dict:
    return {
        "n": report["n"],
        "areas": report["areas"],
        "centroids": report["centroids"],
        "max": round(report["max"], 4),
    }


def _displacement(src: dict, dst: dict) -> dict:
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
    return {"matched": True, "total_distance": round(best[0], 2), "moves": best[1]}


def _frame_np(frame):
    array = frame.detach().cpu().numpy()
    if array.ndim == 3:
        array = array[0]
    return array


def _track_row(row) -> dict:
    last = _component_report(_frame_np(row[4]))
    futures = []
    moves = []
    for h in range(5):
        report = _component_report(_frame_np(row[5 + h]))
        futures.append(_public_components(report))
        moves.append(_displacement(last, report))
    return {
        "last_input": _public_components(last),
        "futures": futures,
        "from_last_input": moves,
    }


def _draw(ax, image, title: str | None = None, ylabel: str | None = None, future: bool = False):
    import numpy as np

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


def _save_figure6a(panels: list[dict], path: Path, caption: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    fig = plt.figure(figsize=(12.8, 20.0))
    outer = GridSpec(
        4, 1, figure=fig, hspace=0.22, left=0.13, right=0.992, top=0.96, bottom=0.12
    )
    fig.suptitle(
        "Figure 6a. Seed-0 symmetrised D+B on the Figure 3 videos",
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
        for r in range(4):
            frames = panels[p]["rows"][r].detach().cpu().numpy()
            for c in range(10):
                ax = fig.add_subplot(inner[r, c])
                title = str(c + 1) if r == 0 else None
                ylabel = panel["row_labels"][r] if c == 0 else None
                _draw(ax, frames[c, 0], title=title, ylabel=ylabel, future=c >= 5)
    wrapped_lines = []
    for paragraph in caption.split("\n"):
        wrapped_lines.extend(textwrap.wrap(paragraph, width=155) or [""])
    fig.text(0.5, 0.012, "\n".join(wrapped_lines), ha="center", va="bottom", fontsize=7.5)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)
    print(f"saved {path}", flush=True)


def _save_gif(panel: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

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
    from PIL import Image

    with Image.open(path) as gif:
        n_frames = int(getattr(gif, "n_frames", 1))
    if n_frames != 5:
        raise SystemExit(f"{path} has {n_frames} frames, expected 5.")
    print(f"saved {path}", flush=True)


def _save_figure6b(windows: dict, preds: dict, path: Path, caption: str) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(9, 5, figsize=(10.6, 8.6))
    motion = {}
    for s, k in enumerate((1, 2, 3)):
        base = s * 3
        inputs = windows[k]["inputs"]
        targets = windows[k]["targets"]
        pred = preds[k]
        in_frames = windows[k]["input_frames"]
        tg_frames = windows[k]["target_frames"][:2]
        for c in range(5):
            _draw(axes[base, c], _frame_np(inputs[c]), title=f"in {in_frames[c]}")
        axes[base, 0].set_ylabel(f"{k}x input", fontsize=8)
        for c in range(5):
            if c < 2:
                _draw(axes[base + 1, c], _frame_np(pred[c]), title=f"pred {tg_frames[c]}")
                _draw(axes[base + 2, c], _frame_np(targets[c]), title=f"true {tg_frames[c]}")
            else:
                axes[base + 1, c].axis("off")
                axes[base + 2, c].axis("off")
        axes[base + 1, 0].set_ylabel("predicted", fontsize=8)
        axes[base + 2, 0].set_ylabel("true", fontsize=8)
        last = _component_report(_frame_np(inputs[-1]))
        speed_motion = []
        for h in range(2):
            true_r = _component_report(_frame_np(targets[h]))
            pred_r = _component_report(_frame_np(pred[h]))
            mse = float((pred[h] - targets[h]).pow(2).mean().item())
            speed_motion.append(
                {
                    "frame": tg_frames[h],
                    "mse": mse,
                    "true": _public_components(true_r),
                    "predicted": _public_components(pred_r),
                    "true_from_last_input": _displacement(last, true_r),
                    "predicted_from_last_input": _displacement(last, pred_r),
                }
            )
        motion[f"{k}x"] = {
            "input_frames": list(in_frames),
            "target_frames_shown": list(tg_frames),
            "last_input": _public_components(last),
            "horizons": speed_motion,
        }
    fig.suptitle(
        f"Figure 6b. Seed-0 symmetrised D+B, dataset {FIGURE5_DATASET}, speeds 1x 2x 3x",
        fontsize=11,
    )
    wrapped = "\n".join(textwrap.wrap(caption, width=140))
    fig.text(0.5, 0.008, wrapped, ha="center", va="bottom", fontsize=7.5)
    fig.tight_layout(rect=(0, 0.10, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, facecolor="white")
    plt.close(fig)
    print(f"saved {path}", flush=True)
    return motion


def _predict_rows(forecaster, video, kind: str, device):
    import torch

    from dl4stg.transforms import mirror, rotate

    transform = mirror if kind == "M" else (lambda z: rotate(z, k=1))
    inputs = video[0:5].unsqueeze(0).to(device)
    targets = video[5:10].unsqueeze(0).to(device)
    with torch.no_grad():
        fx = forecaster(inputs)
        ftx = forecaster(transform(inputs))
    tx = transform(inputs)
    ty = transform(targets)
    tfx = transform(fx)
    rows = [
        torch.cat([inputs, fx], 1)[0].detach().cpu(),
        torch.cat([tx, ty], 1)[0].detach().cpu(),
        torch.cat([tx, ftx], 1)[0].detach().cpu(),
        torch.cat([tx, tfx], 1)[0].detach().cpu(),
    ]
    per_horizon_c = (ftx - tfx).pow(2).mean(dim=(0, 2, 3, 4)).detach().cpu()
    per_horizon_e = (ftx - ty).pow(2).mean(dim=(0, 2, 3, 4)).detach().cpu()
    return rows, [float(v) for v in per_horizon_c.tolist()], [float(v) for v in per_horizon_e.tolist()]


def _build_figures(weight_path: Path, test, device, out: Path) -> dict:
    import torch

    from dl4stg.model_d import ModelD
    from dl4stg.symmetrise import SymmetrisedForecaster
    from dl4stg.windows import make_window

    underlying = ModelD().to(device)
    if _count_params(underlying) != EXPECTED_PARAMS:
        raise SystemExit("Figure pass parameter count is not 480784.")
    _load_state(underlying, weight_path, device)
    forecaster = SymmetrisedForecaster(underlying).to(device)
    forecaster.eval()

    panels = []
    records = []
    for kind, rank, dataset_index, gif_rel in FIGURE3_PANELS:
        local = _local_index(dataset_index)
        rows, horizon_c, horizon_e = _predict_rows(forecaster, test[local], kind, device)
        score = float(sum(horizon_c) / 5.0)
        if score > GAP_LIMIT:
            raise SystemExit(
                f"Figure 6a video {dataset_index} {kind} gap {sci(score)} is not a float zero."
            )
        letter = "M" if kind == "M" else "R"
        word = "mirror" if kind == "M" else "rotation"
        c_name = "C_M" if kind == "M" else "C_R"
        labels = [
            "X | f(X)",
            f"{letter}(X) | {letter}(Y)",
            f"{letter}(X) | f({letter}(X))",
            f"{letter}(X) | {letter}(f(X))",
        ]
        title = (
            f"{word}, rank {rank}  ·  dataset index {dataset_index}  ·  "
            f"{c_name} = {sci(score)}"
        )
        panel = {
            "rows": rows,
            "row_labels": labels,
            "title": title,
            "gif_title": f"{word} rank {rank}, video {dataset_index}, {c_name}={sci(score)}",
        }
        panels.append(panel)
        _save_gif(panel, out / gif_rel)
        names = ["f(X)", "T(Y)", "f(T(X))", "T(f(X))"]
        records.append(
            {
                "rank": rank,
                "transform": kind,
                "dataset_index": dataset_index,
                "same_video_as_figure3": True,
                "C": score,
                "C_sci": sci(score),
                "E": float(sum(horizon_e) / 5.0),
                "horizon_C": horizon_c,
                "horizon_E": horizon_e,
                "gif": gif_rel,
                "tracks": {name: _track_row(row) for name, row in zip(names, rows)},
            }
        )
        print(
            f"figure6a {word} rank {rank} dataset {dataset_index} "
            f"{c_name}={sci(score)} E={three_sig(float(sum(horizon_e) / 5.0))}",
            flush=True,
        )

    gif_paths = ", ".join(spec[3] for spec in FIGURE3_PANELS)
    caption_a = (
        "Figure 6a. Seed-0 symmetrised D+B at 1x, t=1, on the same four videos as Figure 3. "
        "Mirror ranks are dataset indices 9347 and 9257. Rotation ranks are dataset indices "
        "9118 and 9493. These indices are fixed; they are not recomputed from this model. "
        "f is the wrapped forecaster, the average of the eight restored group predictions. "
        "Each panel is four rows by ten frames: columns 1-5 are the input side and columns "
        "6-10 are the future side. Row 1 is X | f(X), row 2 is T(X) | T(Y), "
        "row 3 is T(X) | f(T(X)), row 4 is T(X) | T(f(X)). "
        "GIFs keep those four rows and step through the five forecast horizons, "
        "with the five input frames held on the left and the current future frame on the right.\n"
        f"GIF paths: {gif_paths}."
    )
    figure_a = out / "figures" / "figure6a.png"
    _save_figure6a(panels, figure_a, caption_a)

    local_b = _local_index(FIGURE5_DATASET)
    video = test[local_b]
    windows = {}
    preds = {}
    expected = {
        1: ([1, 2, 3, 4, 5], [6, 7, 8, 9, 10]),
        2: ([1, 3, 5, 7, 9], [11, 13, 15, 17, 19]),
        3: ([1, 4, 7, 10, 13], [16, 19]),
    }
    with torch.no_grad():
        for k, (want_in, want_tg) in expected.items():
            inputs, targets, in_frames, tg_frames = make_window(video, 1, k)
            if in_frames != want_in or tg_frames != want_tg:
                raise SystemExit(f"figure6b {k}x frames {in_frames} {tg_frames}.")
            pred = forecaster(inputs.unsqueeze(0).to(device))[0].detach().cpu()
            if pred.shape[0] != 5:
                raise SystemExit("figure6b prediction does not have 5 frames.")
            windows[k] = {
                "inputs": inputs,
                "targets": targets,
                "input_frames": in_frames,
                "target_frames": tg_frames,
            }
            preds[k] = pred
    caption_b = (
        f"Figure 6b. Seed-0 symmetrised D+B on dataset index {FIGURE5_DATASET}, "
        "the same video as Figure 5. Each speed shows five inputs and the first two "
        "predicted and true futures at t=1. At 3x only those two futures exist inside the clip."
    )
    figure_b = out / "figures" / "figure6b.png"
    motion = _save_figure6b(windows, preds, figure_b, caption_b)
    for path in (figure_a, figure_b):
        if path.stat().st_size < 10_000:
            raise SystemExit(f"{path} is unexpectedly small.")
    del forecaster, underlying
    torch.cuda.empty_cache()
    return {
        "seed": 0,
        "weights": f"weights/{weight_path.name}",
        "a": {
            "path": "figures/figure6a.png",
            "caption": caption_a,
            "panels": records,
            "gif_paths": [spec[3] for spec in FIGURE3_PANELS],
        },
        "b": {
            "path": "figures/figure6b.png",
            "caption": caption_b,
            "dataset_index": FIGURE5_DATASET,
            "same_video_as_figure5": True,
            "motion": motion,
        },
    }


def _preflight() -> None:
    """CPU checks. No dataset download, no training, no weight writes."""
    _ensure_src_on_path()
    import torch
    from torch import nn

    from dl4stg.model_d import ModelD
    from dl4stg.symmetrise import GROUP, SymmetrisedForecaster
    from dl4stg.transforms import apply_g, invert_g, mirror, rotate

    if len(GROUP) != 8 or len(set(GROUP)) != 8:
        raise SystemExit(f"G is not the eight distinct elements: {GROUP}")
    marker = torch.arange(2 * 5 * 64 * 64, dtype=torch.float32).reshape(2, 5, 1, 64, 64)
    for a, mirrored in GROUP:
        restored = invert_g(apply_g(marker, a, mirrored), a, mirrored)
        if not torch.equal(restored, marker):
            raise SystemExit(f"g^{{-1}} g is not the identity for a={a} mirrored={mirrored}.")
    for _kind, _rank, dataset_index, _gif in FIGURE3_PANELS:
        _local_index(dataset_index)
    _local_index(FIGURE5_DATASET)

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

    calls = {"n": 0}

    class _Counting(nn.Module):
        def forward(self, x):
            calls["n"] += 1
            return x[:, :1].repeat(1, 5, 1, 1, 1)

    with torch.no_grad():
        SymmetrisedForecaster(_Counting())(torch.zeros(1, 5, 1, 64, 64))
    if calls["n"] != 8:
        raise SystemExit(f"The wrapper ran the model {calls['n']} times, expected 8.")

    torch.manual_seed(0)
    sample = torch.rand(1, 5, 1, 64, 64)
    untrained = ModelD()
    untrained.eval()
    if _count_params(untrained) != EXPECTED_PARAMS:
        raise SystemExit(f"Untrained Model D parameter count is not {EXPECTED_PARAMS}.")
    if untrained.layer1.dilation != 2 or untrained.layer2.dilation != 2:
        raise SystemExit("Model D dilation is not 2.")
    covered = SymmetrisedForecaster(untrained)
    covered.eval()
    with torch.no_grad():
        plain = (mirror(untrained(sample)) - untrained(mirror(sample))).pow(2).mean()
        closed = (mirror(covered(sample)) - covered(mirror(sample))).pow(2).mean()
    print(
        f"preflight untrained ModelD C_M raw={sci(float(plain))} wrapped={sci(float(closed))}",
        flush=True,
    )
    if float(plain) < 1e-6:
        raise SystemExit("Untrained Model D was already mirror-consistent on the probe.")
    if float(closed) > 1e-8:
        raise SystemExit(f"Wrapped untrained Model D C_M {sci(float(closed))} is not a float zero.")
    print(f"preflight params={EXPECTED_PARAMS}", flush=True)

    out = _output_root()
    for filename in WEIGHTS.values():
        path = out / "weights" / filename
        before = _sha256(path)
        model = ModelD()
        _load_state(model, path, torch.device("cpu"))
        if _count_params(model) != EXPECTED_PARAMS:
            raise SystemExit(f"{filename} parameter count is not {EXPECTED_PARAMS}.")
        if model.layer1.dilation != 2 or model.layer2.dilation != 2:
            raise SystemExit(f"{filename} dilation is not 2.")
        if _sha256(path) != before:
            raise SystemExit(f"{filename} changed during the load check.")
        print(f"preflight loaded {filename} sha256={before} params={EXPECTED_PARAMS}", flush=True)
    copied = _read_epoch_seconds(out / "results" / "table5_db.json")
    print(
        f"preflight table5_db.json epoch seconds "
        f"seed0={copied['0']} seed1={copied['1']} mean={copied['mean']}",
        flush=True,
    )
    print("preflight ok", flush=True)


def main() -> None:
    _ensure_src_on_path()
    import torch

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
    epoch = _read_epoch_seconds(out / "results" / "table5_db.json")
    weight_paths = [out / "weights" / WEIGHTS[seed] for seed in SEEDS]
    hashes_before = {path.name: _sha256(path) for path in weight_paths}

    test, load_seconds = _load_test()
    per_seed = {}
    for seed in SEEDS:
        report = _evaluate_seed(out / "weights" / WEIGHTS[seed], test, device)
        if _sha256(out / "weights" / WEIGHTS[seed]) != report["weights_sha256"]:
            raise SystemExit(f"{WEIGHTS[seed]} changed during evaluation.")
        report["mean_epoch_seconds"] = epoch[str(seed)]
        per_seed[str(seed)] = report
    mean_metrics = _mean_metrics(per_seed["0"]["metrics"], per_seed["1"]["metrics"])
    _check_gaps("D+B mean", mean_metrics)
    inference = [per_seed[str(seed)]["inference_seconds"] for seed in SEEDS]
    member = [per_seed[str(seed)]["group_member_mse"] for seed in SEEDS]
    print(json.dumps(_display(mean_metrics), indent=2), flush=True)
    print(
        f"mean inference_seconds={sum(inference) / len(inference):.6f} "
        f"copied_epoch_seconds={epoch['mean']}",
        flush=True,
    )

    figure6 = _build_figures(out / "weights" / WEIGHTS[0], test, device, out)
    hashes_after = {path.name: _sha256(path) for path in weight_paths}
    if hashes_after != hashes_before:
        raise SystemExit(f"Weight files changed: before={hashes_before} after={hashes_after}")

    table = {
        "table": 6,
        "model": "D+B",
        "forecaster": "symmetrised D+B",
        "definition": "f_bar(X) = (1/8) sum_{g in G} g^{-1}(f(g(X)))",
        "group": "G = {R^a, M R^a : a = 0,1,2,3}. Inverse of R^a is R^{-a}. Each M R^a is its own inverse.",
        "underlying": "ModelD",
        "underlying_dilation": 2,
        "wrapped": "SymmetrisedForecaster",
        "weights": [f"weights/{name}" for name in WEIGHTS.values()],
        "params": EXPECTED_PARAMS,
        "final_model": "symmetrised D+B",
        "trained_this_run": False,
        "weights_modified": False,
        "training_time_source": epoch["source"],
        "training_time_retimed": False,
        "mean_epoch_seconds": epoch["mean"],
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
            "Symmetrised D+B. The clock covers all 8 transforms, underlying forwards, "
            "inverse transforms, and the average, for 1000 original 1x t=1 test videos, "
            "batch 32, eval, no_grad. The clock starts after each batch is on device. "
            "Consistency, speed, and the group-member spread are not included."
        ),
        "group_member_mse_definition": (
            "Mean squared deviation of the eight restored forecasts from their average, "
            "on the 1000 original 1x t=1 test videos. Not part of the inference clock."
        ),
        "cudnn_deterministic": True,
        "gpu_name": gpu_name,
        "cuda_device": cuda_device,
        "cuda": True,
        "torch_version": torch.__version__,
        "kaggle_kernel": "mtayyabtahir/dl4stg-pa2-eval-final",
        "dataset_load_seconds_excluded": load_seconds,
        "weights_sha256": hashes_after,
        "per_seed": per_seed,
        "mean": {
            "metrics": _metric_block(mean_metrics),
            "mean_epoch_seconds": epoch["mean"],
            "inference_seconds": float(sum(inference) / len(inference)),
            "group_member_mse": float(sum(member) / len(member)),
        },
        "inference_seconds": {
            "seed_0": per_seed["0"]["inference_seconds"],
            "seed_1": per_seed["1"]["inference_seconds"],
            "mean": float(sum(inference) / len(inference)),
        },
        "consistency_scientific": {
            "seed_0": {
                "C_M": per_seed["0"]["metrics"]["C_M_sci"],
                "E_M": per_seed["0"]["metrics"]["display"]["E_M"],
                "C_R": per_seed["0"]["metrics"]["C_R_sci"],
                "E_R": per_seed["0"]["metrics"]["display"]["E_R"],
            },
            "seed_1": {
                "C_M": per_seed["1"]["metrics"]["C_M_sci"],
                "E_M": per_seed["1"]["metrics"]["display"]["E_M"],
                "C_R": per_seed["1"]["metrics"]["C_R_sci"],
                "E_R": per_seed["1"]["metrics"]["display"]["E_R"],
            },
            "mean": {
                "C_M": sci(mean_metrics["C_M"]),
                "E_M": three_sig(mean_metrics["E_M"]),
                "C_R": sci(mean_metrics["C_R"]),
                "E_R": three_sig(mean_metrics["E_R"]),
            },
        },
        "figure6": figure6,
    }
    destination = out / "results" / "table6_final.json"
    _write_text(destination, json.dumps(table, indent=2) + "\n")
    if hashes_after != {path.name: _sha256(path) for path in weight_paths}:
        raise SystemExit("Weight files changed while writing the results.")
    print(f"wrote {destination}", flush=True)


if __name__ == "__main__":
    _ensure_src_on_path()
    if "--check" in sys.argv:
        _preflight()
    else:
        main()
