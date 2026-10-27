"""Shared metrics and cached CPU inference."""
from functools import lru_cache
from pathlib import Path
import json
import numpy as np
import torch
from .models import Config, TeachingModel, key
from .worlds import WORLD_VERSION

ROOT = Path(__file__).resolve().parents[1]


def mae(prediction, target, mask):
    return ((prediction - target).abs() * mask).sum() / mask.sum().clamp_min(1)


def baselines(batch):
    return {"zero": float(mae(torch.zeros_like(batch.y), batch.y, batch.mask)),
            "last reading": float(mae(batch.x[:, -1, :, 0], batch.y, batch.mask))}


@lru_cache(maxsize=32)
def load_model(world, architecture, pairs=3, seed=0):
    path = ROOT / "checkpoints" / f"{world}__{key(architecture, pairs)}__s{seed}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing bundled checkpoint: {path.name}. See readme.md.")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["world"] != world or saved["world_version"] != WORLD_VERSION:
        raise ValueError(f"Checkpoint {path.name} does not match the bundled world definitions.")
    model = TeachingModel(Config(**saved["config"]))
    model.load_state_dict(saved["state_dict"])
    return model.eval()


def predict(world, architecture, batch, pairs=3, trace=False, attention=False):
    model = load_model(world, architecture, pairs)
    with torch.inference_mode():
        return model(batch.x.cpu(), trace=trace, attention=attention)


@lru_cache(maxsize=4)
def results(directory=None):
    folder = ROOT / "checkpoints"
    if directory is not None:
        folder = folder / directory
    return json.loads((folder / "manifest.json").read_text())


def score_summary(world, architecture, pairs=3, case=None):
    entries = results()["results"][world][key(architecture, pairs)]
    scores = [e["test_mae"] if case is None else e["cases"][case] for e in entries]
    return float(np.mean(scores)), float(np.std(scores))


def verify_bundle():
    """Check the actual teaching claims, checkpoint coverage, and geometry on CPU.

    These are release checks, not notebook exercises. Thresholds distinguish the
    intended effects from baseline error; they do not require an ordering winner.
    """
    from .worlds import generate, moving_pair, relay_pair
    from .models import receptive_field
    checks = {}
    manifest = results()
    assert manifest["world_version"] == WORLD_VERSION
    for world, models in manifest["results"].items():
        for name, entries in models.items():
            assert [entry["seed"] for entry in entries] == [0, 1, 2], (world, name, "missing seeds")
            architecture, pairs = name.split("-p") if "-p" in name else (name, 3)
            for seed in (0, 1, 2):
                model = load_model(world, architecture, int(pairs), seed)
                assert all(p.device.type == "cpu" for p in model.parameters())
    # The capacity controls are part of the release, not optional examples.
    for width in (4, 32):
        directory = f"moving-width{width}"
        control = results(directory)
        assert control["world_version"] == WORLD_VERSION
        assert control["evaluation"]["test_seed"] == manifest["evaluation"]["test_seed"]
        assert set(control["results"]["moving"]) == {"S-first-p3", "T-first-p3", "ST"}
        for name, entries in control["results"]["moving"].items():
            assert [e["seed"] for e in entries] == [0, 1, 2]
            for entry in entries:
                path = ROOT / "checkpoints" / directory / f"moving__{name}__s{entry['seed']}.pt"
                saved = torch.load(path, map_location="cpu", weights_only=True)
                assert saved["world_version"] == WORLD_VERSION and saved["config"]["width"] == width
                model = TeachingModel(Config(**saved["config"]))
                model.load_state_dict(saved["state_dict"])
    checks["checkpoint_coverage"] = True

    for direction in (-3, -2, -1, 1, 2, 3):
        batch = generate("moving", 32, 8, mode=direction, noise=0)
        oracle = torch.zeros_like(batch.y)
        for sensor in range(3, 13):
            oracle[:, sensor] = batch.x[:, -1, sensor-direction, 0]
        assert float(mae(oracle, batch.y, batch.mask)) == 0
    first, second = moving_pair()
    assert torch.equal(first.x[0, :, 8], second.x[0, :, 8])
    assert abs(float(first.y[0, 8]-second.y[0, 8])) > 1
    assert torch.equal(predict("moving", "T", first)[:, 8], predict("moving", "T", second)[:, 8])
    assert score_summary("moving", "T")[0] > .4
    for architecture in ("S-first", "T-first", "ST"):
        assert score_summary("moving", architecture)[0] < .4
    joint = score_summary("moving", "ST")[0]
    for architecture in ("S-first", "T-first"):
        assert joint < score_summary("moving", architecture)[0] - .01
    wide = results("moving-width32")["results"]["moving"]
    wide_means = [np.mean([e["test_mae"] for e in entries]) for entries in wide.values()]
    assert max(wide_means) - min(wide_means) < .01
    assert max(wide_means) < joint - .05
    checks["moving_narrow_joint_advantage"] = True
    checks["moving_wide_gap_is_small"] = True
    checks["temporal_information_limit"] = True

    # The filtering targets are recoverable, not labels selected to reward models.
    agreement = generate("filtering", 32, 13, mode="agreement")
    oracle = agreement.x[:, -1, :, 0].unfold(1, 3, 1).median(dim=-1).values
    assert torch.equal(oracle[:, 2:12], agreement.y[:, 3:13])
    persistence = generate("filtering", 32, 13, mode="persistence")
    assert torch.equal(persistence.x[:, 1:, :, 0].median(dim=1).values, persistence.y)
    for architecture in ("S-first", "T-first"):
        shallow = score_summary("filtering", architecture, 1, "persistence")[0]
        deep = score_summary("filtering", architecture, 3, "persistence")[0]
        assert shallow > .07 and deep < .025 and deep < shallow / 4
    checks["depth_mitigates_shallow_error"] = True

    probe = generate("filtering", 1, 17, mode="agreement")
    changed = probe.x.clone()
    changed[0, 5, 8, 0] = -.9 if changed[0, 5, 8, 0] > 0 else .9
    for architecture in ("S-first", "T-first"):
        model = load_model("filtering", architecture, 1)
        with torch.inference_mode():
            _, before, _ = model(probe.x, trace=True)
            _, after, _ = model(changed, trace=True)
        first_change = (before[1][0]-after[1][0]).norm(dim=-1)
        if architecture == "S-first":
            assert first_change[5, 7] > 1e-7 and first_change[6].max() == 0
        else:
            assert first_change[6, 8] > 1e-7 and first_change[:, 7].max() == 0
    assert np.array_equal(receptive_field(["S", "T"]*3, 7, 8), receptive_field(["ST"]*3, 7, 8))
    checks["ordering_first_block_paths"] = True

    # Verify actual derivatives cannot cross the advertised local boundary.
    first, second = relay_pair(distance=8)
    query = int(first.meta["query"][0])
    for architecture in ("T", "S-first", "T-first", "ST"):
        model = load_model("relay", architecture)
        x = first.x.clone().requires_grad_(True)
        gradient = torch.autograd.grad(model(x)[0, query], x)[0][0, :, :, 0].detach().numpy()
        support = receptive_field(model.kinds, 7, query)
        assert np.max(np.abs(gradient[~support]), initial=0) == 0
        assert torch.equal(predict("relay", architecture, first)[:, query], predict("relay", architecture, second)[:, query])
        assert score_summary("relay", architecture, case="far")[0] > .4
    a = predict("relay", "A", first)[0, query]
    b = predict("relay", "A", second)[0, query]
    assert abs(float((b-a)-(second.y[0, query]-first.y[0, query]))) < .05
    assert score_summary("relay", "A", case="far")[0] < .03
    assert score_summary("relay", "ST", case="near")[0] < .03
    _, _, weights = predict("relay", "A", first, attention=True)
    source = int(first.meta["source"][0])
    default_weight = float(weights[2][0, 1, 7*16+query, 7*16+source])
    checks["local_reach_and_distant_attention"] = True
    checks["default_latest_source_attention"] = default_weight
    return checks
