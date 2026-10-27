"""Reproducible synthetic fields, measurement faults, and broadcast dynamics."""
from dataclasses import dataclass
import numpy as np
import torch

HISTORY = 8
SENSORS = 16
WORLD_VERSION = 2
WORLD_NAMES = ("moving", "filtering", "relay")


@dataclass
class Batch:
    # Inputs: value, source flag, observed-time coordinate, sensor coordinate.
    x: torch.Tensor             # [batch, history, sensors, 4]
    y: torch.Tensor             # [batch, sensors]
    mask: torch.Tensor          # [batch, sensors]
    meta: dict

    def to(self, device):
        return Batch(self.x.to(device), self.y.to(device), self.mask.to(device), self.meta)


def generate(world, size, seed, *, mode=None, distance=None, history=HISTORY, sensors=SENSORS, noise=0.25):
    """Generate reproducible episodes without exposing future values to the model.

    `mode` is a moving velocity (±1, ±2, or ±3) or filtering case (agreement/persistence).
    Relay distance is optional and used for paired, geometry-controlled examples.
    All models receive the same features; only the relay uses the source flag.
    """
    rng = np.random.default_rng(seed)
    values = np.empty((size, history, sensors), np.float32)
    roles = np.zeros_like(values)
    mask = np.ones((size, sensors), np.float32)
    meta = {"world": world, "seed": seed}
    if world == "moving":
        direction = rng.choice([-3, -2, -1, 1, 2, 3], size=size) if mode is None else np.full(size, int(mode))
        if not np.isin(direction, [-3, -2, -1, 1, 2, 3]).all():
            raise ValueError("Moving velocity must be ±1, ±2, or ±3.")
        if noise < 0:
            raise ValueError("Measurement noise must be nonnegative.")
        # An extended initial field avoids wrap-around and boundary injections.
        initial = rng.uniform(-1, 1, (size, sensors + 6 * (history + 1))).astype(np.float32)
        offset = 3 * (history + 1)
        for b, v in enumerate(direction):
            for t in range(history):
                values[b, t] = initial[b, offset + np.arange(sensors) - v * t]
        y = np.stack([initial[b, offset + np.arange(sensors) - v * history]
                      for b, v in enumerate(direction)])
        values += rng.normal(0, noise, values.shape).astype(np.float32)
        # The upstream sensor is observed for every scored forecast and velocity.
        mask[:, :3] = 0
        mask[:, -3:] = 0
        meta.update(direction=direction, initial=initial, noise=noise)
    elif world == "filtering":
        cases = rng.choice([0, 1], size=size) if mode is None else np.full(size, {"agreement": 0, "persistence": 1}[mode])
        clean = rng.uniform(-0.75, 0.75, (size, sensors)).astype(np.float32)
        corrupt = np.zeros_like(values, dtype=bool)
        for b, case in enumerate(cases):
            if case == 0:
                # A uniform physical field; isolated sensors have stable offsets.
                clean[b] = rng.uniform(-0.75, 0.75)
                values[b] = clean[b]
                bad = np.arange(rng.integers(0, 3), sensors, 3)
                values[b, :, bad] = rng.uniform(-1, 1, (len(bad), 1))
                corrupt[b, :, bad] = True
            else:
                # Independent stable physical readings; brief synchronous interference.
                values[b] = clean[b]
                bad_times = np.array([history - 1, rng.integers(history - 5, history - 1)])
                values[b, bad_times] = rng.uniform(-1, 1, (len(bad_times), sensors))
                corrupt[b, bad_times] = True
        y = clean
        mask[:, :3] = 0
        mask[:, -3:] = 0
        meta.update(case=cases, corrupt=corrupt, clean=clean)
    elif world == "relay":
        values[:] = rng.uniform(-1, 1, values.shape)
        if distance is None:
            source = rng.integers(0, sensors, size=size)
            query = rng.integers(0, sensors, size=size)
        else:
            if not 1 <= int(distance) < sensors:
                raise ValueError("Relay distance must lie between 1 and sensors - 1.")
            source = rng.integers(0, sensors - int(distance), size=size)
            query = source + int(distance)
        roles[np.arange(size), :, source] = 1
        y = np.repeat(values[np.arange(size), -1, source, None], sensors, axis=1)
        # Broadcast forecasting: every non-source sensor receives the signal.
        mask[np.arange(size), source] = 0
        meta.update(source=source, query=query,
                    distances=np.abs(np.arange(sensors)[None, :] - source[:, None]))
    else:
        raise ValueError(f"Unknown world: {world}")
    time = np.broadcast_to(np.linspace(-1, 0, history, dtype=np.float32)[None, :, None], values.shape)
    space = np.broadcast_to(np.linspace(-1, 1, sensors, dtype=np.float32)[None, None, :], values.shape)
    x = np.stack((values, roles, time, space), axis=-1)
    return Batch(torch.from_numpy(x.copy()), torch.from_numpy(y.copy()), torch.from_numpy(mask), meta)


def moving_pair(seed=17, sensor=8):
    """Change one initial-condition value outside the target sensor's history."""
    first = generate("moving", 1, seed, mode=1, noise=0)
    second = generate("moving", 1, seed, mode=1, noise=0)
    # This initial value appears on a diagonal upstream, never at the target.
    old = float(first.y[0, sensor])
    new = -0.9 if old > 0 else 0.9
    offset = 3 * (HISTORY + 1)
    latent_index = offset + sensor - HISTORY
    for t in range(HISTORY):
        s = latent_index - offset + t
        if 0 <= s < SENSORS:
            second.x[0, t, s, 0] = new
    second.y[0, sensor] = new
    second.meta["initial"][0, latent_index] = new
    return first, second


def relay_pair(seed=23, distance=8):
    """A valid episode pair differing only in the latest marked-source value."""
    first = generate("relay", 1, seed, distance=distance)
    second = generate("relay", 1, seed, distance=distance)
    source = int(first.meta["source"][0])
    old = float(first.x[0, -1, source, 0])
    new = -0.9 if old > 0 else 0.9
    second.x[0, -1, source, 0] = new
    second.y[:] = new
    return first, second
