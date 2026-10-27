"""Build the Kaggle script for Model B.

The kernel entry embeds the source files it runs. It does not embed weights.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
KERNEL = Path(__file__).resolve().parent
FILES = [
    "src/dl4stg/__init__.py",
    "src/dl4stg/convlstm.py",
    "src/dl4stg/data.py",
    "src/dl4stg/metrics.py",
    "src/dl4stg/model_a.py",
    "src/dl4stg/transforms.py",
    "src/dl4stg/windows.py",
    "scripts/train_b.py",
]

RUNNER = '''"""Kaggle entry for Model B training.

Unpacks the committed sources and runs scripts/train_b.py.
The embedded bytes are sources only. Weights are not embedded.
GPU and internet are requested in kernel-metadata.json.
"""

from __future__ import annotations

import base64
import hashlib
import os
import runpy
import sys
import tempfile
from pathlib import Path

PAYLOAD = {payload}

def main() -> None:
    dest = Path(tempfile.mkdtemp(prefix="dl4stg_train_b_"))
    for rel, (digest, b64) in PAYLOAD.items():
        if rel.endswith(".pt") or "/weights/" in rel:
            raise RuntimeError(f"Refusing to unpack an embedded weight file: {{rel}}")
        raw = base64.b64decode(b64)
        got = hashlib.sha256(raw).hexdigest()
        if got != digest:
            raise RuntimeError(f"{{rel}} sha256 {{got}} != {{digest}}")
        path = dest / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        print(f"unpacked {{rel}} sha256 {{got}}", flush=True)
    (dest / "figures").mkdir()
    (dest / "results").mkdir()
    (dest / "weights").mkdir()
    working = Path("/kaggle/working")
    if working.is_dir():
        os.chdir(working)
    sys.path.insert(0, str(dest / "src"))
    script = dest / "scripts" / "train_b.py"
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
'''


def main() -> None:
    payload = {}
    for rel in FILES:
        raw = (REPO / rel).read_bytes()
        if rel.endswith(".pt"):
            raise SystemExit(f"Refusing to embed weights: {rel}")
        payload[rel] = (hashlib.sha256(raw).hexdigest(), base64.b64encode(raw).decode("ascii"))
        print(f"pack {rel} bytes={len(raw)}", flush=True)
    text = RUNNER.format(payload=repr(payload))
    if ".pt" in text.split("PAYLOAD = ", 1)[1].split("def main", 1)[0]:
        # Source files do not use a .pt suffix. A hit here would mean a weight blob.
        raise SystemExit("Packed payload mentions a .pt name.")
    out = KERNEL / "train_b.py"
    out.write_text(text, encoding="utf-8", newline="\n")
    meta = {
        "id": "mtayyabtahir/dl4stg-pa2-train-b",
        "title": "DL4STG PA2 Train B",
        "code_file": "train_b.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (KERNEL / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out} bytes={out.stat().st_size}", flush=True)


if __name__ == "__main__":
    main()
