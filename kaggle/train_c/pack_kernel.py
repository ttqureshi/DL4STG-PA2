"""Build the Kaggle script for Model C training.

The kernel source limit is 1 MB. This embeds the Python sources the trainer
imports. It does not embed checkpoints.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FILES = [
    "src/dl4stg/__init__.py",
    "src/dl4stg/convlstm.py",
    "src/dl4stg/model_a.py",
    "src/dl4stg/windows.py",
    "src/dl4stg/metrics.py",
    "src/dl4stg/data.py",
    "src/dl4stg/transforms.py",
    "scripts/train_c.py",
]
RUNNER = REPO / "kaggle" / "train_c" / "train_c.py"
LIMIT = 900_000


def main() -> None:
    payload = {}
    for rel in FILES:
        path = REPO / rel
        raw = path.read_bytes()
        if path.suffix == ".pt" or "weights" in rel:
            raise SystemExit(f"refusing to embed {rel}")
        payload[rel] = (hashlib.sha256(raw).hexdigest(), base64.b64encode(raw).decode("ascii"))
        print(f"pack {rel} bytes={len(raw)}", flush=True)
    header = '''"""Kaggle entry for Model C training.

Unpacks the committed sources and runs scripts/train_c.py.
No checkpoints are embedded. GPU and internet are requested in kernel-metadata.json.
"""

from __future__ import annotations

import base64
import hashlib
import os
import runpy
import sys
import tempfile
from pathlib import Path

PAYLOAD = '''
    footer = '''


def main() -> None:
    dest = Path(tempfile.mkdtemp(prefix="dl4stg_train_c_"))
    for rel, (digest, b64) in PAYLOAD.items():
        if rel.endswith(".pt") or "weights/" in rel.replace("\\\\", "/"):
            raise RuntimeError(f"weights were embedded: {rel}")
        raw = base64.b64decode(b64)
        got = hashlib.sha256(raw).hexdigest()
        if got != digest:
            raise RuntimeError(f"{rel} sha256 {got} != {digest}")
        path = dest / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        print(f"unpacked {rel} sha256 {got}", flush=True)
    (dest / "figures").mkdir()
    (dest / "results").mkdir()
    (dest / "weights").mkdir()
    working = Path("/kaggle/working")
    if working.is_dir():
        os.chdir(working)
    sys.path.insert(0, str(dest / "src"))
    script = dest / "scripts" / "train_c.py"
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
'''
    text = header + repr(payload) + footer
    if len(text.encode("utf-8")) > LIMIT:
        raise SystemExit(f"kernel script is {len(text.encode('utf-8'))} bytes, over {LIMIT}")
    RUNNER.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {RUNNER} bytes={RUNNER.stat().st_size}", flush=True)
    json.loads((REPO / "kaggle" / "train_c" / "kernel-metadata.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
