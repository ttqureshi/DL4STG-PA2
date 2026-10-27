"""Pack D+B training sources into one Kaggle script.

The kernel upload limit is about 1 MB and only the code file is uploaded.
Weights are not embedded. The script trains on the Kaggle GPU and writes
checkpoints to kernel output.
"""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
FILES = [
    "src/dl4stg/__init__.py",
    "src/dl4stg/convlstm.py",
    "src/dl4stg/data.py",
    "src/dl4stg/metrics.py",
    "src/dl4stg/model_a.py",
    "src/dl4stg/model_d.py",
    "src/dl4stg/transforms.py",
    "src/dl4stg/windows.py",
    "scripts/train_db.py",
]
OUT = Path(__file__).resolve().parent / "train_db.py"
LIMIT = 1_000_000


def main() -> None:
    payload_lines = []
    for rel in FILES:
        if rel.endswith(".pt") or "weights" in rel.split("/"):
            raise SystemExit(f"Refusing to embed weights: {rel}")
        raw = (REPO / rel).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        encoded = base64.b64encode(raw).decode("ascii")
        payload_lines.append(f"    {rel!r}: ({digest!r}, {encoded!r}),")
    body = "\n".join(payload_lines)
    script = f'''"""Kaggle entry for D+B training.

Unpacks the committed sources and runs scripts/train_db.py.
The embedded bytes must stay identical to those files.
Weights are not embedded. GPU and internet are requested in kernel-metadata.json.
"""

from __future__ import annotations

import base64
import hashlib
import os
import runpy
import sys
import tempfile
from pathlib import Path

PAYLOAD = {{
{body}
}}


def main() -> None:
    dest = Path(tempfile.mkdtemp(prefix="dl4stg_train_db_"))
    for rel, (digest, b64) in PAYLOAD.items():
        if rel.endswith(".pt") or "weights" in rel.split("/"):
            raise RuntimeError(f"Weights must not be embedded: {{rel}}")
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
    script = dest / "scripts" / "train_db.py"
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
'''
    encoded_out = script.encode("utf-8")
    if b".pt" in encoded_out and b"model_db_seed" in encoded_out:
        # The training source mentions the output filename. That is not a weight blob.
        pass
    if len(encoded_out) >= LIMIT:
        raise SystemExit(f"Kernel script is {len(encoded_out)} bytes, limit {LIMIT}.")
    OUT.write_bytes(encoded_out)
    print(f"wrote {OUT} bytes={len(encoded_out)}", flush=True)


if __name__ == "__main__":
    main()
