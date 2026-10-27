"""Build the private dataset zip and the Kaggle push folder.

The kernel source limit is 1 MB, so the weights cannot live in the script.
bundle.zip holds eval_a.py, src/dl4stg, and both checkpoints. run_eval_a.py
unpacks it into /kaggle/working.
"""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FILES = [
    ("kaggle/eval_a/eval_a.py", "eval_a.py"),
    ("src/dl4stg/__init__.py", "src/dl4stg/__init__.py"),
    ("src/dl4stg/convlstm.py", "src/dl4stg/convlstm.py"),
    ("src/dl4stg/model_a.py", "src/dl4stg/model_a.py"),
    ("src/dl4stg/transforms.py", "src/dl4stg/transforms.py"),
    ("src/dl4stg/windows.py", "src/dl4stg/windows.py"),
    ("src/dl4stg/metrics.py", "src/dl4stg/metrics.py"),
    ("src/dl4stg/data.py", "src/dl4stg/data.py"),
    ("weights/model_a_seed0.pt", "weights/model_a_seed0.pt"),
    ("weights/model_a_seed1.pt", "weights/model_a_seed1.pt"),
]


def main() -> None:
    import sys

    out = Path(sys.argv[1]).resolve()
    if out.exists():
        shutil.rmtree(out)
    dataset = out / "dataset"
    kernel = out / "kernel"
    dataset.mkdir(parents=True)
    kernel.mkdir(parents=True)
    bundle = dataset / "bundle.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for src_rel, arc_name in FILES:
            archive.write(REPO / src_rel, arc_name)
            print(f"zip {arc_name}", flush=True)
    meta = {
        "title": "DL4STG PA2 Eval A Files",
        "id": "mtayyabtahir/dl4stg-pa2-eval-a-files",
        "licenses": [{"name": "CC0-1.0"}],
    }
    (dataset / "dataset-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    shutil.copy(REPO / "kaggle" / "eval_a" / "run_eval_a.py", kernel / "run_eval_a.py")
    shutil.copy(REPO / "kaggle" / "eval_a" / "kernel-metadata.json", kernel / "kernel-metadata.json")
    print(f"bundle_bytes={bundle.stat().st_size}", flush=True)
    print(f"dataset={dataset}", flush=True)
    print(f"kernel={kernel}", flush=True)


if __name__ == "__main__":
    main()
