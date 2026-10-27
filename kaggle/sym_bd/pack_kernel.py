"""Build the private dataset zip for the symmetrised B and D evaluation.

The kernel source limit is about 1 MB. The four checkpoints are several
megabytes, so they are packed into a dataset with the code. They are not
written into run_sym_bd.py.
"""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
KERNEL = Path(__file__).resolve().parent
FILES = [
    ("scripts/eval_sym_bd.py", "scripts/eval_sym_bd.py"),
    ("src/dl4stg/__init__.py", "src/dl4stg/__init__.py"),
    ("src/dl4stg/convlstm.py", "src/dl4stg/convlstm.py"),
    ("src/dl4stg/data.py", "src/dl4stg/data.py"),
    ("src/dl4stg/metrics.py", "src/dl4stg/metrics.py"),
    ("src/dl4stg/model_a.py", "src/dl4stg/model_a.py"),
    ("src/dl4stg/model_d.py", "src/dl4stg/model_d.py"),
    ("src/dl4stg/symmetrise.py", "src/dl4stg/symmetrise.py"),
    ("src/dl4stg/transforms.py", "src/dl4stg/transforms.py"),
    ("src/dl4stg/windows.py", "src/dl4stg/windows.py"),
    ("weights/model_b_seed0.pt", "weights/model_b_seed0.pt"),
    ("weights/model_b_seed1.pt", "weights/model_b_seed1.pt"),
    ("weights/model_d_seed0.pt", "weights/model_d_seed0.pt"),
    ("weights/model_d_seed1.pt", "weights/model_d_seed1.pt"),
    ("results/table5_b.json", "results/table5_b.json"),
    ("results/table5_d.json", "results/table5_d.json"),
]


def main() -> None:
    import sys

    runner = KERNEL / "run_sym_bd.py"
    runner_bytes = runner.stat().st_size
    if runner_bytes > 1_000_000:
        raise SystemExit(f"Kernel script is {runner_bytes} bytes, above 1 MB.")
    runner_text = runner.read_text(encoding="utf-8")
    if "base64" in runner_text or "state_dict" in runner_text:
        raise SystemExit("Refusing to pack a runner that embeds weights.")

    out = Path(sys.argv[1]).resolve()
    if out.exists():
        shutil.rmtree(out)
    dataset = out / "dataset"
    dataset.mkdir(parents=True)
    weight_bytes = 0
    bundle = dataset / "bundle.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for src_rel, arc_name in FILES:
            src = REPO / src_rel
            if not src.is_file():
                raise SystemExit(f"Missing {src}.")
            if src_rel.endswith(".pt"):
                weight_bytes += src.stat().st_size
                if src.stat().st_size < 1_000_000:
                    raise SystemExit(f"{src_rel} is unexpectedly small.")
            archive.write(src, arc_name)
            print(f"zip {arc_name} bytes={src.stat().st_size}", flush=True)
    if weight_bytes < 1_000_000:
        raise SystemExit("Weight files do not exceed 1 MB; the pack decision was wrong.")
    meta = {
        "title": "DL4STG PA2 Sym BD Files",
        "id": "mtayyabtahir/dl4stg-pa2-sym-bd-files",
        "licenses": [{"name": "CC0-1.0"}],
    }
    (dataset / "dataset-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(f"runner_bytes={runner_bytes}", flush=True)
    print(f"weight_bytes={weight_bytes} packed as a dataset, not inlined", flush=True)
    print(f"bundle_bytes={bundle.stat().st_size}", flush=True)
    print(f"dataset={dataset}", flush=True)


if __name__ == "__main__":
    main()
