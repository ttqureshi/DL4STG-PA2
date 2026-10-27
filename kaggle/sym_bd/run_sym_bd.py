"""Kaggle entry. No training, and no weights in this file.

The four checkpoints and the evaluation sources are the private dataset
mtayyabtahir/dl4stg-pa2-sym-bd-files. Those files exceed 1 MB, so they are
not inlined here. This copies the bundle into /kaggle/working and runs
scripts/eval_sym_bd.py.
"""

from __future__ import annotations

import runpy
import shutil
import sys
import zipfile
from pathlib import Path

NEEDED = (
    "scripts/eval_sym_bd.py",
    "src/dl4stg/symmetrise.py",
    "src/dl4stg/transforms.py",
    "src/dl4stg/model_a.py",
    "src/dl4stg/model_d.py",
    "weights/model_b_seed0.pt",
    "weights/model_b_seed1.pt",
    "weights/model_d_seed0.pt",
    "weights/model_d_seed1.pt",
    "results/table5_b.json",
    "results/table5_d.json",
)


def _print_input_files(root: Path) -> None:
    print(f"kaggle_input={root}", flush=True)
    if not root.is_dir():
        return
    for path in sorted(root.rglob("*")):
        if path.is_file():
            print(f"input_file {path} bytes={path.stat().st_size}", flush=True)


def _is_bundle(base: Path) -> bool:
    return all((base / rel).is_file() for rel in NEEDED)


def _unzip(archive: Path) -> Path:
    unpacked = Path("/tmp/dl4stg_sym_bd_bundle")
    if unpacked.exists():
        shutil.rmtree(unpacked)
    unpacked.mkdir(parents=True)
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(unpacked)
    print(f"unzipped {archive} to {unpacked}", flush=True)
    if _is_bundle(unpacked):
        return unpacked
    for script in unpacked.rglob("eval_sym_bd.py"):
        base = script.parent.parent if script.parent.name == "scripts" else script.parent
        if _is_bundle(base):
            return base
    raise SystemExit(f"Unzipped {archive} but the evaluation bundle is incomplete.")


def find_bundle_root() -> Path:
    root = Path("/kaggle/input")
    if not root.is_dir():
        raise SystemExit("No /kaggle/input. Attach mtayyabtahir/dl4stg-pa2-sym-bd-files.")
    _print_input_files(root)
    for script in sorted(root.rglob("eval_sym_bd.py")):
        base = script.parent.parent if script.parent.name == "scripts" else script.parent
        if _is_bundle(base):
            return base
    zips = sorted(root.rglob("bundle.zip"))
    if len(zips) == 1:
        return _unzip(zips[0])
    raise SystemExit(
        "The attached dataset has neither a complete bundle nor one bundle.zip."
    )


def copy_into_working(bundle: Path) -> Path:
    dest = Path("/kaggle/working")
    dest.mkdir(parents=True, exist_ok=True)
    for rel in NEEDED:
        src = bundle / rel
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)
        print(f"copied {rel} bytes={target.stat().st_size}", flush=True)
    missing = [rel for rel in NEEDED if not (dest / rel).is_file()]
    if missing:
        raise SystemExit("Copy into /kaggle/working missed: " + ", ".join(missing))
    # The evaluator imports the rest of the package from the same tree.
    for rel in (
        "src/dl4stg/__init__.py",
        "src/dl4stg/convlstm.py",
        "src/dl4stg/data.py",
        "src/dl4stg/metrics.py",
        "src/dl4stg/windows.py",
    ):
        src = bundle / rel
        if not src.is_file():
            raise SystemExit(f"Bundle is missing {rel}.")
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)
    return dest / "scripts" / "eval_sym_bd.py"


def main() -> None:
    script = copy_into_working(find_bundle_root())
    sys.path.insert(0, str(script.parents[1] / "src"))
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
