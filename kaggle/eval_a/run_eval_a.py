"""Kaggle entry. No training.

The CLI uploads only this file (the source limit is 1 MB). Model A, the
package, and both weight files are the private dataset
mtayyabtahir/dl4stg-pa2-eval-a-files. This copies that bundle into
/kaggle/working and then runs eval_a.py.
"""

from __future__ import annotations

import runpy
import shutil
import sys
import zipfile
from pathlib import Path


def _print_input_files(root: Path) -> None:
    print(f"kaggle_input={root}", flush=True)
    if not root.is_dir():
        return
    for path in sorted(root.rglob("*")):
        if path.is_file():
            print(f"input_file {path} bytes={path.stat().st_size}", flush=True)


def find_bundle_root() -> Path:
    root = Path("/kaggle/input")
    if not root.is_dir():
        raise SystemExit("No /kaggle/input. Attach mtayyabtahir/dl4stg-pa2-eval-a-files.")
    _print_input_files(root)
    zips = sorted(root.rglob("bundle.zip"))
    if len(zips) == 1:
        unpacked = Path("/tmp/dl4stg_eval_a_bundle")
        if unpacked.exists():
            shutil.rmtree(unpacked)
        unpacked.mkdir(parents=True)
        with zipfile.ZipFile(zips[0]) as archive:
            archive.extractall(unpacked)
        print(f"unzipped {zips[0]} to {unpacked}", flush=True)
        return unpacked
    for script in sorted(root.rglob("eval_a.py")):
        base = script.parent
        weights = base / "weights" / "model_a_seed0.pt"
        transforms = base / "src" / "dl4stg" / "transforms.py"
        if weights.is_file() and transforms.is_file() and (base / "weights" / "model_a_seed1.pt").is_file():
            return base
    raise SystemExit(
        "The attached dataset has neither bundle.zip nor eval_a.py beside src/ and weights/."
    )


def copy_into_working(bundle: Path) -> Path:
    dest = Path("/kaggle/working")
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(bundle / "eval_a.py", dest / "eval_a.py")
    shutil.copytree(bundle / "src", dest / "src", dirs_exist_ok=True)
    shutil.copytree(bundle / "weights", dest / "weights", dirs_exist_ok=True)
    needed = [
        dest / "eval_a.py",
        dest / "src" / "dl4stg" / "model_a.py",
        dest / "src" / "dl4stg" / "transforms.py",
        dest / "src" / "dl4stg" / "windows.py",
        dest / "weights" / "model_a_seed0.pt",
        dest / "weights" / "model_a_seed1.pt",
    ]
    missing = [str(path) for path in needed if not path.is_file()]
    if missing:
        raise SystemExit("Copy into /kaggle/working missed: " + ", ".join(missing))
    print(f"copied bundle from {bundle} into {dest}", flush=True)
    return dest / "eval_a.py"


def main() -> None:
    script = copy_into_working(find_bundle_root())
    sys.path.insert(0, str(script.parent / "src"))
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
