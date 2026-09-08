"""Download the declared LAFAN1 G1 walk/run CSVs used by G1 AMP.

Official Unitree HF dataset was pulled; this uses the public mirror
``lvhaidong/LAFAN1_Retargeting_Dataset``. Prefers hf-mirror, then huggingface.co.
"""

from __future__ import annotations

import argparse
import urllib.error
import urllib.request
from pathlib import Path

from legged_lab.assets.unitree_g1.schemas import (
    AMP_HELD_OUT_MOTIONS,
    AMP_MOTION_CLASSES,
    AMP_MOTION_SOURCE_DIR,
    LAFAN1_SOURCE_DATASET,
)

ROOT = Path(__file__).resolve().parents[2]
MIRRORS = (
    f"https://hf-mirror.com/datasets/{LAFAN1_SOURCE_DATASET}/resolve/main/g1",
    f"https://huggingface.co/datasets/{LAFAN1_SOURCE_DATASET}/resolve/main/g1",
)


def _stems() -> list[str]:
    return sorted(stem for stem in AMP_MOTION_CLASSES if stem not in AMP_HELD_OUT_MOTIONS)


def download_motions(output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for stem in _stems():
        dest = output_dir / f"{stem}.csv"
        if dest.is_file() and dest.stat().st_size > 10_000:
            print(f"keep {dest} ({dest.stat().st_size} bytes)", flush=True)
            written.append(dest)
            continue
        last_error: Exception | None = None
        for base in MIRRORS:
            url = f"{base}/{stem}.csv"
            print(f"get {url}", flush=True)
            try:
                urllib.request.urlretrieve(url, dest)
                if dest.stat().st_size < 10_000:
                    raise RuntimeError(f"{dest} is too small after download ({dest.stat().st_size} bytes)")
                print(f"wrote {dest} ({dest.stat().st_size} bytes)", flush=True)
                last_error = None
                break
            except (urllib.error.URLError, OSError, RuntimeError) as error:
                last_error = error
                print(f"failed {url}: {error}", flush=True)
        if last_error is not None:
            raise RuntimeError(f"could not download {stem}.csv") from last_error
        written.append(dest)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Download LAFAN1 G1 walk/run CSVs for G1 AMP.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / AMP_MOTION_SOURCE_DIR)
    args = parser.parse_args()
    paths = download_motions(args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir)
    print(f"g1_lafan_source clips={len(paths)} dir={paths[0].parent}", flush=True)


if __name__ == "__main__":
    main()
