# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.

"""CLI: build the curated Z2 AMP v2 source set (forward-motion clips only).

Copies the three promoted PKLs from the pinned upstream fixture into
``motion_source_raw`` (mirroring the upstream layout) and converts the curated
v2 whitelist into ``motion_source_z2_v2``. Isaac-free; run before
``generate_z2_amp_expert.py --motion-dir legged_lab/envs/z2/datasets/motion_source_z2_v2``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from legged_lab.assets.z2.amp_source import CURATED_V2_RAW_FILES, ROOT, convert_curated_v2

DEFAULT_UPSTREAM = ROOT / "work/upstream-z2/legged_lab/envs/z2/datasets/z2_data/z2_data_pkl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-dir", type=Path, default=DEFAULT_UPSTREAM)
    args = parser.parse_args()
    raw_dir = ROOT / "legged_lab/envs/z2/datasets/motion_source_raw"
    copied = []
    for stem, filename in CURATED_V2_RAW_FILES.items():
        destination = raw_dir / filename
        if destination.is_file():
            continue
        source = args.upstream_dir / filename
        if not source.is_file():
            raise FileNotFoundError(f"upstream fixture missing {source}; run fetch_z2_upstream_fixtures.py")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        copied.append(str(destination.relative_to(ROOT)))
    manifest = convert_curated_v2(raw_dir=raw_dir)
    print(
        json.dumps(
            {
                "copied": copied,
                "stems": [item["stem"] for item in manifest["motions"]],
                "weights": {item["stem"]: item["motion_weight"] for item in manifest["motions"]},
                "source_dir": "legged_lab/envs/z2/datasets/motion_source_z2_v2",
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
