# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

"""Materialize the pinned upstream Z2 repo used by the Z2 asset-contract tests.

``tests/test_z2_asset_contract.py`` proves our Z2 assets are correct
transformations of the upstream originals (joint order, URDF/MJCF/USD lineage,
expert pkl -> txt conversion). The upstream tree is ~380 MB, so it is fetched
once into the git-ignored ``work/upstream-z2`` instead of being vendored:

    python scripts/fetch_z2_upstream_fixtures.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/nubot-zhixing/z2-lab-stable-AMP.git"
PINNED_COMMIT = "c78eb1f8e31b7f7872733110c10276b7b2159414"
DEST = Path(__file__).resolve().parents[1] / "work" / "upstream-z2"


def run(command: list[str], cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    if not (DEST / ".git").exists():
        DEST.parent.mkdir(parents=True, exist_ok=True)
        print(f"cloning {REPO_URL} into {DEST}")
        run(["git", "clone", REPO_URL, str(DEST)])
    # the asset-contract tests compare raw bytes against our LF-canonical tree
    run(["git", "config", "core.autocrlf", "false"], cwd=DEST)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=DEST, check=True, capture_output=True, text=True
    ).stdout.strip()
    if head != PINNED_COMMIT:
        print(f"checkout pinned fixture commit {PINNED_COMMIT[:12]} (was {head[:12]})")
        run(["git", "fetch", "origin"], cwd=DEST)
        run(["git", "checkout", "--detach", PINNED_COMMIT], cwd=DEST)
    print(f"upstream-z2 fixtures ready at {DEST} @ {PINNED_COMMIT[:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
