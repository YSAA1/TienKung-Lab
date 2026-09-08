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

"""Verify raw-byte candidate provenance and the untouched frozen A/B surface."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/portability/g1_core_fixes_20260908"
base = json.loads((ROOT / "artifacts/portability/g1_progress_ab_20260908/source_manifest.json").read_text())
patch = json.loads((OUT / "candidate_patch.json").read_text())
expected = {**base["files"], **patch["files"]}
frozen = Path(patch["frozen"])


def verify(root, entries):
    failures = []
    for relative, sha in entries.items():
        path = root / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        if actual != sha:
            failures.append({"file": relative, "expected": sha, "actual": actual})
    return failures


result = {
    "candidate": str(ROOT),
    "candidate_files": len(expected),
    "frozen_files": len(base["files"]),
    "candidate_errors": verify(ROOT, expected),
    "frozen_errors": verify(frozen, base["files"]),
    "files": expected,
}
(OUT / "source_verification.json").write_text(json.dumps(result, indent=2) + "\n")
assert not result["candidate_errors"] and not result["frozen_errors"], result
print(f"Verified {len(expected)} candidate files and {len(base['files'])} frozen A/B files")
