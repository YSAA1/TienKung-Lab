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

"""Record the exact unchanged IsaacLab dependency and its sole content patch."""

import hashlib
import json
import subprocess
from pathlib import Path

LAB = Path("/home/nubot/IsaacLab")
OUT = Path(__file__).resolve().parent


def git(*args):
    return subprocess.check_output(["git", "-C", str(LAB), *args], text=True)


commit = git("rev-parse", "HEAD").strip()
changes = [line.split("\t", 2) for line in git("diff", "--numstat").splitlines()]
tree = {}
for line in git("ls-tree", "-r", "HEAD").splitlines():
    metadata, name = line.split("\t", 1)
    tree[name] = metadata.split()[2]
content = []
for row in changes:
    name = row[2]
    data = (LAB / name).read_bytes()
    blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    if blob != tree[name]:
        content.append(row)
asset = "source/isaaclab/isaaclab/utils/assets.py"
assert commit == "3d5ea25ddbcba05bef4c9acd1dacb9fa728b289b"
assert len(content) == 1 and content[0][2] == asset, content
patch = git("diff", "--", asset)
(OUT / "isaaclab_assets.patch").write_text(patch)
result = {
    "commit": commit,
    "modified_count": len(changes),
    "mode_only_count": len(changes) - len(content),
    "content_changes": content,
    "assets_path": str(LAB / asset),
    "assets_sha256": hashlib.sha256((LAB / asset).read_bytes()).hexdigest(),
    "assets_patch_sha256": hashlib.sha256((OUT / "isaaclab_assets.patch").read_bytes()).hexdigest(),
}
(OUT / "dependency.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
