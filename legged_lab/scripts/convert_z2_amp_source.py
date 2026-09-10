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

"""CLI: whitelist Z2 PKL/NPZ -> policy-order CSV. Isaac-free."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from legged_lab.assets.z2.amp_source import convert_whitelist


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--csv-dir", type=Path)
    args = parser.parse_args()
    manifest = convert_whitelist(raw_dir=args.raw_dir, csv_dir=args.csv_dir)
    print(json.dumps({"motions": [item["stem"] for item in manifest["motions"]], "n": len(manifest["motions"])}))


if __name__ == "__main__":
    main()
