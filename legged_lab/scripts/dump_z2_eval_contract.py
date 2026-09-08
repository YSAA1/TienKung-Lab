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

"""Write the Z2 eval/training contract JSON. Isaac-free. Does not run evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from legged_lab.assets.z2.eval_contract import z2_eval_provenance

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/z2_migration/z2_eval_contract.json")
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    payload = z2_eval_provenance()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps({"wrote": str(output), "task": payload["task"], "promotion": payload["lightlp_promotion_distance"]})
    )


if __name__ == "__main__":
    main()
