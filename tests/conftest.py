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

"""Shared pytest bootstrap: repo paths on ``sys.path`` and a writable temp base.

Makes ``python -m pytest tests/...`` work in any interpreter without editable
installs (``legged_lab`` and the vendored ``rsl_rl`` both resolve to this repo),
and falls back to a repo-local temp base when the default ``pytest-of-<user>``
directory is corrupted (seen on Windows after an unrelated session wedged it).
"""

from __future__ import annotations

import sys
import tempfile
from getpass import getuser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _bootstrap_sys_path() -> None:
    for entry in (ROOT, ROOT / "rsl_rl"):
        text = str(entry)
        if text not in sys.path:
            sys.path.insert(0, text)


def pytest_configure(config):
    _bootstrap_sys_path()
    if config.option.basetemp is not None:
        return
    default_base = Path(tempfile.gettempdir()) / f"pytest-of-{getuser() or 'unknown'}"
    try:
        default_base.mkdir(parents=True, exist_ok=True)
        probe = default_base / ".write-probe"
        probe.touch()
        probe.unlink()
    except OSError:
        fallback = ROOT / "artifacts" / "pytest-tmp"
        fallback.mkdir(parents=True, exist_ok=True)
        config.option.basetemp = fallback
