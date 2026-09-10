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

"""Z2 29-DoF asset package.

Import ``legged_lab.assets.z2.z2`` inside an Isaac Lab process for the
articulation config. The package root stays pure Python so contract tests can
import the joint order and AMP schema without launching Isaac Sim.
"""

from .constants import (
    NUM_Z2_29DOF_JOINTS,
    Z2_29DOF_JOINT_NAMES,
    Z2_NOMINAL_FEET_Y_DISTANCE,
    Z2_STANDING_PELVIS_Z,
)

__all__ = [
    "NUM_Z2_29DOF_JOINTS",
    "Z2_29DOF_JOINT_NAMES",
    "Z2_NOMINAL_FEET_Y_DISTANCE",
    "Z2_STANDING_PELVIS_Z",
]
