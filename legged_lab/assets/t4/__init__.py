"""T4 27-DoF robot asset package.

Import ``legged_lab.assets.t4.t4`` inside an IsaacLab process for ``T4_CFG``.
The package root stays pure Python so offline motion tools and contract tests can
import the joint contract and the observation schemas without launching IsaacSim.
"""

from . import schemas
from .constants import T4_JOINT_NAMES, T4_NOMINAL_FEET_Y_DISTANCE

__all__ = ["T4_JOINT_NAMES", "T4_NOMINAL_FEET_Y_DISTANCE", "schemas"]
