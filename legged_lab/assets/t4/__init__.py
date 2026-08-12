"""T4 27-DoF robot asset package.

Import ``legged_lab.assets.t4.t4`` inside an IsaacLab process for ``T4_CFG``.
The package root stays pure Python so offline motion tools can import the joint
contract without launching IsaacSim.
"""

from .constants import T4_JOINT_NAMES

__all__ = ["T4_JOINT_NAMES"]
