"""Unitree's pure 29-DoF velocity asset using its documented URDF option."""

from pathlib import Path

from .official_velocity_g1 import UNITREE_G1_29DOF_CFG, UnitreeUrdfFileCfg

G1_29DOF_CFG = UNITREE_G1_29DOF_CFG.copy()
G1_29DOF_CFG.spawn = UnitreeUrdfFileCfg(
    asset_path=str(Path(__file__).resolve().parent / "urdf" / "g1_29dof_rev_1_0.urdf"),
)
