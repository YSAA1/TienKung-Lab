# Z2 29DoF source

- Upstream: `https://github.com/nubot-zhixing/z2-lab-stable-AMP`
- Commit: `c78eb1f8e31b7f7872733110c10276b7b2159414`
- Robot: original compiled `usd/assembly.usd` (29DoF AMP plant, same as upstream `z2.py`)
- URDF `assembly_urdf_29/assembly.urdf` is source/kinematics, not the training spawn
- Teacher plant: `Z2_29DOF_WALK_POSE_DAMPED_PD_CFG` on that USD, root z=0.75
- Generic `Z2_29DOF_CFG` (ankle 75 Nm, asset root z=0.8) is listed, not used by `z2_loco_teacher`
- 20DoF and 23DoF assets are not copied

Documented adapters, not plant changes:

1. MJCF sphere-hand `quat="0 0 0 0"` → identity in `mjcf/assembly.xml` (MuJoCo rejects a zero quaternion). Original XML is `mjcf/upstream_assembly.xml`.
2. Training URDF left-foot collision stays the upstream `R_ankle_roll_link.STL` reference.
