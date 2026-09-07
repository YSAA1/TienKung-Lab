"""Official G1 locomanipulation asset adapted to the pinned Isaac 5.1 runtime.

PD, limits, motor types, geometry and initial state come from official_g1.
Only the asset root and contact-sensor activation adapt the locomotion host.
The legacy mode15 URDF configuration remains in legacy_mode15.py.
"""

from .official_g1 import G1_29DOF_CFG as OFFICIAL_G1_29DOF_CFG

OFFICIAL_G1_USD_URL = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1" "/Isaac/Robots/Unitree/G1/g1.usd"
)
G1_29DOF_CFG = OFFICIAL_G1_29DOF_CFG.copy()
# This server's IsaacLab asset constant points to an incomplete local 4.5 cache.
G1_29DOF_CFG.spawn.usd_path = OFFICIAL_G1_USD_URL
G1_29DOF_CFG.spawn.activate_contact_sensors = True
