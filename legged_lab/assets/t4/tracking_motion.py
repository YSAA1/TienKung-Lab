"""T4 reference-motion recipe using the shared named-motion loader."""

from pathlib import Path

from legged_lab.assets.t4.constants import T4_JOINT_NAMES
from legged_lab.motion_tracking.loader import REQUIRED_KEYS, body_indices, load_tracking_motion  # noqa: F401

TRACKING_MOTION_SCHEMA_VERSION = "t4_tracking_motion.v1"
TRACKING_MOTION_DIR = Path(__file__).resolve().parents[2] / "envs" / "t4" / "datasets" / "motion_tracking"


def load_t4_tracking_motion(path: Path | str) -> dict:
    """Preserve the T4 joint order and metadata expected by existing artifacts."""
    return load_tracking_motion(path, joint_names=T4_JOINT_NAMES, schema_version=TRACKING_MOTION_SCHEMA_VERSION)
