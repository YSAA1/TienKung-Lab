"""Isaac-free contract for the G2 heightscan vault skill policy.

The student shares the Stage E teacher actor layout (1155D). The frozen G1
tracking observation (150D) exists only as a teacher query and must not enter
the student policy group.
"""

from __future__ import annotations

from legged_lab.assets.t4.schemas import (
    PROPRIO_FIELDS,
    PROPRIO_FRAME_DIM,
    PROPRIO_HISTORY_LENGTH,
    TEACHER_ACTOR_OBS_DIM,
    TEACHER_SCAN_BODY,
    TEACHER_SCAN_CLIP,
    TEACHER_SCAN_DIM,
    TEACHER_SCAN_HEIGHT_OFFSET,
    TEACHER_SCAN_INVALID_VALUE,
    TEACHER_SCAN_OFFSET,
    TEACHER_SCAN_RESOLUTION,
    TEACHER_SCAN_SIZE,
)
from legged_lab.assets.t4.vault_contract import T4_VAULT_ACTION_SCALE, T4_VAULT_BOX_POS, T4_VAULT_BOX_SIZE

G2_POLICY_OBS_DIM = TEACHER_ACTOR_OBS_DIM
G2_TEACHER_OBS_DIM = 150
G2_ACTION_DIM = 27
G2_ACTION_SCALE = T4_VAULT_ACTION_SCALE
G2_SKILL_LIN_VEL_X = 0.8
G2_GAIT_CYCLE = 0.85
G2_GAIT_AIR_RATIO = (0.38, 0.38)
G2_GAIT_PHASE_OFFSET = (0.38, 0.88)
G2_PROPRIO_FRAME_DIM = PROPRIO_FRAME_DIM
G2_PROPRIO_HISTORY_LENGTH = PROPRIO_HISTORY_LENGTH
G2_SCAN_DIM = TEACHER_SCAN_DIM

G2_POLICY_FORBIDDEN_SUBSTRINGS = (
    "motion",
    "reference",
    "box_size",
    "box_pos",
    "obstacle",
    "anchor_pos",
    "anchor_ori",
)

assert G2_POLICY_OBS_DIM == G2_PROPRIO_FRAME_DIM * G2_PROPRIO_HISTORY_LENGTH + G2_SCAN_DIM
assert G2_POLICY_OBS_DIM == 1155


def teacher_scan_local_xy(device=None):
    """IsaacLab ``GridPatternCfg`` xy grid, plus the forward sensor offset."""
    import torch

    x = torch.arange(
        start=-TEACHER_SCAN_SIZE[0] / 2.0,
        end=TEACHER_SCAN_SIZE[0] / 2.0 + 1.0e-5,
        step=TEACHER_SCAN_RESOLUTION,
    )
    y = torch.arange(
        start=-TEACHER_SCAN_SIZE[1] / 2.0,
        end=TEACHER_SCAN_SIZE[1] / 2.0 + 1.0e-5,
        step=TEACHER_SCAN_RESOLUTION,
    )
    grid_x, grid_y = torch.meshgrid(x, y, indexing="xy")
    local = torch.stack((grid_x.flatten() + TEACHER_SCAN_OFFSET[0], grid_y.flatten() + TEACHER_SCAN_OFFSET[1]), dim=-1)
    if local.shape[0] != TEACHER_SCAN_DIM:
        raise RuntimeError(f"scan grid has {local.shape[0]} points, expected {TEACHER_SCAN_DIM}")
    if device is not None:
        local = local.to(device=device)
    return local


def analytic_vault_height_scan(trunk_xy, trunk_yaw, trunk_z, box_center_w, box_half_xyz=None):
    """Height scan for the flat-ground + 1 m box scene, no RayCaster mesh required."""
    import torch

    local = teacher_scan_local_xy(device=trunk_xy.device).to(dtype=trunk_xy.dtype)
    cos_y = torch.cos(trunk_yaw)
    sin_y = torch.sin(trunk_yaw)
    world_x = trunk_xy[:, 0:1] + cos_y.unsqueeze(-1) * local[:, 0] - sin_y.unsqueeze(-1) * local[:, 1]
    world_y = trunk_xy[:, 1:2] + sin_y.unsqueeze(-1) * local[:, 0] + cos_y.unsqueeze(-1) * local[:, 1]
    hits = torch.stack((world_x, world_y), dim=-1)
    dummy_ground = torch.zeros(trunk_xy.shape[0], TEACHER_SCAN_DIM, device=trunk_xy.device, dtype=trunk_xy.dtype)
    ground_scan = (trunk_z.reshape(-1, 1) - 0.0 - TEACHER_SCAN_HEIGHT_OFFSET).expand_as(dummy_ground).clone()
    return overlay_box_on_height_scan(
        ground_scan,
        hits,
        trunk_z,
        box_center_w,
        box_half_xyz=box_half_xyz,
    )


def overlay_box_on_height_scan(
    ground_scan,
    hit_xy,
    sensor_z,
    box_center,
    box_half_xyz=None,
    height_offset: float = TEACHER_SCAN_HEIGHT_OFFSET,
    clip=TEACHER_SCAN_CLIP,
    invalid_value: float = TEACHER_SCAN_INVALID_VALUE,
):
    """Replace ground-scan cells whose hit xy falls in the box footprint.

    IsaacLab RayCaster can only warp one mesh, so the 1 m box is painted onto
    the flat-ground scan from the known AABB instead of a second mesh prim.
    Accepts torch tensors; shapes scan/hits ``(N, R)`` / ``(N, R, 2)``.
    """
    import torch

    scan = torch.as_tensor(ground_scan)
    xy = torch.as_tensor(hit_xy, device=scan.device, dtype=scan.dtype)
    z = torch.as_tensor(sensor_z, device=scan.device, dtype=scan.dtype).reshape(-1, 1)
    center = torch.as_tensor(box_center, device=scan.device, dtype=scan.dtype)
    if box_half_xyz is None:
        box_half_xyz = (T4_VAULT_BOX_SIZE[0] / 2.0, T4_VAULT_BOX_SIZE[1] / 2.0, T4_VAULT_BOX_SIZE[2] / 2.0)
    hx, hy, hz = (float(box_half_xyz[0]), float(box_half_xyz[1]), float(box_half_xyz[2]))
    inside = (
        (xy[..., 0] >= (center[:, 0:1] - hx))
        & (xy[..., 0] <= (center[:, 0:1] + hx))
        & (xy[..., 1] >= (center[:, 1:2] - hy))
        & (xy[..., 1] <= (center[:, 1:2] + hy))
    )
    box_top = center[:, 2:3] + hz
    box_scan = (z - box_top - float(height_offset)).clamp(clip[0], clip[1])
    painted = torch.where(inside, box_scan, scan)
    return torch.nan_to_num(painted, nan=invalid_value, posinf=invalid_value, neginf=invalid_value)

__all__ = [
    "G2_ACTION_DIM",
    "G2_ACTION_SCALE",
    "G2_POLICY_FORBIDDEN_SUBSTRINGS",
    "G2_POLICY_OBS_DIM",
    "G2_PROPRIO_FRAME_DIM",
    "G2_PROPRIO_HISTORY_LENGTH",
    "G2_SCAN_DIM",
    "G2_GAIT_AIR_RATIO",
    "G2_GAIT_CYCLE",
    "G2_GAIT_PHASE_OFFSET",
    "G2_SKILL_LIN_VEL_X",
    "G2_TEACHER_OBS_DIM",
    "PROPRIO_FIELDS",
    "TEACHER_SCAN_BODY",
    "TEACHER_SCAN_CLIP",
    "TEACHER_SCAN_HEIGHT_OFFSET",
    "TEACHER_SCAN_INVALID_VALUE",
    "TEACHER_SCAN_OFFSET",
    "TEACHER_SCAN_RESOLUTION",
    "TEACHER_SCAN_SIZE",
]
