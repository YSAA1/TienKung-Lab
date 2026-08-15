"""Pure-Python 1 m vault corridor layout and strict evaluator.

Isaac-free so contract tests run anywhere. Geometry is derived from
``vault_contract`` (fixed 1 m box). Success semantics are the zip evaluator's
sustained crossing + stable landing, plus corridor / ordered-gate checks that
reject side-bypass and skipped / reversed gates.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from legged_lab.assets.t4.vault_contract import (
    T4_VAULT_ACTION_SCALE,
    T4_VAULT_BOX_POS,
    T4_VAULT_BOX_ROT,
    T4_VAULT_BOX_SIZE,
)

EVALUATOR_NAME = "t4_vault_corridor_v1"
SUCCESS_DEFINITION = (
    "ordered approach+exit gates inside the 1 m corridor, sustained crossing of "
    "box_back_x, then a stable upright landing past the box, with no hard violation"
)

T4_VAULT_MIN_CROSSING_STEPS = 10
T4_VAULT_LANDING_STEPS = 25
T4_VAULT_LANDING_HEIGHT_RANGE = (0.65, 1.05)
T4_VAULT_MIN_ROOT_UPRIGHT = 0.9
T4_VAULT_MAX_ROOT_LIN_SPEED = 0.5
T4_VAULT_MAX_ROOT_ANG_SPEED = 1.0

T4_VAULT_APPROACH_CLEARANCE = 0.30
T4_VAULT_EXIT_CLEARANCE = 0.40
T4_VAULT_WALL_THICKNESS = 0.10
T4_VAULT_WALL_HEIGHT = 1.50
T4_VAULT_WALL_X_PAD = 1.00

REPORT_FIELDS = (
    "evaluator",
    "lineage",
    "seed",
    "checkpoint",
    "bucket",
    "policy",
    "success_definition",
    "summary",
    "episodes",
    "layout",
)


@dataclass(frozen=True)
class VaultCorridorLayout:
    """Axis-aligned 1 m box corridor used by the evaluator and eval scene."""

    box_size: tuple[float, float, float]
    box_pos: tuple[float, float, float]
    box_rot: tuple[float, float, float, float]
    box_aabb: tuple[tuple[float, float, float], tuple[float, float, float]]
    box_front_x: float
    box_back_x: float
    corridor_y_min: float
    corridor_y_max: float
    approach_gate_x: float
    exit_gate_x: float
    wall_aabbs: tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...]


@dataclass(frozen=True)
class VaultTrajectoryResult:
    success: bool
    reasons: tuple[str, ...]
    crossing_start_step: int | None
    landing_start_step: int | None
    approach_gate_step: int | None
    exit_gate_step: int | None
    start_x: float
    max_x: float
    final_x: float
    hard_violation: bool
    termination_step: int | None
    termination_terms: tuple[str, ...]
    forbidden_contact_count: int
    hard_limit_count: int


@dataclass(frozen=True)
class VaultBatchResult:
    total: int
    successes: int
    success_rate: float
    episodes: tuple[VaultTrajectoryResult, ...]


def canonical_vault_corridor_layout(
    box_size: Sequence[float] = T4_VAULT_BOX_SIZE,
    box_pos: Sequence[float] = T4_VAULT_BOX_POS,
    box_rot: Sequence[float] = T4_VAULT_BOX_ROT,
) -> VaultCorridorLayout:
    """Build the G1/G2 1 m corridor: box fills the lane so side-bypass is impossible."""
    size = (float(box_size[0]), float(box_size[1]), float(box_size[2]))
    pos = (float(box_pos[0]), float(box_pos[1]), float(box_pos[2]))
    rot = (float(box_rot[0]), float(box_rot[1]), float(box_rot[2]), float(box_rot[3]))
    hx, hy, hz = size[0] / 2.0, size[1] / 2.0, size[2] / 2.0
    box_min = (pos[0] - hx, pos[1] - hy, pos[2] - hz)
    box_max = (pos[0] + hx, pos[1] + hy, pos[2] + hz)
    box_front_x = box_min[0]
    box_back_x = box_max[0]
    corridor_y_min = box_min[1]
    corridor_y_max = box_max[1]
    approach_gate_x = box_front_x - T4_VAULT_APPROACH_CLEARANCE
    exit_gate_x = box_back_x + T4_VAULT_EXIT_CLEARANCE
    wall_x0 = approach_gate_x - T4_VAULT_WALL_X_PAD
    wall_x1 = exit_gate_x + T4_VAULT_WALL_X_PAD
    ht = T4_VAULT_WALL_THICKNESS
    left = ((wall_x0, corridor_y_min - ht, 0.0), (wall_x1, corridor_y_min, T4_VAULT_WALL_HEIGHT))
    right = ((wall_x0, corridor_y_max, 0.0), (wall_x1, corridor_y_max + ht, T4_VAULT_WALL_HEIGHT))
    return VaultCorridorLayout(
        box_size=size,
        box_pos=pos,
        box_rot=rot,
        box_aabb=(box_min, box_max),
        box_front_x=box_front_x,
        box_back_x=box_back_x,
        corridor_y_min=corridor_y_min,
        corridor_y_max=corridor_y_max,
        approach_gate_x=approach_gate_x,
        exit_gate_x=exit_gate_x,
        wall_aabbs=(left, right),
    )


def wall_spawn_pose(
    aabb: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return ``(size, center)`` for an IsaacLab cuboid matching ``aabb``."""
    (x0, y0, z0), (x1, y1, z1) = aabb
    size = (x1 - x0, y1 - y0, z1 - z0)
    center = ((x0 + x1) / 2.0, (y0 + y1) / 2.0, (z0 + z1) / 2.0)
    return size, center


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    path = Path(path)
    if not path.is_file():
        return ""
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _first_sustained_run(mask: Sequence[bool], length: int, start: int = 0) -> int | None:
    if length <= 0:
        raise ValueError(f"Sustained-run length must be positive, got {length}.")
    run_start = start
    run_length = 0
    for index in range(start, len(mask)):
        if mask[index]:
            if run_length == 0:
                run_start = index
            run_length += 1
            if run_length >= length:
                return run_start
        else:
            run_length = 0
    return None


def _first_forward_gate(
    root_x: Sequence[float], root_y: Sequence[float], gate_x: float, y_min: float, y_max: float
) -> int | None:
    for index in range(1, len(root_x)):
        if float(root_x[index - 1]) < gate_x <= float(root_x[index]) and y_min <= float(root_y[index]) <= y_max:
            return index
    return None


def evaluate_vault_trajectory(
    root_x: Sequence[float],
    root_y: Sequence[float],
    root_z: Sequence[float],
    root_upright: Sequence[float],
    root_lin_speed: Sequence[float],
    root_ang_speed: Sequence[float],
    layout: VaultCorridorLayout,
    min_crossing_steps: int = T4_VAULT_MIN_CROSSING_STEPS,
    landing_steps: int = T4_VAULT_LANDING_STEPS,
    landing_height_range: tuple[float, float] = T4_VAULT_LANDING_HEIGHT_RANGE,
    min_root_upright: float = T4_VAULT_MIN_ROOT_UPRIGHT,
    max_root_lin_speed: float = T4_VAULT_MAX_ROOT_LIN_SPEED,
    max_root_ang_speed: float = T4_VAULT_MAX_ROOT_ANG_SPEED,
    hard_violation: bool | Sequence[bool] = False,
    termination_step: int | None = None,
    termination_terms: Sequence[str] = (),
    forbidden_contact_count: int = 0,
    hard_limit_count: int = 0,
) -> VaultTrajectoryResult:
    """Require a real approach, corridor stay, sustained crossing, and stable landing."""
    lengths = {len(root_x), len(root_y), len(root_z), len(root_upright), len(root_lin_speed), len(root_ang_speed)}
    if len(lengths) != 1:
        raise ValueError("All root trajectory fields must contain the same number of steps.")
    if len(root_x) == 0:
        raise ValueError("A vault trajectory must contain at least one step.")
    landing_height_min, landing_height_max = landing_height_range
    if landing_height_min > landing_height_max:
        raise ValueError("landing_height_range must be ordered low-to-high.")

    reasons: list[str] = []
    if isinstance(hard_violation, bool):
        has_hard_violation = hard_violation
    else:
        has_hard_violation = any(bool(value) for value in hard_violation)
    if has_hard_violation:
        reasons.append("hard_violation")
    if int(forbidden_contact_count) > 0:
        reasons.append("forbidden_contact")
    if int(hard_limit_count) > 0:
        reasons.append("hard_limit")
    if float(root_x[0]) >= layout.box_back_x:
        reasons.append("start_already_crossed")

    left_corridor = any(not (layout.corridor_y_min <= float(y) <= layout.corridor_y_max) for y in root_y)
    if left_corridor:
        reasons.append("left_corridor")

    approach_step = _first_forward_gate(
        root_x, root_y, layout.approach_gate_x, layout.corridor_y_min, layout.corridor_y_max
    )
    exit_step = _first_forward_gate(root_x, root_y, layout.exit_gate_x, layout.corridor_y_min, layout.corridor_y_max)
    if approach_step is not None and exit_step is not None and exit_step < approach_step:
        reasons.append("reversed_gate")
    elif exit_step is not None and approach_step is None:
        reasons.append("skipped_gate")

    crossed = [float(x) >= layout.box_back_x for x in root_x]
    crossing_start = _first_sustained_run(crossed, min_crossing_steps)
    if crossing_start is None:
        reasons.append("crossing_not_sustained")

    landing_start = None
    if crossing_start is not None:
        landing_search_start = crossing_start + min_crossing_steps
        stable_landing = [
            float(x) >= layout.box_back_x
            and landing_height_min <= float(z) <= landing_height_max
            and float(upright) >= min_root_upright
            and float(lin_speed) <= max_root_lin_speed
            and float(ang_speed) <= max_root_ang_speed
            and layout.corridor_y_min <= float(y) <= layout.corridor_y_max
            for x, y, z, upright, lin_speed, ang_speed in zip(
                root_x, root_y, root_z, root_upright, root_lin_speed, root_ang_speed, strict=True
            )
        ]
        landing_start = _first_sustained_run(stable_landing, landing_steps, start=landing_search_start)
    if landing_start is None:
        reasons.append("no_stable_landing")

    return VaultTrajectoryResult(
        success=not reasons,
        reasons=tuple(reasons),
        crossing_start_step=crossing_start,
        landing_start_step=landing_start,
        approach_gate_step=approach_step,
        exit_gate_step=exit_step,
        start_x=float(root_x[0]),
        max_x=max(float(x) for x in root_x),
        final_x=float(root_x[-1]),
        hard_violation=has_hard_violation,
        termination_step=termination_step,
        termination_terms=tuple(termination_terms),
        forbidden_contact_count=int(forbidden_contact_count),
        hard_limit_count=int(hard_limit_count),
    )


def _as_step_major(values, num_envs_hint: int | None = None):
    rows = values.tolist() if hasattr(values, "tolist") else values
    if len(rows) == 0:
        raise ValueError("A vault batch must contain at least one step.")
    if not isinstance(rows[0], (list, tuple)):
        rows = [[value] for value in rows]
    return rows


def evaluate_vault_batch(
    root_x,
    root_y,
    root_z,
    root_upright,
    root_lin_speed,
    root_ang_speed,
    layout: VaultCorridorLayout | None = None,
    min_crossing_steps: int = T4_VAULT_MIN_CROSSING_STEPS,
    landing_steps: int = T4_VAULT_LANDING_STEPS,
    landing_height_range: tuple[float, float] = T4_VAULT_LANDING_HEIGHT_RANGE,
    min_root_upright: float = T4_VAULT_MIN_ROOT_UPRIGHT,
    max_root_lin_speed: float = T4_VAULT_MAX_ROOT_LIN_SPEED,
    max_root_ang_speed: float = T4_VAULT_MAX_ROOT_ANG_SPEED,
    hard_violation=None,
    termination_steps=None,
    termination_terms=None,
    forbidden_contact_counts=None,
    hard_limit_counts=None,
) -> VaultBatchResult:
    """Evaluate arrays shaped ``[steps, envs]`` and aggregate per-env results."""
    if layout is None:
        layout = canonical_vault_corridor_layout()
    root_x_values = _as_step_major(root_x)
    root_y_values = _as_step_major(root_y)
    root_z_values = _as_step_major(root_z)
    root_upright_values = _as_step_major(root_upright)
    root_lin_speed_values = _as_step_major(root_lin_speed)
    root_ang_speed_values = _as_step_major(root_ang_speed)
    fields = (
        root_x_values,
        root_y_values,
        root_z_values,
        root_upright_values,
        root_lin_speed_values,
        root_ang_speed_values,
    )
    if len({len(values) for values in fields}) != 1:
        raise ValueError("All root trajectory fields must contain the same number of steps.")
    num_envs = len(root_x_values[0])
    all_rows = [row for values in fields for row in values]
    if num_envs == 0 or any(len(row) != num_envs for row in all_rows):
        raise ValueError("All batch rows must have the same non-zero environment count.")

    violation_values = None
    if hard_violation is not None:
        violation_values = hard_violation.tolist() if hasattr(hard_violation, "tolist") else hard_violation
        if violation_values and not isinstance(violation_values[0], (list, tuple)):
            if len(violation_values) == num_envs:
                violation_values = [violation_values]
            else:
                violation_values = [[value] for value in violation_values]

    if termination_steps is None:
        termination_step_values = [-1] * num_envs
    else:
        termination_step_values = (
            termination_steps.tolist() if hasattr(termination_steps, "tolist") else list(termination_steps)
        )
    if len(termination_step_values) != num_envs:
        raise ValueError("termination_steps must contain one value per environment.")
    if termination_terms is None:
        termination_terms = [()] * num_envs
    if len(termination_terms) != num_envs:
        raise ValueError("termination_terms must contain one sequence per environment.")
    if forbidden_contact_counts is None:
        forbidden_contact_counts = [0] * num_envs
    if hard_limit_counts is None:
        hard_limit_counts = [0] * num_envs

    episodes = []
    for env_index in range(num_envs):
        env_violation = False
        if violation_values is not None:
            env_violation = any(bool(row[env_index]) for row in violation_values)
        term_step = int(termination_step_values[env_index])
        episodes.append(
            evaluate_vault_trajectory(
                [row[env_index] for row in root_x_values],
                [row[env_index] for row in root_y_values],
                [row[env_index] for row in root_z_values],
                [row[env_index] for row in root_upright_values],
                [row[env_index] for row in root_lin_speed_values],
                [row[env_index] for row in root_ang_speed_values],
                layout=layout,
                min_crossing_steps=min_crossing_steps,
                landing_steps=landing_steps,
                landing_height_range=landing_height_range,
                min_root_upright=min_root_upright,
                max_root_lin_speed=max_root_lin_speed,
                max_root_ang_speed=max_root_ang_speed,
                hard_violation=env_violation,
                termination_step=None if term_step < 0 else term_step,
                termination_terms=termination_terms[env_index],
                forbidden_contact_count=int(forbidden_contact_counts[env_index]),
                hard_limit_count=int(hard_limit_counts[env_index]),
            )
        )
    successes = sum(result.success for result in episodes)
    return VaultBatchResult(
        total=num_envs,
        successes=successes,
        success_rate=successes / num_envs,
        episodes=tuple(episodes),
    )


def layout_as_dict(layout: VaultCorridorLayout) -> dict:
    return {
        "box_size": list(layout.box_size),
        "box_pos": list(layout.box_pos),
        "box_rot": list(layout.box_rot),
        "box_front_x": layout.box_front_x,
        "box_back_x": layout.box_back_x,
        "corridor_y_min": layout.corridor_y_min,
        "corridor_y_max": layout.corridor_y_max,
        "approach_gate_x": layout.approach_gate_x,
        "exit_gate_x": layout.exit_gate_x,
        "wall_aabbs": [list(map(list, aabb)) for aabb in layout.wall_aabbs],
    }


def build_evaluation_report(
    result: VaultBatchResult,
    *,
    checkpoint: str,
    motion: str,
    task: str,
    seed: int,
    git_head: str,
    git_dirty: bool,
    git_dirty_diff_sha256: str,
    bucket: str = "g1_fixed_1m",
    policy: str = "checkpoint",
    action_scale: float = T4_VAULT_ACTION_SCALE,
    layout: VaultCorridorLayout | None = None,
    checkpoint_sha256: str | None = None,
) -> dict:
    """Build the lineage-bearing JSON payload required by the V2 contract."""
    if layout is None:
        layout = canonical_vault_corridor_layout()
    failure_reasons: dict[str, int] = {}
    forbidden = 0
    hard_limits = 0
    hard_violations = 0
    for episode in result.episodes:
        forbidden += episode.forbidden_contact_count
        hard_limits += episode.hard_limit_count
        hard_violations += int(episode.hard_violation)
        for reason in episode.reasons:
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
    report = {
        "evaluator": EVALUATOR_NAME,
        "lineage": {
            "git_head": git_head,
            "git_dirty": git_dirty,
            "git_dirty_diff_sha256": git_dirty_diff_sha256,
            "task": task,
            "motion": motion,
            "motion_sha256": sha256_file(motion),
            "checkpoint": checkpoint,
            "checkpoint_sha256": checkpoint_sha256 if checkpoint_sha256 is not None else sha256_file(checkpoint),
            "action_scale": action_scale,
        },
        "seed": seed,
        "checkpoint": checkpoint,
        "bucket": bucket,
        "policy": policy,
        "success_definition": SUCCESS_DEFINITION,
        "summary": {
            "successes": result.successes,
            "failures": result.total - result.successes,
            "success_rate": result.success_rate,
            "failure_reasons": failure_reasons,
            "forbidden_contact_count": forbidden,
            "hard_limit_count": hard_limits,
            "hard_violation_count": hard_violations,
        },
        "episodes": [asdict(episode) for episode in result.episodes],
        "layout": layout_as_dict(layout),
    }
    missing = [field for field in REPORT_FIELDS if field not in report]
    if missing:
        raise RuntimeError(f"Evaluator report is missing required fields: {missing}")
    return report
