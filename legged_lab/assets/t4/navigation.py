"""T4 深度 locomotion 训练/部署共用的速度命令合同。"""

from __future__ import annotations

import math

import numpy as np


COMMAND_RANGES = {"vx": (-0.6, 1.0), "vy": (-0.5, 0.5), "yaw": (-1.57, 1.57)}
HEADING_STIFFNESS = 0.5
TURN_IN_PLACE_YAW = math.radians(50.0)
NAV_LOOKAHEAD = 1.4


def wrap_to_pi(angle: float) -> float:
    """Wrap an angle in radians to ``(-pi, pi]``."""
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def heading_velocity_command(
    root_xy: np.ndarray,
    yaw: float,
    goal_xy: np.ndarray,
    cruise_vx: float,
    arrive: float = 0.45,
) -> np.ndarray:
    """构造训练时 ``heading_command=True`` 的 ``[vx, 0, wz]``。"""
    delta = np.asarray(goal_xy, dtype=np.float64) - np.asarray(root_xy, dtype=np.float64)
    dist = float(np.linalg.norm(delta))
    if dist <= arrive:
        return np.zeros(3, dtype=np.float64)
    desired_yaw = math.atan2(delta[1], delta[0])
    yaw_err = wrap_to_pi(desired_yaw - yaw)
    yaw_rate = float(np.clip(HEADING_STIFFNESS * yaw_err, *COMMAND_RANGES["yaw"]))
    if abs(yaw_err) > TURN_IN_PLACE_YAW:
        vx = 0.0
    else:
        vx = float(np.clip(cruise_vx * max(0.0, math.cos(yaw_err)), 0.0, COMMAND_RANGES["vx"][1]))
    return np.array([vx, 0.0, yaw_rate], dtype=np.float64)


def polyline_lookahead(root_xy: np.ndarray, waypoints: np.ndarray, lookahead: float = NAV_LOOKAHEAD) -> np.ndarray:
    """Return a short centerline carrot so lateral drift produces a useful yaw correction."""
    root_xy = np.asarray(root_xy, dtype=np.float64)
    pts = np.asarray(waypoints, dtype=np.float64).reshape(-1, 2)
    if len(pts) == 1:
        pts = np.vstack([[min(root_xy[0], pts[0, 0]), pts[0, 1]], pts[0]])
    segments: list[tuple[float, float, np.ndarray, np.ndarray]] = []
    acc = 0.0
    best_dist = math.inf
    best_s = 0.0
    for index in range(len(pts) - 1):
        start = pts[index]
        chord = pts[index + 1] - start
        length = float(np.linalg.norm(chord))
        if length < 1e-6:
            continue
        ratio = float(np.clip(np.dot(root_xy - start, chord) / (length * length), 0.0, 1.0))
        projected = start + ratio * chord
        dist = float(np.linalg.norm(root_xy - projected))
        if dist < best_dist:
            best_dist = dist
            best_s = acc + ratio * length
        segments.append((acc, length, start, chord))
        acc += length
    if not segments:
        return pts[-1]
    carrot = max(0.7, float(lookahead) - 1.1 * min(best_dist, 0.8))
    target_s = min(best_s + carrot, acc)
    for acc0, length, start, chord in segments:
        if target_s <= acc0 + length:
            return start + ((target_s - acc0) / length) * chord
    return pts[-1]


class CourseNavigator:
    """Centerline carrot follower using the frozen T4 heading-command contract."""

    def __init__(
        self,
        waypoints: np.ndarray,
        cruise_vx: float = 0.55,
        reach: float = 0.55,
        lookahead: float = NAV_LOOKAHEAD,
        max_yaw_rate: float | None = None,
    ):
        if len(waypoints) < 1:
            raise ValueError("navigator needs at least one waypoint")
        self.waypoints = np.asarray(waypoints, dtype=np.float64)
        self.cruise_vx = float(cruise_vx)
        self.reach = float(reach)
        self.lookahead = float(lookahead)
        self.max_yaw_rate = None if max_yaw_rate is None else abs(float(max_yaw_rate))
        self.index = 0

    def reset(self) -> None:
        self.index = 0

    def _advance(self, root_xy: np.ndarray) -> None:
        while self.index < len(self.waypoints) - 1:
            if root_xy[0] + 0.15 >= self.waypoints[self.index, 0]:
                self.index += 1
            else:
                break

    def command(self, root_xy: np.ndarray, yaw: float) -> np.ndarray:
        root_xy = np.asarray(root_xy, dtype=np.float64)
        self._advance(root_xy)
        carrot = polyline_lookahead(root_xy, self.waypoints, self.lookahead)
        command = heading_velocity_command(root_xy, yaw, carrot, self.cruise_vx, arrive=0.40)
        if self.max_yaw_rate is not None:
            command[2] = np.clip(command[2], -self.max_yaw_rate, self.max_yaw_rate)
        return command

    @property
    def current_waypoint(self) -> np.ndarray:
        return self.waypoints[min(self.index, len(self.waypoints) - 1)]
