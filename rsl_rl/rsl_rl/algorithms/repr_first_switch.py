"""Isaac-free switch rules for representation-first S12 distillation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

OCC_THRESHOLD = 0.5
GLOBAL_RATIO = 1.5
BASELINE_DROP = 0.5
BASELINE_START_ITER = 200
BASELINE_END_ITER = 400
DEFAULT_CAP_ITERS = 4000
DEFAULT_PATIENCE = 3
DEFAULT_MIN_COUNT = 8
DEFAULT_PROBE_INTERVAL = 100


@dataclass(frozen=True)
class ScanProbe:
    mse_global: float
    mse_stepping_stones: float
    mse_raised_pillars: float
    occ_agree_stepping_stones: float
    occ_agree_raised_pillars: float
    count_global: int
    count_stepping_stones: int
    count_raised_pillars: int

    def valid(self, min_count: int = DEFAULT_MIN_COUNT) -> bool:
        if self.count_global < min_count:
            return False
        if self.count_stepping_stones < min_count:
            return False
        if self.count_raised_pillars < min_count:
            return False
        return all(
            math.isfinite(value)
            for value in (
                self.mse_global,
                self.mse_stepping_stones,
                self.mse_raised_pillars,
                self.occ_agree_stepping_stones,
                self.occ_agree_raised_pillars,
            )
        )


@dataclass
class ReprSwitchState:
    phase: str = "representation"
    switch_iter: int | None = None
    switch_reason: str | None = None
    consecutive_hits: int = 0
    baseline_stone_mses: list[float] = field(default_factory=list)
    baseline_pillar_mses: list[float] = field(default_factory=list)
    last_probe: ScanProbe | None = None

    def baseline_medians(self) -> tuple[float | None, float | None]:
        if not self.baseline_stone_mses or not self.baseline_pillar_mses:
            return None, None
        return _median(self.baseline_stone_mses), _median(self.baseline_pillar_mses)

    def to_dict(self) -> dict:
        probe = None
        if self.last_probe is not None:
            probe = {
                "mse_global": self.last_probe.mse_global,
                "mse_stepping_stones": self.last_probe.mse_stepping_stones,
                "mse_raised_pillars": self.last_probe.mse_raised_pillars,
                "occ_agree_stepping_stones": self.last_probe.occ_agree_stepping_stones,
                "occ_agree_raised_pillars": self.last_probe.occ_agree_raised_pillars,
                "count_global": self.last_probe.count_global,
                "count_stepping_stones": self.last_probe.count_stepping_stones,
                "count_raised_pillars": self.last_probe.count_raised_pillars,
            }
        return {
            "phase": self.phase,
            "switch_iter": self.switch_iter,
            "switch_reason": self.switch_reason,
            "consecutive_hits": self.consecutive_hits,
            "baseline_stone_mses": list(self.baseline_stone_mses),
            "baseline_pillar_mses": list(self.baseline_pillar_mses),
            "last_probe": probe,
        }

    @classmethod
    def from_dict(cls, payload: dict | None) -> "ReprSwitchState":
        if not payload:
            return cls()
        probe_payload = payload.get("last_probe")
        probe = ScanProbe(**probe_payload) if isinstance(probe_payload, dict) else None
        return cls(
            phase=str(payload.get("phase", "representation")),
            switch_iter=payload.get("switch_iter"),
            switch_reason=payload.get("switch_reason"),
            consecutive_hits=int(payload.get("consecutive_hits", 0)),
            baseline_stone_mses=list(payload.get("baseline_stone_mses") or []),
            baseline_pillar_mses=list(payload.get("baseline_pillar_mses") or []),
            last_probe=probe,
        )


def occupancy_agreement(recon, teacher_scan, threshold: float = OCC_THRESHOLD):
    """Elementwise pit/support agreement in [0, 1]. ``recon``/``teacher_scan`` are tensors."""
    pred = recon >= threshold
    target = teacher_scan >= threshold
    return (pred == target).to(dtype=recon.dtype)


def probe_meets_switch_rule(probe: ScanProbe, baseline_stones: float, baseline_pillars: float) -> bool:
    if not math.isfinite(baseline_stones) or not math.isfinite(baseline_pillars):
        return False
    if baseline_stones <= 0.0 or baseline_pillars <= 0.0:
        return False
    if probe.mse_global <= 0.0:
        return False
    if probe.mse_stepping_stones > GLOBAL_RATIO * probe.mse_global:
        return False
    if probe.mse_raised_pillars > GLOBAL_RATIO * probe.mse_global:
        return False
    if probe.mse_stepping_stones > (1.0 - BASELINE_DROP) * baseline_stones:
        return False
    if probe.mse_raised_pillars > (1.0 - BASELINE_DROP) * baseline_pillars:
        return False
    return True


def observe_probe(
    state: ReprSwitchState,
    *,
    iteration: int,
    probe: ScanProbe | None,
    min_count: int = DEFAULT_MIN_COUNT,
    cap_iters: int = DEFAULT_CAP_ITERS,
    patience: int = DEFAULT_PATIENCE,
    baseline_start: int = BASELINE_START_ITER,
    baseline_end: int = BASELINE_END_ITER,
) -> ReprSwitchState:
    """Advance representation→action at most once."""
    if state.phase != "representation":
        return state

    next_state = ReprSwitchState(
        phase=state.phase,
        switch_iter=state.switch_iter,
        switch_reason=state.switch_reason,
        consecutive_hits=state.consecutive_hits,
        baseline_stone_mses=list(state.baseline_stone_mses),
        baseline_pillar_mses=list(state.baseline_pillar_mses),
        last_probe=probe if probe is not None else state.last_probe,
    )
    probe_valid = probe is not None and probe.valid(min_count)
    in_baseline = baseline_start <= iteration <= baseline_end
    if probe_valid:
        assert probe is not None
        if in_baseline:
            next_state.baseline_stone_mses.append(probe.mse_stepping_stones)
            next_state.baseline_pillar_mses.append(probe.mse_raised_pillars)
            next_state.consecutive_hits = 0
        else:
            stones, pillars = next_state.baseline_medians()
            if stones is None or pillars is None:
                next_state.consecutive_hits = 0
            elif probe_meets_switch_rule(probe, stones, pillars):
                next_state.consecutive_hits += 1
            else:
                next_state.consecutive_hits = 0
    else:
        next_state.consecutive_hits = 0

    if next_state.consecutive_hits >= patience:
        next_state.phase = "action"
        next_state.switch_iter = int(iteration)
        next_state.switch_reason = "metric"
        return next_state

    if iteration >= cap_iters:
        next_state.phase = "action"
        next_state.switch_iter = int(iteration)
        stones, pillars = next_state.baseline_medians()
        next_state.switch_reason = "cap" if stones is not None and pillars is not None else "baseline_invalid"
        return next_state

    return next_state


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return float(ordered[mid])
    return 0.5 * (float(ordered[mid - 1]) + float(ordered[mid]))
