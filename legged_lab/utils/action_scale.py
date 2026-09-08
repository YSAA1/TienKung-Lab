"""Convert dimensionless position actions to nominal actuator torque fractions."""

import math

import torch


def effort_scaled_action_scale(stiffness: torch.Tensor, effort_limit: torch.Tensor, fraction: float) -> torch.Tensor:
    """For zero position/velocity error, action=1 requests fraction * torque limit.

    This is a nominal PD target conversion, not a bound on actual torque: velocity
    feedback, gravity and target tracking error still contribute to actuator load.
    Inputs and returned scales use simulator joint order. Resolve once, before DR.
    """
    if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise ValueError("action effort fraction must be finite and in (0, 1]")
    if stiffness.shape != effort_limit.shape:
        raise ValueError("stiffness and effort limits must have identical joint layout")
    valid = torch.isfinite(stiffness) & torch.isfinite(effort_limit) & (stiffness > 0) & (effort_limit > 0)
    if not bool(valid.all()):
        raise ValueError("effort-scaled position actions require finite positive stiffness and effort limits")
    return (fraction * effort_limit / stiffness).detach().clone()
