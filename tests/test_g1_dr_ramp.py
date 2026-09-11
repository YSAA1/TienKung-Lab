# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

"""Unit tests for the pure vital_v3 DR ramp math (no Isaac import needed)."""

import importlib.util
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def _load_ramp_module():
    spec = importlib.util.spec_from_file_location("legged_lab.mdp.ramp", ROOT / "legged_lab/mdp/ramp.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ramp = _load_ramp_module()
covers_all_bodies = ramp.covers_all_bodies
interpolate_range = ramp.interpolate_range
ramp_fraction = ramp.ramp_fraction
ramped_time_lags = ramp.ramped_time_lags
scale_range_around_nominal = ramp.scale_range_around_nominal


def test_covers_all_bodies_accepts_both_resolve_spellings():
    assert covers_all_bodies(slice(None), 29)
    assert covers_all_bodies(list(range(29)), 29)
    assert not covers_all_bodies(list(range(28)), 29)
    assert not covers_all_bodies(slice(0, 28), 29)


def test_ramp_fraction_is_monotone_and_clamped():
    assert ramp_fraction(0, 100) == 0.0
    assert ramp_fraction(50, 100) == 0.5
    assert ramp_fraction(100, 100) == 1.0
    assert ramp_fraction(10_000, 100) == 1.0
    assert ramp_fraction(-5, 100) == 0.0
    with pytest.raises(ValueError):
        ramp_fraction(1, 0)


def test_interpolate_range_moves_both_bounds():
    assert interpolate_range((0.6, 1.0), (0.6, 1.2), 0.0) == (0.6, 1.0)
    assert interpolate_range((0.6, 1.0), (0.6, 1.2), 1.0) == (0.6, 1.2)
    assert interpolate_range((0.4, 0.8), (0.5, 1.0), 0.5) == pytest.approx((0.45, 0.9))
    with pytest.raises(ValueError):
        interpolate_range((0.0, 1.0), (0.0, 1.0), 1.5)


def test_scale_range_around_nominal_ramps_from_identity():
    assert scale_range_around_nominal((0.9, 1.1), 0.0) == (1.0, 1.0)
    assert scale_range_around_nominal((0.9, 1.1), 1.0) == (0.9, 1.1)
    assert scale_range_around_nominal((0.9, 1.1), 0.5) == pytest.approx((0.95, 1.05))


def test_ramped_time_lags_delay_probability_grows_with_fraction():
    generator = torch.Generator().manual_seed(0)
    draws = torch.rand(200_000, generator=generator)
    assert (ramped_time_lags(draws, 0, 1, 0.0) == 0).all()
    # deliberate slow-in: the first half of the ramp carries no excess latency
    assert (ramped_time_lags(draws, 0, 1, 0.4) == 0).all()
    full = ramped_time_lags(draws, 0, 1, 1.0)
    assert set(full.unique().tolist()) == {0, 1}
    assert float((full == 1).float().mean()) == pytest.approx(0.5, abs=0.01)
    # excess latency eases in monotonically across the second half
    fractions = [0.6, 0.8, 1.0]
    rates = [float((ramped_time_lags(draws, 0, 1, f) == 1).float().mean()) for f in fractions]
    assert rates == sorted(rates)
    assert rates[0] > 0.0
    with pytest.raises(ValueError):
        ramped_time_lags(draws, 2, 1, 1.0)


def _load_scan_occlusion_module():
    spec = importlib.util.spec_from_file_location(
        "legged_lab.locomotion.mdp.scan_occlusion",
        ROOT / "legged_lab/locomotion/mdp/scan_occlusion.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


occlusion = _load_scan_occlusion_module()
apply_scan_occlusion = occlusion.apply_scan_occlusion
sample_column_band_masks = occlusion.sample_column_band_masks


def test_sample_column_band_masks_zero_probability_is_empty():
    masks = sample_column_band_masks(64, (15, 13), (0.1, 0.3), 0.0, torch.device("cpu"))
    assert masks.shape == (64, 15 * 13)
    assert masks.dtype == torch.bool
    assert not bool(torch.any(masks))


def test_sample_column_band_masks_full_probability_max_band_covers_all():
    masks = sample_column_band_masks(64, (15, 13), (1.0, 1.0), 1.0, torch.device("cpu"))
    assert bool(torch.all(masks))


def test_sample_column_band_masks_bands_are_lateral_rows_in_scan_layout():
    # The scanner flattens y-outer/x-inner: flat = iy * nx + ix (see
    # tests/test_t4_observation_contracts.py). A lateral band therefore masks a
    # contiguous run of iy across every forward ix.
    torch.manual_seed(7)
    nx, ny = 15, 13
    masks = sample_column_band_masks(512, (nx, ny), (0.1, 0.5), 0.5, torch.device("cpu"))
    occluded_envs = masks.any(dim=1)
    assert 0.1 < float(occluded_envs.float().mean()) < 0.9
    grid = masks.view(-1, ny, nx)
    for env in grid[occluded_envs]:
        # every lateral row is fully masked or fully clear: the band spans all ix
        assert bool(torch.equal(env.all(dim=1), env.any(dim=1)))
        iy = env.any(dim=1).nonzero().flatten()
        assert 0 < len(iy) <= ny
        if len(iy) > 1:
            assert bool((iy[1:] - iy[:-1] == 1).all())


def test_apply_scan_occlusion_preserves_unmasked_and_fills_within_clip():
    torch.manual_seed(11)
    scan = torch.zeros(8, 15 * 13)
    masks = torch.zeros(8, 15 * 13, dtype=torch.bool)
    masks[2, 4:9] = True
    out = apply_scan_occlusion(scan, masks, -1.0, 1.0)
    assert torch.equal(out[~masks], scan[~masks])
    assert bool((out[masks] > -1.0).all()) and bool((out[masks] < 1.0).all())
    assert not torch.equal(out[masks], scan[masks])


def test_apply_scan_occlusion_empty_mask_is_identity():
    scan = torch.randn(4, 15 * 13)
    masks = torch.zeros(4, 15 * 13, dtype=torch.bool)
    out = apply_scan_occlusion(scan, masks, -1.0, 1.0)
    assert torch.equal(out, scan)
