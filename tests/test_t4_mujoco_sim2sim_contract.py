"""Isaac MuJoCo sim2sim contract: PD assign, friction, vault contact surfaces."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from legged_lab.assets.t4.constants import T4_JOINT_NAMES
from legged_lab.assets.t4.mujoco_sim2sim import (
    ISAAC_FOOT_BOXES,
    ISAAC_FRICTION,
    apply_isaac_contact_friction,
    apply_isaac_pd,
    isaac_pd_gains,
    rewrite_vault_mjcf,
)
from legged_lab.assets.t4.vault_contract import T4_VAULT_BOX_POS, T4_VAULT_BOX_SIZE

ROOT = Path(__file__).resolve().parents[1]
MJCF = ROOT / "legged_lab/assets/t4/mjcf/t4_std.xml"
MESHDIR = ROOT / "legged_lab/assets/t4/meshes"
T4_PY = ROOT / "legged_lab/assets/t4/t4.py"
ZL_T4_MJCF = (
    ROOT
    / "zl_deploy/install/mj_sim/share/mj_sim/description/T4_std_add_head/xml/t4_std_add_head.xml"
)
mujoco = pytest.importorskip("mujoco")


def _load_base():
    xml = MJCF.read_text().replace('meshdir="../meshes/"', f'meshdir="{MESHDIR}"')
    return mujoco.MjModel.from_xml_string(xml)


def _load_vault():
    xml = rewrite_vault_mjcf(
        MJCF.read_text(),
        meshdir=MESHDIR,
        box_pos=T4_VAULT_BOX_POS,
        box_size=T4_VAULT_BOX_SIZE,
    )
    return mujoco.MjModel.from_xml_string(xml)


def test_isaac_pd_gains_match_t4_py_literals() -> None:
    src = T4_PY.read_text()
    assert '"J_arm_[lr]_0[1-5]": 20.0' in src
    assert '"J_arm_[lr]_0[6-7]": 10.0' in src
    assert "damping=1.0" in src
    assert "stiffness=50.0" in src and "damping=2.0" in src
    assert '"J_hip_[lr]_pitch": 100.0' in src
    assert '"J_hip_[lr]_pitch": 4.0' in src
    assert "stiffness=100.0" in src
    assert '"J_ankle_[lr]_pitch": 80.0' in src
    assert '"J_ankle_[lr]_roll": 20.0' in src
    assert isaac_pd_gains("J_arm_l_01") == (20.0, 1.0, 36.0)
    assert isaac_pd_gains("J_arm_r_07") == (10.0, 1.0, 12.0)
    assert isaac_pd_gains("J_waist_yaw") == (50.0, 2.0, 120.0)
    assert isaac_pd_gains("J_hip_l_pitch") == (100.0, 4.0, 130.0)
    assert isaac_pd_gains("J_hip_r_yaw") == (50.0, 2.0, 120.0)
    assert isaac_pd_gains("J_knee_l_pitch") == (100.0, 4.0, 130.0)
    assert isaac_pd_gains("J_ankle_l_pitch") == (80.0, 4.0, 72.0)
    assert isaac_pd_gains("J_ankle_r_roll") == (20.0, 1.0, 72.0)


def test_apply_isaac_pd_assigns_kd_and_clears_frictionloss() -> None:
    model = _load_base()
    hip = int(model.jnt_dofadr[model.joint("J_hip_l_pitch").id])
    assert model.dof_damping[hip] == pytest.approx(0.05)
    assert model.dof_frictionloss[hip] == pytest.approx(0.1)
    apply_isaac_pd(model)
    for name in T4_JOINT_NAMES:
        kp, kd, effort = isaac_pd_gains(name)
        dof = int(model.jnt_dofadr[model.joint(name).id])
        act = model.actuator(f"M{name[1:]}")
        assert model.dof_damping[dof] == pytest.approx(kd)
        assert model.dof_frictionloss[dof] == pytest.approx(0.0)
        assert act.gainprm[0] == pytest.approx(kp)
        assert act.biasprm[1] == pytest.approx(-kp)
        assert tuple(act.forcerange) == pytest.approx((-effort, effort))


def test_t4_std_ground_has_isaac_friction() -> None:
    text = MJCF.read_text()
    assert 'name="ground"' in text
    assert 'condim="1"' not in text.split('name="ground"', 1)[1].split("/>", 1)[0]
    model = _load_base()
    apply_isaac_contact_friction(model)
    gid = int(model.geom("ground").id)
    assert int(model.geom_condim[gid]) == 3
    assert np.allclose(model.geom_friction[gid], ISAAC_FRICTION)


@pytest.mark.skipif(
    not ZL_T4_MJCF.exists(),
    reason="zl_deploy T4_std_add_head.xml not present on this host (zhuoqun deploy parity check)",
)
def test_zl_t4_sim_ground_matches_direct_sim2sim_friction() -> None:
    model = mujoco.MjModel.from_xml_path(str(ZL_T4_MJCF))
    gid = int(model.geom("ground").id)
    assert int(model.geom_condim[gid]) == 3
    assert np.allclose(model.geom_friction[gid], ISAAC_FRICTION)
    world_geoms = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) == 0
    ]
    assert world_geoms == ["ground"]


def test_direct_and_zl_stair_probe_geometries_match() -> None:
    from legged_lab.scripts import eval_t4_depth_plant_gap as gap_eval
    from legged_lab.scripts import sim2sim_t4_depth_student as sim2sim

    direct = mujoco.MjModel.from_xml_path(sim2sim.build_model_xml(course="stairs"))
    zl = mujoco.MjModel.from_xml_path(gap_eval.build_zl_model_xml("stairs"))

    def stair_geoms(model):
        return {
            name: (tuple(model.geom(name).size), tuple(model.geom(name).pos))
            for name in [
                *(f"stair_probe_up_{index}_geom" for index in range(sim2sim.LOCO_STAIR_STEPS)),
                "stair_probe_landing_geom",
            ]
        }

    direct_geoms = stair_geoms(direct)
    zl_geoms = stair_geoms(zl)
    assert direct_geoms.keys() == zl_geoms.keys()
    for name in direct_geoms:
        assert direct_geoms[name][0] == pytest.approx(zl_geoms[name][0])
        assert direct_geoms[name][1] == pytest.approx(zl_geoms[name][1])


def test_stair_success_requires_joint_position_height_and_duration() -> None:
    from legged_lab.scripts import eval_t4_depth_plant_gap as gap_eval

    samples = [
        {"x_m": 3.8, "y_m": 0.0, "height_m": 1.6},
        {"x_m": 3.9, "y_m": 0.0, "height_m": 1.7},
        {"x_m": 4.0, "y_m": 0.0, "height_m": 1.8},
        {"x_m": 4.1, "y_m": 0.0, "height_m": 1.9},
        {"x_m": 4.2, "y_m": 0.0, "height_m": 1.9},
    ]
    assert gap_eval.longest_stair_top_run(samples) == 5
    samples[2]["y_m"] = 2.0
    assert gap_eval.longest_stair_top_run(samples) == 2


def test_vault_scene_box_hands_feet_match_isaac_surfaces() -> None:
    model = _load_vault()
    apply_isaac_contact_friction(model)
    box = model.geom("vault_box")
    assert int(model.geom_condim[int(box.id)]) == 3
    assert np.allclose(model.geom_friction[int(box.id)], ISAAC_FRICTION)
    np.testing.assert_allclose(box.size, [s / 2.0 for s in T4_VAULT_BOX_SIZE], atol=1e-6)

    for name in ("left_sphere_hand", "right_sphere_hand"):
        geom = model.geom(name)
        assert int(model.geom_condim[int(geom.id)]) == 3
        assert np.allclose(model.geom_friction[int(geom.id)], ISAAC_FRICTION)

    for side in ("left", "right"):
        for i, (pos, half) in enumerate(ISAAC_FOOT_BOXES, start=1):
            geom = model.geom(f"{side}_foot{i}_collision")
            np.testing.assert_allclose(geom.pos, pos, atol=1e-6)
            np.testing.assert_allclose(geom.size[:3], half, atol=1e-6)
        with pytest.raises(KeyError):
            model.geom(f"{side}_foot7_collision")


def test_g1_and_teacher_runners_call_shared_contract() -> None:
    g1 = (ROOT / "legged_lab/scripts/play_t4_vault_g1_mujoco.py").read_text()
    teacher = (ROOT / "legged_lab/scripts/play_t4_teacher_viser.py").read_text()
    assert "apply_isaac_pd" in g1 and "rewrite_vault_mjcf" in g1
    assert "dof_damping[" not in g1
    assert "apply_isaac_pd" in teacher
    assert "dof_damping" not in teacher
