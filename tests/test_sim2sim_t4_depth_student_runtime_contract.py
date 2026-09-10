import pytest

pytest.importorskip("mujoco")
import mujoco
import numpy as np

from legged_lab.assets.t4.constants import T4_JOINT_NAMES
from legged_lab.scripts.sim2sim_t4_depth_student import (
    DepthStudentSim,
    body_velocities_world,
    build_model_xml,
    pd_group,
)


def test_depth_student_position_servos_replace_passive_joint_terms():
    model = mujoco.MjModel.from_xml_path(build_model_xml(course="flat"))
    sim = DepthStudentSim.__new__(DepthStudentSim)
    sim.model = model
    sim.joint_ids = np.asarray(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in T4_JOINT_NAMES]
    )
    sim.dof_adr = np.asarray([model.jnt_dofadr[joint_id] for joint_id in sim.joint_ids])
    sim.actuator_ids = np.asarray(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "M_" + name[2:]) for name in T4_JOINT_NAMES]
    )
    sim.kp = np.asarray([pd_group(name)[0] for name in T4_JOINT_NAMES])
    sim.kd = np.asarray([pd_group(name)[1] for name in T4_JOINT_NAMES])
    sim.effort_limit = np.asarray(model.jnt_actfrcrange[sim.joint_ids, 1])

    assert np.all(model.dof_frictionloss[sim.dof_adr] > 0.0)
    sim._configure_position_servos()

    np.testing.assert_allclose(model.dof_damping[sim.dof_adr], sim.kd)
    np.testing.assert_allclose(model.dof_frictionloss[sim.dof_adr], 0.0)


def test_body_velocities_world_uses_supported_mujoco_api():
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="moving">
              <freejoint/>
              <geom type="sphere" size="0.1" mass="1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "moving")
    data.qvel[:6] = [1.0, 2.0, 3.0, 0.1, 0.2, 0.3]
    mujoco.mj_forward(model, data)

    linear, angular = body_velocities_world(model, data)

    np.testing.assert_allclose(linear[body_id], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(angular[body_id], [0.1, 0.2, 0.3])
