import argparse, json, os, threading
from pathlib import Path
from isaaclab.app import AppLauncher
from legged_lab.scripts.isaaclab_runtime_compat import (
    patch_missing_physx_material_attributes,
    patch_physx_backward_compatibility_setting,
)

p = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(p)
patch_physx_backward_compatibility_setting(AppLauncher)
a = p.parse_args()
app = AppLauncher(a).app
import torch
from legged_lab.envs import *
from legged_lab.utils import task_registry


@torch.inference_mode()
def main():
    patch_missing_physx_material_attributes()
    cfg, ac = task_registry.get_cfgs("g1_loco_teacher")
    cfg.device = a.device
    cfg.sim.device = a.device
    cfg.scene.num_envs = 16
    cfg.scene.seed = 42
    cfg.noise.add_noise = False
    cfg.scene.terrain_generator.sub_terrains = {"flat": cfg.scene.terrain_generator.sub_terrains["flat"]}
    cfg.scene.terrain_generator.sub_terrains["flat"].proportion = 1.0
    cfg.scene.terrain_generator.num_rows = 1
    cfg.scene.terrain_generator.num_cols = 1
    cfg.scene.max_init_terrain_level = 0
    cfg.random_level_reset_fraction = 0
    cfg.domain_rand.events.push_robot = None
    cfg.domain_rand.events.add_base_mass = None
    cfg.domain_rand.events.reset_base.params["pose_range"] = {"x": (0, 0), "y": (0, 0), "yaw": (0, 0)}
    cfg.scene.robot.init_state.pos = (0, 0, 1.8)
    cfg.scene.robot.spawn.rigid_props.disable_gravity = True
    e = task_registry.get_task_class("g1_loco_teacher")(cfg, True)
    e.reset = lambda ids: None
    r = e.robot
    records = []
    u = torch.zeros(e.num_envs, e.num_actions, device=e.device)
    u[4:8, e.policy_joint_names.index("left_shoulder_roll_joint")] = -8
    u[4:8, e.policy_joint_names.index("right_shoulder_roll_joint")] = 8
    for side in ["left", "right"]:
        u[8:12, e.policy_joint_names.index(side + "_hip_pitch_joint")] = -8
        u[8:12, e.policy_joint_names.index(side + "_knee_joint")] = 8
    u[12:] = torch.randn(4, e.num_actions, device=e.device) * 4
    torso = r.body_names.index("torso_link")
    sensorbody = e.contact_sensor.body_names.index("torso_link")
    for s in range(50):
        obs, rew, dones, extra = e.step(u)
        force = e.contact_sensor.data.net_forces_w[:, sensorbody].norm(dim=-1)
        records.append(
            {
                "step": s + 1,
                "torso_height": r.data.body_pos_w[:, torso, 2].tolist(),
                "root_height": r.data.root_pos_w[:, 2].tolist(),
                "torso_force": force.tolist(),
                "reset_torso": e.reset_reason_masks["torso"].tolist(),
                "root_tilt": e.last_tilt_rad.tolist(),
            }
        )
    result = {
        "asset": cfg.scene.robot.spawn.asset_path,
        "joints": r.num_joints,
        "self_collisions": cfg.scene.robot.spawn.articulation_props.enabled_self_collisions,
        "purpose": "airborne arm contact attribution, no policy/MDP change",
        "hard_contact_bodies": cfg.robot.terminate_contacts_body_names,
        "groups": ["zero", "shoulder_roll_inward_2rad", "hip_knee_tuck_2rad", "random_std4"],
        "records": records,
    }
    assert e.termination_contact_cfg.body_ids == []
    assert all(not any(x["reset_torso"]) for x in records)
    assert any(max(x["torso_force"]) > 100 for x in records)
    Path("artifacts/portability/v6/self_contact_probe_extreme.json").write_text(json.dumps(result, indent=2))
    print(
        "SELF_CONTACT",
        [(x["step"], sum(x["reset_torso"]), min(x["torso_height"]), max(x["torso_force"])) for x in records],
        flush=True,
    )


if __name__ == "__main__":
    code = 0
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()
        code = 1
    finally:
        threading.Timer(15, os._exit, args=(code,)).start()
        app.close()
        os._exit(code)
