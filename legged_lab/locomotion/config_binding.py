"""Bind algorithm reward roles to a robot's explicitly declared body semantics."""

from isaaclab.managers import SceneEntityCfg


def bind_reward_roles(reward_cfg, spec):
    bodies = {"torso": (spec.torso,), "feet": spec.feet, **spec.reward_bodies}
    for term in vars(reward_cfg).values():
        for value in (getattr(term, "params", None) or {}).values():
            if not isinstance(value, SceneEntityCfg):
                continue
            for attribute, roles in (("body_names", bodies), ("joint_names", spec.reward_joints)):
                names = getattr(value, attribute, None)
                if isinstance(names, str) and names.startswith("$"):
                    try:
                        setattr(value, attribute, list(roles[names[1:]]))
                    except KeyError as error:
                        raise ValueError(f"{spec.name}: missing reward role {names}") from error
    if hasattr(reward_cfg, "feet_y_distance"):
        reward_cfg.feet_y_distance.params["target"] = spec.nominal_feet_distance


def bind_robot_spec(cfg):
    spec = cfg.robot_spec
    spec.validate()
    cfg.policy_joint_names = spec.joint_names
    cfg.feet_link_names = list(spec.feet)
    cfg.left_leg_joint_names = list(spec.left_leg)
    cfg.right_leg_joint_names = list(spec.right_leg)
    cfg.ankle_joint_names = list(spec.ankles)
    cfg.diagnostic_contact_body_names = spec.diagnostic_bodies
    cfg.scene.height_scanner.prim_body_name = spec.torso
    cfg.scene.foot_scanner.body_names = spec.feet
    cfg.robot.terminate_contacts_body_names = [spec.torso]
    cfg.robot.feet_body_names = list(spec.feet)
    bind_reward_roles(cfg.reward, spec)
    cfg.domain_rand.events.add_base_mass.params["asset_cfg"] = SceneEntityCfg("robot", body_names=spec.torso)
