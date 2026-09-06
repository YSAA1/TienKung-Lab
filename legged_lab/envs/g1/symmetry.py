"""Compatibility mirror entrypoints using the shared locomotion implementation."""

from types import SimpleNamespace
from legged_lab.assets.unitree_g1.locomotion import G1_LOCOMOTION as SPEC
from legged_lab.locomotion.schemas import ObservationLayout
from legged_lab.locomotion import symmetry as shared

SPARSE_LAYOUT = ObservationLayout(len(SPEC.joint_names), scan_history=5,
                                  actor_contact=True, critic_foot_scan=True, critic_immunity=True)


def joint_mirror_map():
    return SPEC.mirror_indices, SPEC.mirror_signs


def mirror_actions(actions):
    return shared.mirror_actions(actions, SPEC)


def _layout(width, is_critic):
    if width == (SPARSE_LAYOUT.critic_dim if is_critic else SPARSE_LAYOUT.actor_dim):
        return SPARSE_LAYOUT
    base = ObservationLayout(len(SPEC.joint_names))
    contact = not is_critic and width == base.actor_dim + 2
    frame = base.critic_frame_dim if is_critic else base.proprio_dim
    remaining = width - base.scan_dim - 2 * contact
    if remaining <= 0 or remaining % frame:
        raise ValueError(f"observation width {width} does not match {SPEC.name} legacy layouts")
    history = remaining // frame
    return ObservationLayout(len(SPEC.joint_names), actor_history=history,
                             critic_history=history, actor_contact=contact)


def mirror_observations(obs, is_critic):
    return shared.mirror_observations(obs, is_critic, SPEC, _layout(obs.shape[-1], is_critic))


def get_symmetric_states(obs=None, actions=None, env=None, obs_type=None, is_critic=None):
    critic = obs_type == "critic" if is_critic is None else bool(is_critic)
    if env is None:
        layout = _layout(obs.shape[-1], critic) if obs is not None else SPARSE_LAYOUT
        env = SimpleNamespace(cfg=SimpleNamespace(robot_spec=SPEC), observation_layout=layout)
    return shared.get_symmetric_states(obs, actions, env, obs_type, is_critic)

G1_PROPRIO_FRAME_DIM = SPARSE_LAYOUT.proprio_dim
G1_CRITIC_FRAME_DIM = SPARSE_LAYOUT.critic_frame_dim
G1_SPARSE_ACTOR_OBS_DIM = SPARSE_LAYOUT.actor_dim
G1_SPARSE_CRITIC_OBS_DIM = SPARSE_LAYOUT.critic_dim
