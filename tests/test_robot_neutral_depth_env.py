"""Execute the depth observation assembly without starting the simulator."""

import ast
from pathlib import Path
import pytest
import torch

from legged_lab.locomotion import schemas

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "legged_lab/locomotion/depth_env.py"


class _Buffer:
    def __init__(self, frames):
        self.buffer = frames.clone()

    def append(self, frame):
        self.buffer = torch.cat([self.buffer[:, 1:], frame[:, None]], dim=1)


class _Physics:
    def compute_observations(self):
        self.actor_obs_buffer.append(self.current_actor)
        return self.teacher.clone(), torch.zeros(2, self.observation_layout.critic_dim)


def _runtime_classes():
    # Run the production class bodies; only their Isaac physics parent is replaced.
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    namespace = dict(vars(schemas), torch=torch, F=torch.nn.functional, LocomotionEnv=_Physics)
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace["DepthDistillationEnv"], namespace["LightLPDepthDistillationEnv"]


def _env(cls, joints, sparse):
    env = object.__new__(cls)
    env.num_envs = 2
    env.clip_obs = 100.0
    env.observation_layout = schemas.ObservationLayout(
        joints, scan_history=5 if sparse else 1, actor_contact=sparse,
    )
    layout = env.observation_layout
    env.actor_obs_buffer = _Buffer(torch.randn(2, 10, layout.proprio_dim))
    env.critic_obs_buffer = _Buffer(torch.randn(2, 10, layout.critic_frame_dim))
    env.current_actor = torch.randn(2, layout.proprio_dim)
    env.compute_current_observations = lambda: (env.current_actor, torch.randn(2, layout.critic_frame_dim))
    env.scan = torch.randn(2, layout.scan_dim)
    env.compute_teacher_terrain_privilege = lambda: env.scan
    env.teacher = torch.randn(2, layout.actor_dim)
    env.depth_history = torch.rand(2, 1 if sparse else 3, *schemas.DEPTH_POLICY_SIZE)
    env._update_depth_history = lambda: None
    return env


@pytest.mark.parametrize("joints,student_dim,teacher_dim", [(27, 10176, 1155), (29, 10236, 1215), (21, 9996, 975)])
def test_feedforward_depth_layout_uses_joint_count(joints, student_dim, teacher_dim):
    feedforward, _ = _runtime_classes()
    env = _env(feedforward, joints, sparse=False)
    student, teacher = env.compute_observations()
    history = env.actor_obs_buffer.buffer.flatten(1)
    assert student.shape == (2, student_dim)
    assert teacher.shape == (2, teacher_dim)
    assert torch.equal(student, torch.cat([history, env.depth_history.flatten(1)], dim=-1))
    assert torch.equal(teacher, torch.cat([history, env.scan], dim=-1))


@pytest.mark.parametrize("joints,student_dim,teacher_dim", [(27, 3168, 1937), (29, 3174, 1997), (21, 3150, 1757)])
def test_recurrent_student_gets_only_latest_proprio_and_policy_depth(joints, student_dim, teacher_dim):
    _, recurrent = _runtime_classes()
    env = _env(recurrent, joints, sparse=True)
    student, teacher = env.compute_observations()
    assert student.shape == (2, student_dim)
    assert teacher.shape == (2, teacher_dim)
    assert torch.equal(student, torch.cat([env.current_actor, env.depth_history[:, -1].flatten(1)], dim=-1))
    assert torch.equal(teacher, env.teacher)
    # Teacher scan/contact changes cannot leak into the deployable input.
    env.teacher.fill_(77)
    student_again, teacher_again = env.compute_observations()
    assert torch.equal(student_again, student)
    assert torch.all(teacher_again == 77)


@pytest.mark.parametrize("wrong_field", ["teacher", "current_actor", "depth_history"])
def test_recurrent_rejects_malformed_observation_streams(wrong_field):
    _, recurrent = _runtime_classes()
    env = _env(recurrent, 29, sparse=True)
    value = getattr(env, wrong_field)
    setattr(env, wrong_field, value[..., :-1])
    with pytest.raises((RuntimeError, ValueError)):
        env.compute_observations()
