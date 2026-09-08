"""Run the production AMP update with known raw samples and capture its inputs."""

import json
from pathlib import Path
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parents[3]
if not (ROOT / 'rsl_rl').exists():
    ROOT = Path('/home/nubot/phn_ws/t4_train/TienKung-Lab-g1-portability-20260906')
sys.path.insert(0, str(ROOT / 'rsl_rl'))
import numpy as np
import torch
from rsl_rl.algorithms.amp_ppo import AMPPPO
from rsl_rl.modules import ActorCritic, Discriminator
from rsl_rl.utils import Normalizer

torch.manual_seed(42)
raw = torch.tensor([[10., -20., 30.], [12., -22., 34.]]).repeat(4, 1)
next_raw = raw + 0.25
class Recorder(Normalizer):
    def __init__(self):
        super().__init__(3)
        self.mean[:] = [2., -3., 4.]
        self.var[:] = [4., 9., 16.]
        self.updates = []
    def update(self, arr):
        self.updates.append(arr.copy())
        super().update(arr)

norm = Recorder()
expected_norm = norm.normalize_torch(raw, 'cpu').clone()
discriminator = Discriminator(6, 0.3, [8], 'cpu', 0.7)
grad_inputs = []
compute_grad_pen = discriminator.compute_grad_pen
def capture(*args, **kwargs):
    grad_inputs.append([x.clone() for x in args])
    return compute_grad_pen(*args, **kwargs)
discriminator.compute_grad_pen = capture
policy = ActorCritic(5, 5, 2, actor_hidden_dims=[8], critic_hidden_dims=[8])
loader = SimpleNamespace(feed_forward_generator=lambda *args: iter([(raw, next_raw)]))
alg = AMPPPO(policy, discriminator, loader, norm, amp_replay_buffer_size=16, device='cpu')
alg.init_storage('rl', 4, 2, [5], [5], [2])
for _ in range(2):
    obs = torch.randn(4, 5)
    alg.act(obs, obs, raw[:4])
    alg.process_env_step(torch.ones(4), torch.zeros(4, dtype=torch.bool), {}, next_raw[:4])
alg.compute_returns(torch.randn(4, 5))
alg.amp_storage.feed_forward_generator = lambda *args: iter([(raw, next_raw)])
loss = alg.update()
result = {
    'source': str(ROOT / 'rsl_rl/rsl_rl/algorithms/amp_ppo.py'),
    'raw_expert_first_row': raw[0].tolist(),
    'normalized_expert_first_row': expected_norm[0].tolist(),
    'normalizer_update_first_row': norm.updates[-1][0].tolist(),
    'grad_penalty_input_first_row': grad_inputs[0][0][0].tolist(),
    'normalizer_receives_raw': bool(np.allclose(norm.updates[-1], raw.numpy())),
    'gradient_penalty_receives_discriminator_coordinates': bool(torch.allclose(grad_inputs[0][0], expected_norm)),
    'loss': loss,
}
print(json.dumps(result, indent=2))
Path(sys.argv[1]).write_text(json.dumps(result, indent=2)+'\n')
