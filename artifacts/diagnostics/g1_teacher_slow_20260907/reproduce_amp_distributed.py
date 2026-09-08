"""Two-process test of the production AMP broadcast and gradient reduction."""

from pathlib import Path
import json, os, sys
import numpy as np
import torch
import torch.distributed as dist
from rsl_rl.algorithms.amp_ppo import AMPPPO
from rsl_rl.modules import Discriminator
from rsl_rl.utils import Normalizer

dist.init_process_group('gloo')
rank, world = dist.get_rank(), dist.get_world_size()
torch.manual_seed(42 + rank)
alg = AMPPPO.__new__(AMPPPO)
alg.device, alg.gpu_global_rank, alg.gpu_world_size, alg.is_multi_gpu = 'cpu', rank, world, True
alg.policy = torch.nn.Linear(3, 2)
alg.discriminator = Discriminator(6, 0.3, [8], 'cpu', 0.7)
alg.amp_normalizer, alg.rnd = Normalizer(3), None
alg.amp_normalizer.mean[:] = rank
alg.broadcast_parameters()
def vector(module): return torch.cat([p.detach().reshape(-1) for p in module.parameters()])
def maximum_rank_difference(value):
    gathered = [torch.zeros_like(value) for _ in range(world)]
    dist.all_gather(gathered, value)
    return max(float((v - gathered[0]).abs().max()) for v in gathered)
result = {'world_size': world,
          'policy_difference_after_broadcast': maximum_rank_difference(vector(alg.policy)),
          'discriminator_difference_after_broadcast': maximum_rank_difference(vector(alg.discriminator)),
          'normalizer_difference_after_broadcast': maximum_rank_difference(torch.tensor(alg.amp_normalizer.mean))}
for module in (alg.policy, alg.discriminator):
    for param in module.parameters(): param.grad = torch.full_like(param, rank + 1.)
alg.reduce_parameters()
result['policy_gradient_difference_after_reduce'] = maximum_rank_difference(
    torch.cat([p.grad.reshape(-1) for p in alg.policy.parameters()]))
result['discriminator_gradient_difference_after_reduce'] = maximum_rank_difference(
    torch.cat([p.grad.reshape(-1) for p in alg.discriminator.parameters()]))
if hasattr(alg, '_update_amp_normalizer'):
    alg.amp_normalizer = Normalizer(3)
    data = torch.arange(3).float().unsqueeze(0).repeat(rank + 1, 1) + (2 * rank + 1)
    alg._update_amp_normalizer(data, data + 1)
    expected = Normalizer(3)
    all_samples = np.concatenate([np.tile(np.arange(3) + 2 * r + offset, (r + 1, 1))
                                  for r in range(world) for offset in (1, 2)])
    expected.update(all_samples)
    result['normalizer_mean_error'] = float(np.max(np.abs(alg.amp_normalizer.mean - expected.mean)))
    result['normalizer_variance_error'] = float(np.max(np.abs(alg.amp_normalizer.var - expected.var)))
    result['normalizer_difference_after_update'] = maximum_rank_difference(torch.tensor(alg.amp_normalizer.mean))
    result['normalizer_count'] = alg.amp_normalizer.count
if rank == 0:
    Path(sys.argv[1]).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)
if '--require-consistent' in sys.argv:
    assert all(value < 1e-9 for key, value in result.items() if 'difference' in key or 'error' in key), result
dist.destroy_process_group()
