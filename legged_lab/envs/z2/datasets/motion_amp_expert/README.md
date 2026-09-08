# Z2 AMP experts

This directory is the formal 70D expert location for `z2_loco_teacher`.

Generate on Isaac (nubot tmux):

```
bash scripts/nubot_run.sh legged_lab/scripts/generate_z2_amp_expert.py \
  --output-dir legged_lab/envs/z2/datasets/motion_amp_expert --headless
```

Do not copy:

- upstream `motion_amp_expert/*.txt` (64D, no feet; unrepaired `run.txt` has wrong qvel)
- G1 70D experts
- `g1_compat/`
- visualization 70D (`root+euler+q+dq`)
