# Z2 AMP experts

This directory is the formal 70D expert location for `z2_loco_teacher`.

The full expert set was generated on the original Z2 USD in Isaac Sim 5.1 / IsaacLab 2.1:
`walk` 74 frames, `walk_l` 280 frames, `run` 39 frames (393 total).
All frames passed independent original-URDF FK comparison; maximum endpoint error was below 1.5e-6 m.
`_manifest.json` binds source CSV/raw hashes, the five original USD layers, generator provenance and each expert SHA256.

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
