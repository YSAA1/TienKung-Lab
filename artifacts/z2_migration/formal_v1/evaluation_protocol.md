# Z2 vs current G1 early training evidence protocol

Scope: compare checkpoints 1000/2000/3000, never substitute startup/loss for behavior. Z2 frozen training source933e08a; G1 reference is the independent vital_motion_v1 run2026-09-08_16-25-01. Do not modify either live training runtime.

## Fixed evaluator

- Each robot runs from its own remote source directory, with its own asset and expert data.
- Z2 registry config; G1 explicitly applies the frozen helper apply_vital_motion_v1(env_cfg), whose captured source is g1_reference/motion_experiment.py. G1 full17 expert manifest must match the saved agent.yaml.
- Common eval seed42, deterministic actor, commandvx0.7m/s, zero lateral/yaw commands, num_envs32, episodes64, scan normal, default evaluator randomization disabling. Retain morphology-specific termination thresholds and record them.
- Evaluate flat difficulty0.0 and stepping_stones difficulty0.0 for each checkpoint. These controlled low-difficulty checks are not samples of the training curriculum or evidence for hard sparse terrain.
- Bind each JSON to checkpointSHA256, source/helper hashes, exact argv and effective profile fields. Use the actual evaluator completed episode count.
- Report forward reach2m (max displacement projected along initial heading), terminal radial progress (relative to terrain origin), OOB reset reason count / completed episodes and other terminal causes separately. Causes may overlap. OOB is not a success count.

## Training snapshots

Only1000/2000/3000. Last100iteration arithmetic scalar means are descriptive training statistics, not pooled per-episode success rates. Preserve full snapshots with checkpointSHA. G1 snapshots already extracted without nonfinite tags. Compare common tags per terrain rather than collapsing sparse and dense terrain.

## Continuous replay

Record checkpoint3000 for both robots on flat and stepping_stones with --terrain, explicit terrain_types, difficulty0.0, commandvx0.7, num_envs1, seed42, duration20s. Preserve full MP4, reset text and diagnosticsJSON; inspect actual frames and trajectory. Existing play changes episode horizon to40s whereas evaluator retains20s, so this is supplementary observation of the same checkpoint, not the identical evaluator protocol. Continuous video may include automatic resets; report their count and do not call it an unbroken successful episode.

## Decision

Require actual Z2 nonzero trainingOOB and reach2m, finite training, and learned movement without persistent collapse/standing. Quantify gaps vs G1 by terrain and fixed eval; G1 itself has near-zero early sparse success. If criteria fail, investigate using these outputs and preserve lineage; do not hotpatch running source or claim completion because training is live.
