# S12 model_5999 targeted-FT decision

Date: 2026-08-29

## Decision

- A generic continuation of the existing `s12_rtx_deploy_ft_v2` recipe is rejected.
- Targeted robustness FT is not a prerequisite for a tethered flat or large-platform hardware smoke test.
- Targeted robustness FT is required before claiming readiness for real narrow stepping stones or raised pillars.
- The parent remains the full Phase-B `model_5999.pt`; Phase A is retained as a cross-domain diagnostic reference, not used as the deployment parent.

## Evidence boundary

MuJoCo is not treated as ground truth for Isaac or the robot. The useful signal is whether a policy retains margin under modest, real-robot-relevant perturbations and whether Phase A/B changes produce a consistent cross-domain trade-off.

## Current Phase-B coverage

The accepted Phase-B lineage already trained with:

- depth range noise `sigma = 0.005 + 0.02 d`;
- global depth scale in `[0.95, 1.05]`;
- 2--3 depth-frame delay steps and hold-2 refresh;
- six structured dropout blocks;
- camera position jitter `+-0.01 m` and orientation jitter `+-0.025 rad` on all axes;
- proprioceptive observation noise;
- robot material, trunk mass, reset-state, joint-reset and push randomization.

Action delay was disabled. Actuator gains, effort and armature were not randomized. Pillar geometry at a fixed difficulty is deterministic and has no independent top tilt, XY placement error or per-pillar height error.

## Isaac student-only gates

| Gate | Phase A pure DAgger | Phase B joint PPO |
| --- | ---: | ---: |
| Easy stones strict / reach2m | 32/32 / 32/32 | 32/32 / 32/32 |
| Easy pillars strict / reach2m | 27/32 / 28/32 | 29/32 / 30/32 |
| Hard stones strict / reach2m | 4/32 / 10/32 | 18/32 / 28/32 |
| Hard pillars strict / reach2m | 5/32 / 15/32 | 27/32 / 30/32 |

Phase B therefore gained real Isaac hard-terrain capability; reverting to Phase A would discard verified capability.

## MuJoCo robustness probes

The corrected MuJoCo plant uses aligned position-servo Kp/Kd/effort, clears XML joint friction loss, and uses the MuJoCo 3.x velocity API.

Phase-B single-variable matrix, easy terrain:

- flat baseline remains stable for 12 s;
- easy stones reach 2 m at baseline but fail under several small gain, armature, latency and camera perturbations;
- easy pillars reach 2 m in 0/19 single-variable conditions;
- terrain sliding friction in `[0.6, 1.0]` is not the dominant factor.

Phase-B 20-domain combined stress test:

| Course | No fall for 10 s | Reach 2 m |
| --- | ---: | ---: |
| Flat | 20/20 | 20/20 |
| Easy stones | 5/20 | 10/20 |

Factor isolation on easy stones:

| Perturbation family | No fall for 10 s | Reach 2 m |
| --- | ---: | ---: |
| Perception only | 9/20 | 16/20 |
| Plant only | 9/20 | 14/20 |

The narrow-terrain weakness is therefore not attributable to only perception or only dynamics.

## Phase A versus Phase B cross-domain trade-off

The same 20 combined domains were paired by perturbation seed.

| Checkpoint | Flat no-fall / reach2m | Stones no-fall / reach2m | Stones mean max x |
| --- | ---: | ---: | ---: |
| Phase A model_6000 | 20/20 / 20/20 | 2/20 / 15/20 | 3.28 m |
| Phase B model_5999 | 20/20 / 20/20 | 5/20 / 10/20 | 2.43 m |

Phase B moved less far in 15/20 paired stone domains, losing an average `0.85 m`, while sometimes surviving longer. On MuJoCo single-variable probes, Phase A reached 2 m on raised pillars in 16/19 conditions versus 0/19 for Phase B.

Phase A to Phase B deploy-network relative L2 change:

- depth encoder: `19.1%`;
- GRU: `6.4%`;
- student action MLP: `4.6%`.

Same-observation counterfactual action differences average `0.08--0.12` normalized action units and include ankle pitch, knee pitch and hip roll/yaw. This is consistent with cumulative Isaac specialization, not merely a rendering bug.

## Required targeted FT domain expansion

1. Enable action delay randomization over `0--2` policy steps.
2. Randomize Kp/Kd by `+-10%`, effort by `+-10%`, and armature by `+-20%`.
3. Retain existing depth noise, delay, scale and camera extrinsic DR.
4. Add structured depth hit/no-hit boundary corruption rather than matching a single valid-pixel fraction.
5. Add independent sparse-terrain manufacturing variation: per-tile seed, pillar XY offset, per-pillar height error, small top tilt, and bounded diameter/pitch variation.
6. Keep friction/material/mass/reset/push DR already present.

## Conservative optimization recipe

- New lineage name, never overwrite Phase B.
- Parent: full Phase-B `model_5999.pt`, with optimizer/iteration reset.
- `teacher_mix = 0`.
- 200-update critic warmup with actor frozen.
- Maximum PPO coefficient `0.1` initially; do not use the old Phase-C `0.5`.
- Keep teacher behavior coefficient at least `1.0` and reconstruction coefficient `1.0`.
- Start at learning rate `1e-5`; promotion to `3e-5` requires the short health/capability gate.
- Add a frozen Phase-B reference-policy KL/action anchor on nominal-domain rows. Per-update PPO KL is insufficient to prevent cumulative 6000-update drift.
- Use one user-visible tmux training job with internal smoke and rollback gates, not multiple manual training stages.

## Acceptance and rollback gates

Isaac fixed-seed gates must not regress below the accepted Phase-B checkpoint:

- easy stones strict/reach2m: `32/32` / `32/32`;
- easy pillars: at least `29/32` / `30/32`;
- hard stones: at least `18/32` / `28/32`;
- hard pillars: at least `27/32` / `30/32`.

MuJoCo robustness targets are diagnostics, not claims of robot success:

- combined flat: retain `20/20` no-fall and reach2m;
- combined easy stones: improve from `10/20` reach2m and `5/20` no-fall to at least `16/20` and `10/20`;
- easy pillars: must recover from `0/19` reach2m to a non-degenerate result, with Phase A kept as the comparison reference;
- require continuous replay, evaluator JSON and complete lineage for every accepted checkpoint.

Any Isaac gate regression, cumulative clean-policy drift, numerical rejection or loss of flat robustness rolls back to Phase-B `model_5999`.

## Hardware boundary

Even after FT, the first robot sequence remains separate: tethered flat -> large platforms -> wide stones -> narrow stones/pillars. MuJoCo success cannot replace this hardware progression.
