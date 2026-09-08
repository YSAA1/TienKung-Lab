# Milestone evidence wrapper — implementer report

Local worktree only: `D:/TienKung-Lab-z2-teacher-20260909`. Frozen training source `933e08a` on nubot was not SSHed, deployed, or changed. `eval_locomotion.py`, `play.py`, `train.py`, and training packages were not modified. No git commit.

## Feedback 10 (bounded)

1. **Frozen G1 helper.** Loader now calls `apply_vital_motion_v1(env_cfg)` only. That export exists on the captured frozen file `artifacts/z2_migration/formal_v1/g1_reference/motion_experiment.py` (no `apply_vital_motion_experiment`) and on the local G1 file as a compatibility wrapper. Internal apply signature is one argument.
2. **`--g1_*` rejected before simulation** in both eval and play. Frozen G1 eval/play would silently drop those flags. Not a general unknown-flag parser.
3. **`success` is not "no exception".** It requires `get_cfgs_consumed`, an existing checkpoint SHA256, and either eval's returned dict with `completed_episodes` or the specified play `--record` file on disk. `--help` / `SystemExit(0)` cannot be success.
4. **`--seed` other than 42 is rejected**, including `--seed=43` and `--seed -1`, so `play.update_rsl_rl_cfg` cannot override the snapshot. `--seed 42` / `--seed=42` / omitted `--seed` are allowed.

## Why this wrapper exists

`eval_locomotion.py` / `play.py` call `task_registry.get_cfgs(task)` and then construct the env. They do not call `apply_vital_motion_experiment`. Leftover `--g1_*` flags now raise in eval instead of being dropped. Fair G1 `vital_v1` eval therefore cannot be done by passing train.py flags into eval/play.

The wrapper imports those scripts with `runpy.run_path(target, run_name='milestone_import')` so AppLauncher and `from legged_lab.envs import *` run, but the bottom `__main__` does not. It then replaces `task_registry.get_cfgs` so the returned `env_cfg` is mutated **before** `evaluate()` / `play()` consume it.

## Exact changes

| Path | Change |
| --- | --- |
| `work/z2-migration/run-milestone-evidence.py` | **New.** Standalone wrapper. |
| `tests/test_milestone_evidence_wrapper.py` | **New.** CPU-only stubs. |
| `docs/README.md` | Single Z2 table row: `未开训` / “70D 专家需 Isaac FK” → current training state matching `execution-state.md`. Intro paragraph already said training; no second Z2 row. Other table rows untouched. |
| `.harness/work_index.md` | Removed the obsolete second Z2 row (`implementing` / `未开训`). Kept the existing top row `Z2 29DoF teacher` / `active / training`. Other rows untouched. |
| `legged_lab/scripts/eval_locomotion.py` | Unchanged. |
| `legged_lab/scripts/play.py` | Unchanged. |
| `legged_lab/scripts/train.py` | Unchanged. |
| `legged_lab/envs/g1/motion_experiment.py` | Unchanged. |

`git diff --stat` on docs + existing scripts: `.harness/work_index.md` 1 deletion, `docs/README.md` 1 line replacement.

## Wrapper contract

Required flags (stripped from downstream `sys.argv`):

- `--mode {eval,play}`
- `--profile {registry,vital_v1}`
- `--project-root PATH` — source checkout used for `sys.path`, `chdir`, and the target script. This is how a separate G1 frozen tree is used instead of this Z2 runtime.
- `--evidence-manifest PATH`

Optional wrapper-only flag (also stripped):

- `--amp-expert-manifest` / `--amp_expert_manifest` — exact `train.py` clip+sha256 loop (`manifest["clips"][name]["sha256"]` vs `{name}.txt` next to the manifest). No default path. No full17 discovery.

Remaining script flags (`--task`, `--checkpoint`, `--load_run`, `--output`, `--record`, `--terrain`, `--terrain_types`, `--command_vx`, AppLauncher flags) are forwarded unchanged as `sys.argv = [target_script] + downstream`. Wrapper flags are never added to that list. `--g1_*` is **not** forwarded: it is rejected before AppLauncher. `--seed` other than 42 is rejected the same way.

Profiles:

- `registry` — no VITAL mutation. Seed 42 is still written onto `agent_cfg.seed` and `env_cfg.scene.seed`. Intended default for Z2 (`z2_loco_teacher` already has `max_radial` + `progress_monitor` in its recipe).
- `vital_v1` — `apply_vital_motion_v1(env_cfg)` from `{project-root}/legged_lab/envs/g1/motion_experiment.py`. Frozen G1 only has that one-arg function; local G1 keeps it as a wrapper around `apply_vital_motion_experiment(..., "vital_v1")`. Non-G1 still raises. Seed 42 applied after a successful profile apply.

Seed: wrapper sets 42 at the `get_cfgs` seam. Any CLI `--seed` other than 42 is rejected before simulation so `update_rsl_rl_cfg` cannot rewrite it.

Close: `try/finally` writes the evidence JSON, then `threading.Timer(timeout, os._exit, args=(exit_code,))` + `simulation_app.close()` + `os._exit(exit_code)`. Eval timeout is **90s**, copied from `eval_locomotion.py`. Play.py has no hang timer; the wrapper still bounds play close at 90s.

Eval JSON: produced by existing `evaluate()` to `--output`. Play `--record`: existing code writes MP4 (or GIF fallback), `{stem}.txt` reset events, and `{stem}.diagnostics.json`.

## Evidence manifest fields

Written even if the target raises (as long as `--evidence-manifest` parsed). Includes:

- `full_argv`, `downstream_argv`, `mode`, `profile`, `seed` (42), `project_root`, `target_script`
- checkpoint path + SHA256 if the `--checkpoint` / `--checkpoint_path` / eval JSON `checkpoint` field names an existing file. **Does not search `logs/` or guess run directories.**
- SHA256 of wrapper, `{project-root}` copies of `eval_locomotion.py`, `play.py`, `train.py`, `motion_experiment.py`, `task_registry.py` (missing files recorded as `exists: false`)
- optional AMP manifest SHA256 and **declared** clip hashes only when the parent passed `--amp-expert-manifest`
- `effective_profile` captured at `get_cfgs` return: termination (`acceleration_termination_enabled`, `deterministic_fall_limits`), gait gate, `action_rate_l2` weight, `lightlp_promotion_distance`, `progress_monitor_enabled`, `sparse_command_min_speed_scale`, `actor_hidden_dims` / `critic_hidden_dims` / `amp_frame_dim` / `amp_discr_hidden_dims` if present, AMP file **count** (paths only if parent supplied a manifest)
- `success` is true only when all of: `get_cfgs_consumed`, checkpoint file exists with SHA256, and eval returned a dict containing `completed_episodes` **or** play's `--record` path exists as a file. `--help` / `SystemExit(0)` / missing outputs are `success: false` and `exit_code: 1`.
- `status` / `exception_type` / `exception_message` / `exit_code`

G1 run `params/env.yaml` and the remote full17 datapaths are **not** filled in.

## Tests

Command (local CPU, no Isaac, no torch):

```
D:\anaconda\envs\pytorch\python.exe -m pytest tests/test_milestone_evidence_wrapper.py -v --basetemp=artifacts/z2_migration/pytest-tmp
```

**16 passed in 0.07s** (Python 3.12.13, pytest 9.1.1).

| Test | What it proves |
| --- | --- |
| `test_wrapper_flags_are_stripped_and_g1_flags_stay_in_downstream` | Wrapper flags are not forwarded; leftover `--g1_*` stay in the parsed remainder and are then rejected. |
| `test_amp_manifest_alias_is_wrapper_only_not_forwarded` | `--amp_expert_manifest` is consumed by the wrapper, not passed to play/eval. |
| `test_profile_is_applied_before_get_cfgs_return_is_consumed` | Uses the **captured frozen** `apply_vital_motion_v1`; caller sees VITAL fields and seed 42 before env construction. |
| `test_registry_profile_sets_seed_without_vital_mutation` | Z2/registry does not call VITAL; seed 42 still applied. |
| `test_vital_v1_rejects_non_g1_before_mutation` | Captured helper raises `G1-specific`; gait/action_rate/seed left unchanged. |
| `test_unknown_profile_is_rejected` | Only `registry` / `vital_v1`. |
| `test_runpy_milestone_import_skips_main` | `run_name='milestone_import'` does not execute `__main__`; `__main__` would. |
| `test_train_amp_manifest_plumbing_matches_clip_sha256` | train.py clip hash loop: match, mismatch, empty clips. |
| `test_evidence_manifest_records_argv_seed_and_exception` | Manifest records seed 42, exception, stripped downstream argv, source hashes. |
| `test_captured_frozen_g1_apply_vital_motion_v1_fields` | Loads **exact** `formal_v1/g1_reference/motion_experiment.py` (no `apply_vital_motion_experiment`) and checks termination / gait / action_rate / max_radial / progress_monitor / speed scale. |
| `test_loader_uses_exact_captured_helper_from_project_layout` | Production `{project-root}/legged_lab/envs/g1/motion_experiment.py` loader with a byte-identical copy of the captured file. |
| `test_local_helper_still_exports_apply_vital_motion_v1` | Current local G1 compatibility wrapper still works as one-arg `apply_vital_motion_v1`. |
| `test_wrapper_rejects_g1_flags_before_simulation` | `--g1_motion_experiment` and `--g1_progress_ab=B` raise before sim; ordinary flags do not. |
| `test_rejects_seed_other_than_42_including_equals_form` | Allows omitted / `42` / `--seed=42`; rejects `43`, `--seed=43`, `--seed -1`, `--seed=-1`. |
| `test_success_requires_get_cfgs_checkpoint_and_eval_or_record` | Success needs consumed get_cfgs + checkpoint SHA + eval `completed_episodes` JSON or an existing play record; `SystemExit(0)` is not success. |
| `test_build_manifest_help_exit_is_not_success` | Manifest `success=false`, `exit_code=1` for help-style SystemExit(0). |

Isolated pre-commit on authored files: **pass** (black, flake8 via hook, isort, license, codespell). Direct `python -m flake8` in the pytorch env is unavailable (`No module named flake8`); the pre-commit flake8 hook is the project checker.

Runtime Isaac eval/play, checkpoint SHA on nubot, G1 `env.yaml` comparison, and 1000/2000/3000 behavior JSON/mp4 remain **parent** work.

## Parent usage (not executed here)

Do **not** pass `--g1_motion_experiment` / `--g1_progress_ab`. The wrapper rejects them before simulation (frozen G1 eval/play would drop them). Use `--profile vital_v1`. Omit `--seed` or pass `--seed 42` only.

Z2 registry eval (this tree / frozen `933e08a` runtime):

```
python work/z2-migration/run-milestone-evidence.py \
  --mode eval --profile registry \
  --project-root <this-or-nubot-z2-checkout> \
  --evidence-manifest <abs>/z2_eval.evidence.json \
  --task z2_loco_teacher \
  --load_run 2026-09-09_03-45-12_z2_source_usd_teacher_v1 \
  --checkpoint <abs>/model_1000.pt \
  --output <abs>/z2_eval.json \
  --terrain_type flat --command_vx <shared> --num_envs <n> --episodes <n>
```

G1 current baseline (`vital_v1`), from a **G1 frozen checkout**, not this Z2 `g1_loco_teacher` registry unless that is actually the training source:

```
python work/z2-migration/run-milestone-evidence.py \
  --mode eval --profile vital_v1 \
  --project-root <G1-frozen-checkout> \
  --evidence-manifest <abs>/g1_eval.evidence.json \
  --amp-expert-manifest <parent-provided training manifest> \
  --task g1_loco_teacher \
  --load_run 2026-09-08_16-25-01_vital_motion_v1 \
  --checkpoint <abs>/model_1000.pt \
  --output <abs>/g1_eval.json \
  --terrain_type flat --command_vx <same> --num_envs <same> --episodes <same>
```

Play / continuous replay (existing `--record` writes mp4 + diagnostics json):

```
python work/z2-migration/run-milestone-evidence.py \
  --mode play --profile registry|vital_v1 \
  --project-root <checkout> \
  --evidence-manifest <abs>/play.evidence.json \
  --task <z2_loco_teacher|g1_loco_teacher> \
  --checkpoint_path <abs>/model_1000.pt \
  --record <abs>/replay.mp4 --duration 12 \
  --terrain --terrain_types <name> --command_vx <same>
```

`--load_run` remains required by eval even when `--checkpoint` is an absolute file. That is existing eval argparse, not a wrapper invention.

## Limitations (not claimed)

1. **No Isaac run.** Wrapper + stubs only. Parent must run eval JSON and play mp4 on the Isaac runtime (`scripts/nubot_run.sh`) and bind checkpoint SHA / lineage.
2. **No guessed G1 full17 paths and no guessed `params/env.yaml`.** This Z2 tree’s G1 registry AMP dir is `motion_amp_expert_unitree_v5` (6 clips), which is **not** asserted to be the GPU1/3 vital run’s training mixture. If AmpOnPolicyRunner needs the training clips to construct, parent passes `--amp-expert-manifest` pointing at that run’s real manifest.
3. **`--g1_*` is rejected before simulation.** Frozen G1 eval/play would silently drop it; new Z2 eval would raise later. Parent must use `--profile vital_v1` and never pass train leftovers.
4. **`--seed` other than 42 is rejected** so play cannot override the snapshot. GIF fallback instead of the specified `--record` path is not success.
5. **Play gym/vault path** (`task` not in `task_registry.train_cfgs`) never calls `get_cfgs`; the wrap would not apply. Locomotion tasks `g1_loco_teacher` / `z2_loco_teacher` do.
6. **chdir to `--project-root`.** Relative `--output` / `--record` / checkpoints resolve there. Prefer absolute paths.
7. **Actor/AMP dims in the evidence JSON are cfg fields**, not live observation widths from a constructed env.
8. **Not a job scheduler.** One process, one eval or play, one manifest. 1000/2000/3000 looping, tmux, and G1/Z2 pairing stay with parent.
9. **Wrapper is not in frozen `933e08a`.** Parent must copy/use this file when launching on nubot. Training code on the live tmux job must stay untouched.
10. **README G1 table cell** still describes an older GPU0/2 G1 story. Left unchanged per “no other stale unrelated rows”.

## Non-goals honored

- No SSH / no training restart / no recipe hotpatch.
- Existing `work/` and `artifacts/` untracked files were not deleted.
- No commit.
