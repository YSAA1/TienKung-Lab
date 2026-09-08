#!/usr/bin/env bash
set -u
cd /home/nubot/phn_ws/t4_train/TienKung-Lab-g1-core-fixes-20260908
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
timeout --signal=INT --kill-after=30s 600s bash scripts/nubot_run.sh /home/nubot/phn_ws/t4_train/TienKung-Lab-g1-core-fixes-20260908/artifacts/portability/g1_core_fixes_20260908/acceptance/server_probe.py --task g1_loco_teacher --terrain mixed --num_envs 32 --steps 500 --pulse_steps 20 --pulse_amplitude 0.1 --output /home/nubot/phn_ws/t4_train/TienKung-Lab-g1-core-fixes-20260908/artifacts/portability/g1_core_fixes_20260908/acceptance/g1_mixed.json --headless --device cuda:0 --g1_progress_ab B --amp_expert_manifest /home/nubot/phn_ws/t4_train/TienKung-Lab-g1-core-fixes-20260908/legged_lab/envs/g1/datasets/motion_amp_expert_t4_rob2rob_full17_v1/_manifest.json > /home/nubot/phn_ws/t4_train/TienKung-Lab-g1-core-fixes-20260908/artifacts/portability/g1_core_fixes_20260908/acceptance/g1_mixed.log 2>&1
code=$?
printf "%s\n" "$code" > /home/nubot/phn_ws/t4_train/TienKung-Lab-g1-core-fixes-20260908/artifacts/portability/g1_core_fixes_20260908/acceptance/g1_mixed.exit
timeout --signal=INT --kill-after=30s 600s bash scripts/nubot_run.sh /home/nubot/phn_ws/t4_train/TienKung-Lab-g1-core-fixes-20260908/artifacts/portability/g1_core_fixes_20260908/acceptance/server_probe.py --task t4_loco_teacher_sparse --terrain mixed --num_envs 32 --steps 500 --pulse_steps 20 --pulse_amplitude 0.1 --output /home/nubot/phn_ws/t4_train/TienKung-Lab-g1-core-fixes-20260908/artifacts/portability/g1_core_fixes_20260908/acceptance/t4_mixed.json --headless --device cuda:0 > /home/nubot/phn_ws/t4_train/TienKung-Lab-g1-core-fixes-20260908/artifacts/portability/g1_core_fixes_20260908/acceptance/t4_mixed.log 2>&1
code=$?
printf "%s\n" "$code" > /home/nubot/phn_ws/t4_train/TienKung-Lab-g1-core-fixes-20260908/artifacts/portability/g1_core_fixes_20260908/acceptance/t4_mixed.exit
timeout --signal=INT --kill-after=30s 600s bash scripts/nubot_run.sh /home/nubot/phn_ws/t4_train/TienKung-Lab-g1-core-fixes-20260908/artifacts/portability/g1_core_fixes_20260908/acceptance/server_probe.py --task g1_loco_teacher --terrain flat --num_envs 32 --steps 500 --pulse_steps 20 --pulse_amplitude 0.1 --output /home/nubot/phn_ws/t4_train/TienKung-Lab-g1-core-fixes-20260908/artifacts/portability/g1_core_fixes_20260908/acceptance/g1_flat.json --headless --device cuda:0 --g1_progress_ab B --amp_expert_manifest /home/nubot/phn_ws/t4_train/TienKung-Lab-g1-core-fixes-20260908/legged_lab/envs/g1/datasets/motion_amp_expert_t4_rob2rob_full17_v1/_manifest.json > /home/nubot/phn_ws/t4_train/TienKung-Lab-g1-core-fixes-20260908/artifacts/portability/g1_core_fixes_20260908/acceptance/g1_flat.log 2>&1
code=$?
printf "%s\n" "$code" > /home/nubot/phn_ws/t4_train/TienKung-Lab-g1-core-fixes-20260908/artifacts/portability/g1_core_fixes_20260908/acceptance/g1_flat.exit
date -Is > /home/nubot/phn_ws/t4_train/TienKung-Lab-g1-core-fixes-20260908/artifacts/portability/g1_core_fixes_20260908/acceptance/completed.txt
