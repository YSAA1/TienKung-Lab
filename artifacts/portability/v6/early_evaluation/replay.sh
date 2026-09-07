#!/bin/bash
set -euo pipefail
cd /home/nubot/phn_ws/t4_train/TienKung-Lab-g1-unitree-v6-20260907
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/home/nubot/phn_ws/t4_train/TienKung-Lab-g1-unitree-v6-20260907:${PYTHONPATH:-}"
isaac_nv=/home/nubot/isaac-sim-standalone-5.1.0-linux-x86_64/kit/python/lib/python3.11/site-packages/nvidia
export LD_LIBRARY_PATH="$isaac_nv/nvjitlink/lib:$isaac_nv/cusparse/lib:${LD_LIBRARY_PATH:-}"

set -euo pipefail
exec > /home/nubot/phn_ws/t4_train/TienKung-Lab-g1-unitree-v6-20260907/artifacts/portability/v6/early_evaluation/replay_supervisor.log 2>&1

timeout -k 15s 480s bash scripts/nubot_run.sh legged_lab/scripts/play.py --task g1_loco_teacher --num_envs 1 --seed 42 --checkpoint_path /home/nubot/phn_ws/t4_train/TienKung-Lab-g1-unitree-v6-20260907/logs/g1_loco_teacher_sparse/2026-09-07_13-55-07_g1_unitree_29dof_teacher_v6/model_500.pt --command_vx .7 --duration 16 --headless --terrain --terrain_types flat --difficulty 0 --record /home/nubot/phn_ws/t4_train/TienKung-Lab-g1-unitree-v6-20260907/artifacts/portability/v6/early_evaluation/model_500/flat.mp4 > /home/nubot/phn_ws/t4_train/TienKung-Lab-g1-unitree-v6-20260907/artifacts/portability/v6/early_evaluation/model_500/flat_replay.log 2>&1
timeout -k 15s 480s bash scripts/nubot_run.sh legged_lab/scripts/play.py --task g1_loco_teacher --num_envs 1 --seed 42 --checkpoint_path /home/nubot/phn_ws/t4_train/TienKung-Lab-g1-unitree-v6-20260907/logs/g1_loco_teacher_sparse/2026-09-07_13-55-07_g1_unitree_29dof_teacher_v6/model_500.pt --command_vx .7 --duration 16 --headless --terrain --terrain_types stepping_stones --difficulty 0 --record /home/nubot/phn_ws/t4_train/TienKung-Lab-g1-unitree-v6-20260907/artifacts/portability/v6/early_evaluation/model_500/stepping_stones.mp4 > /home/nubot/phn_ws/t4_train/TienKung-Lab-g1-unitree-v6-20260907/artifacts/portability/v6/early_evaluation/model_500/stepping_stones_replay.log 2>&1
date -Is > /home/nubot/phn_ws/t4_train/TienKung-Lab-g1-unitree-v6-20260907/artifacts/portability/v6/early_evaluation/model_500/completed.txt
date -Is > /home/nubot/phn_ws/t4_train/TienKung-Lab-g1-unitree-v6-20260907/artifacts/portability/v6/early_evaluation/completed.txt
