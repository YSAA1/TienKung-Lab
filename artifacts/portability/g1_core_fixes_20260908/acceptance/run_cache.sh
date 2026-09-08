#!/usr/bin/env bash
cd /home/nubot/phn_ws/t4_train/TienKung-Lab-g1-core-fixes-20260908
export CUDA_VISIBLE_DEVICES=0
export PYTHONDONTWRITEBYTECODE=1
timeout --signal=INT --kill-after=30s 300s bash scripts/nubot_run.sh /home/nubot/phn_ws/t4_train/TienKung-Lab-g1-core-fixes-20260908/artifacts/portability/g1_core_fixes_20260908/acceptance/reset_cache_probe.py --headless --device cuda:0 --output /home/nubot/phn_ws/t4_train/TienKung-Lab-g1-core-fixes-20260908/artifacts/portability/g1_core_fixes_20260908/acceptance/reset_cache.json > /home/nubot/phn_ws/t4_train/TienKung-Lab-g1-core-fixes-20260908/artifacts/portability/g1_core_fixes_20260908/acceptance/reset_cache.log 2>&1
printf "%s\n" "$?" > /home/nubot/phn_ws/t4_train/TienKung-Lab-g1-core-fixes-20260908/artifacts/portability/g1_core_fixes_20260908/acceptance/reset_cache.exit

timeout --signal=INT --kill-after=30s 300s bash scripts/nubot_run.sh /home/nubot/phn_ws/t4_train/TienKung-Lab-g1-core-fixes-20260908/artifacts/portability/g1_core_fixes_20260908/acceptance/reset_cache_probe.py --task t4_loco_teacher_sparse --headless --device cuda:0 --output /home/nubot/phn_ws/t4_train/TienKung-Lab-g1-core-fixes-20260908/artifacts/portability/g1_core_fixes_20260908/acceptance/t4_reset_cache.json > /home/nubot/phn_ws/t4_train/TienKung-Lab-g1-core-fixes-20260908/artifacts/portability/g1_core_fixes_20260908/acceptance/t4_reset_cache.log 2>&1
printf "%s\n" "$?" > /home/nubot/phn_ws/t4_train/TienKung-Lab-g1-core-fixes-20260908/artifacts/portability/g1_core_fixes_20260908/acceptance/t4_reset_cache.exit
