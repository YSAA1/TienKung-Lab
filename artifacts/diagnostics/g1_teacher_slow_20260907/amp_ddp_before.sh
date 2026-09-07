set -euo pipefail
root=/home/nubot/phn_ws/t4_train/TienKung-Lab-g1-portability-20260906
out=/tmp/g1_teacher_audit_20260907
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1
bash "$root/scripts/nubot_run.sh" -m torch.distributed.run --nproc_per_node=2 --master_port=29602 \
  "$out/reproduce_amp_distributed.py" "$out/amp_ddp_before.json" >"$out/amp_ddp_before.log" 2>&1
