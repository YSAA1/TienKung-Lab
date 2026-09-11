set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
case "$PWD" in
  *TienKung-Lab-z2-vital-v31-20260912) ;;
  *) echo "refusing to run outside the isolated z2 v31 worktree: $PWD" >&2; exit 1 ;;
esac
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,3
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
isaac_nv="$HOME/isaac-sim-standalone-5.1.0-linux-x86_64/kit/python/lib/python3.11/site-packages/nvidia"
export LD_LIBRARY_PATH="$isaac_nv/nvjitlink/lib:$isaac_nv/cusparse/lib:${LD_LIBRARY_PATH:-}"
mkdir -p artifacts/z2_migration/vital_v31
printf '%s\n' "$$" > artifacts/z2_migration/vital_v31/launcher.pid
bash scripts/nubot_run.sh -m torch.distributed.run --nproc_per_node=2 --master_port=29661 \
  legged_lab/scripts/train.py --task z2_loco_teacher --z2_motion_experiment z2_vital_v31 \
  --num_envs 2048 --seed 42 --distributed --headless --max_iterations 30000 \
  --experiment_name z2_loco_teacher_sparse --run_name z2_vital_v31_curated_v2 \
  --amp_expert_manifest "$PWD/legged_lab/envs/z2/datasets/motion_amp_expert_z2_v2/_manifest.json" \
  > artifacts/z2_migration/vital_v31/train.log 2>&1
training_exit=$?
printf '%s\n' "$training_exit" > artifacts/z2_migration/vital_v31/exit_code.txt
exit "$training_exit"
