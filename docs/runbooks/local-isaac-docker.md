# 本机 Isaac play Docker

拉下来的 G1/G2/loco checkpoint 要在和 zhuoqun **同一套**运行时里 play：Isaac Sim 5.1 standalone + IsaacLab **2.1.0** + 仓库内 `rsl_rl`。本机 conda `env_isaaclab` 是 IsaacLab 2.3.2 + torch 2.7，对不上，不要用它当正式 play。

## 一次安装

不要从训练机 rsync：Tailscale 只有几十 KB/s。脚本走国内能到的源：

- IsaacLab v2.1.0：`ghfast.top` 代理 GitHub
- Isaac Sim 5.1（8.2 GiB）：NVIDIA 官方 zip + `aria2c` 16 线程（没有国内整包镜像）

```bash
tmux new -s t4-setup-isaac
bash scripts/setup_local_isaac_docker.sh
```

docker 组还需要一次 sudo（`nvidia-container-toolkit` 已装可跳过安装那几行）：

```bash
sudo usermod -aG docker "$USER"
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
newgrp docker
bash scripts/setup_local_isaac_docker.sh
```

用户已在 `docker` 组时，当前 shell 若还没吃到组（`docker info` 报 permission denied），`setup_local_isaac_docker.sh` 和 `local_run.sh` 会自己 `sg docker` 再跑，不必新开终端。`local_run.sh` 不继承桌面/conda 的 `ISAACLAB_ROOT`（那边常是 2.3.2）；要用别的树就设 `T4_ISAACLAB_ROOT` / `T4_ISAACSIM_ROOT`。本机默认 runtime 是 `runc`，脚本会显式 `--runtime nvidia` 并挂上 host 的 `nvidia_icd.json`，否则容器里 Vulkan 只看得到 llvmpipe，Isaac GPU foundation 起不来。

## 播模型

```bash
scripts/local_run.sh legged_lab/scripts/play.py --task t4_vault_mimic --num_envs=1 \
  --load_run 2026-08-13_09-30-13 --checkpoint model_28500.pt
```

旧学生 0.30 PPO 微调也走同一容器（本机 1 卡，tmux）：

```bash
tmux new -s t4-student-ft
scripts/local_run.sh legged_lab/scripts/train_t4_depth_student_ft.py --headless \
  --student_checkpoint artifacts/checkpoints/nubot/t4_loco_depth_student_stage_s_head35/model_24999.pt \
  --task_num_envs 32 --rendering_mode performance --run_name hurdle030_ft --logger tensorboard --max_iterations 8000
```

本机约 30 GiB RAM + 桌面占用 GPU。深度相机 128 env 会把 RTX descriptor 打满并卡住；先 32 + `performance` 渲染。不要和本机 teacher 训练或批量渲染并行。

MuJoCo 诊断回放不走这套 Docker：`PYTHONPATH=. python legged_lab/scripts/play_t4_vault_g1_mujoco.py`。
