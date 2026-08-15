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

## 播模型

```bash
scripts/local_run.sh legged_lab/scripts/play.py --task t4_vault_mimic --num_envs=1 \
  --load_run 2026-08-13_09-30-13 --checkpoint model_28500.pt
```

MuJoCo 诊断回放不走这套 Docker：`PYTHONPATH=. python legged_lab/scripts/play_t4_vault_g1_mujoco.py`。
