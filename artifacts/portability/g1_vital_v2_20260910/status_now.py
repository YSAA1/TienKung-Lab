#!/usr/bin/env python3
from __future__ import annotations

import paramiko

HOST = "100.100.188.39"
G1 = "/home/nubot/phn_ws/t4_train/TienKung-Lab-g1-vital-v2-20260910/logs/g1_vital_motion/2026-09-10_01-05-50_vital_motion_v2"
Z2 = "/home/nubot/phn_ws/t4_train/TienKung-Lab-z2-reset-20260909/logs/z2_loco_teacher_sparse/2026-09-09_10-33-53_z2_reset_aligned_v1"


def run(client, cmd, timeout=25):
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = (stdout.read() + stderr.read()).decode("utf-8", "replace")
    code = stdout.channel.recv_exit_status()
    print(f"===== {cmd[:80]} exit {code} =====", flush=True)
    print(out[:3500], flush=True)
    return out


def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print("connecting", flush=True)
    client.connect(HOST, username="nubot", password=" ", timeout=20)
    print("connected", flush=True)
    run(client, "date")
    run(client, "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv")
    run(client, "tmux ls || true")
    run(
        client,
        "ps -eo pid,etime,cmd | grep -E 'train.py|tensorboard.main' | grep -v grep | sed -E 's/ +/ /g'",
    )
    run(
        client,
        f"ls -lt {G1}/model_*.pt 2>/dev/null | head -n 5; echo ---; "
        f"stat -c '%y %s' {G1}/events.out.tfevents* 2>/dev/null | tail -n 1",
    )
    run(
        client,
        f"ls -lt {Z2}/model_*.pt 2>/dev/null | head -n 3; echo ---; "
        f"stat -c '%y %s' {Z2}/events.out.tfevents* 2>/dev/null | tail -n 1",
    )
    run(client, "ss -tlnp | grep -E ':(8045|8050|8043)\\s' || echo no-tb-ports")
    client.close()


if __name__ == "__main__":
    main()
